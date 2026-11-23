"""Same-reference, two-instruction ranking and chroma interventions on natural annotations."""

import itertools
import re
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES, DATA, ROOT, read
from baseline.heads import read_head
from baseline.model import Encoder, checkpoint_paths
from baseline.runtime import Runtime, load_cache, parser, run_key, save, write_json

from .color_views import load_views
from .review import read_lines

COLOR = re.compile(
    r"\b(?:colou?rs?|colou?red|red|blue|green|yellow|purple|pink|orange|brown|black|white|gr[ae]y|navy|beige|maroon|burgundy|teal|turquoise|gold|silver)\b",
    re.I,
)
KEEP = re.compile(r"\b(?:same|similar|keep|matching)\s+(?:\w+\s+){0,2}colou?r\b", re.I)


def pairs_for(dataset, split, qs):
    # An alternative target for the same reference is an annotation-derived foil.
    # Exclude known soft positives, identical captions and identical targets.
    soft = {}
    if dataset == "cirr":
        ann = read(DATA / "cirr_dataset/cirr/captions" / f"cap.rc2.{split}.json")
        soft = {str(q["pairid"]): {k for k, v in q["target_soft"].items() if v > 0} for q in ann}
    groups = defaultdict(list)
    for i, q in enumerate(qs):
        groups[q["reference"]].append(i)
    pairs = []
    for ref, ids in groups.items():
        for i, j in itertools.combinations(ids, 2):
            a, b = qs[i], qs[j]
            if (
                a["target"] == b["target"]
                or a["text"].strip().lower() == b["text"].strip().lower()
                or a["target"] in soft.get(b["id"], ())
                or b["target"] in soft.get(a["id"], ())
            ):
                continue
            pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def cluster_stat(values, group):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, refs=0)
    ids, ix = np.unique(group, return_inverse=True)
    sums = np.bincount(ix, weights=values)
    counts = np.bincount(ix)
    gen = np.random.default_rng(42)
    draws = gen.integers(len(ids), size=(1000, len(ids)))
    ci = np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975])
    return dict(
        n=len(values), refs=len(ids), mean=float(values.mean()), lo=float(ci[0]), hi=float(ci[1])
    )


def chunks_compose(encoder, images, texts, device):
    return torch.cat(
        [
            encoder.compose(x.to(device), t.to(device)).cpu()
            for x, t in zip(images.split(128), texts.split(128), strict=True)
        ]
    )


def mean_edit_queries(encoder, references, texts, device):
    collected = []
    edits = texts.to(device)
    for batch in references.split(32):
        batch = batch.to(device)
        count, dim = len(batch), batch.shape[1]
        q = encoder.compose(
            batch[:, None].expand(-1, len(edits), -1).reshape(-1, dim),
            edits[None].expand(count, -1, -1).reshape(-1, dim),
        )
        collected.append(q.view(count, len(edits), dim).mean(1).cpu())
    return torch.cat(collected)


def stats_for_margins(margins, clean, refs):
    joint = (margins > 0).all(1)
    baseline = (clean > 0).all(1)
    return dict(
        joint=cluster_stat(joint.numpy(), refs),
        joint_delta=cluster_stat((joint.float() - baseline.float()).numpy(), refs),
        assignment=cluster_stat((margins.sum(1) > 0).numpy(), refs),
        one_only=float(((margins > 0).sum(1) == 1).float().mean()),
        individual=float((margins > 0).float().mean()),
        mean_edit_separation=float(margins.mean(1).mean()),
        mean_abs_shared_bias=float(((margins[:, 0] - margins[:, 1]) / 2).abs().mean()),
    )


def risk(m):
    return F.softplus(-m / 0.07)


@torch.inference_mode()
def analyze(args, rt, encoder, category):
    meta, im, qf = load_cache(args, args.split, category)
    views, altered = load_views(args, args.split, category)
    qs, names = meta["queries"], meta["names"]
    ids = {n: i for i, n in enumerate(names)}
    view_ids = {n: i for i, n in enumerate(views["names"])}
    ri = torch.tensor([ids[q["reference"]] for q in qs])
    vi = torch.tensor([view_ids[q["reference"]] for q in qs])
    ti = torch.tensor([ids[q["target"]] for q in qs])
    pairs = pairs_for(args.dataset, args.split, qs)
    if not len(pairs):
        raise ValueError("No eligible same-reference pairs")
    left, right = pairs.T
    ref = im["global"][ri]
    gallery = F.normalize(im["global"])
    t = qf["text"]
    clean = qf["base"]
    gray = chunks_compose(encoder, altered["global"][vi], t, rt.device)
    luma = chunks_compose(encoder, altered["luma"][vi], t, rt.device)
    other = torch.randperm(len(names), generator=torch.Generator().manual_seed(42))
    other[other == torch.arange(len(names))] = (other[other == torch.arange(len(names))] + 1) % len(
        names
    )
    wrong = chunks_compose(encoder, im["global"][other[ri]], t, rt.device)
    gen = torch.Generator().manual_seed(42)
    noise = torch.randn(clean.shape, generator=gen)
    tangent = F.normalize(noise - (noise * clean).sum(1, keepdim=True) * clean)
    cos = (clean * gray).sum(1, keepdim=True).clamp(-1, 1)
    random_q = cos * clean + (1 - cos.square()).clamp_min(0).sqrt() * tangent
    queries = {
        "base": clean,
        "text": F.normalize(t),
        "reference": F.normalize(ref),
        "clip_sum": F.normalize(ref + t),
        "gray": gray,
        "luma": luma,
        "random_direction": random_q,
        "wrong_ref": wrong,
        "minus_ref": F.normalize(clean - 0.25 * F.normalize(ref)),
        "plus_ref": F.normalize(clean + 0.25 * F.normalize(ref)),
    }
    for name in args.heads:
        headfile = ROOT / "checkpoints/diagnosis" / run_key(args) / name / "last.pt"
        head, source, _ = read_head(headfile, args, meta["weights"], rt.device)
        queries[name] = torch.cat(
            [
                head(x.to(rt.device), y.to(rt.device), z.to(rt.device)).cpu()
                for x, y, z in zip(
                    clean.split(256), im[source][ri].split(256), t.split(256), strict=True
                )
            ]
        )
    if args.operator:
        from .color_operator import checkpoint, compose_signed
        from baseline.runtime import load

        op = load(checkpoint(args))
        if op["encoder"] != meta["weights"][0] or op["dataset"] != args.dataset:
            raise ValueError("Incompatible color operator")
        queries["text_operator"] = compose_signed(t, clean, gray, op)
    identity = encoder.texts([""]).cpu().expand(len(qs), -1)
    q0 = chunks_compose(encoder, ref, identity, rt.device)
    if args.center_reference:
        # A label-free estimate of the score contribution shared across possible edits.
        _, train_images, train_queries = load_cache(args, "train", category)
        gen = torch.Generator().manual_seed(42)
        sample = torch.randperm(len(train_queries["text"]), generator=gen)[:32]
        anchors = train_queries["text"][sample]
        bank = train_images["global"]
        train_refs = bank[torch.randperm(len(bank), generator=gen)[:256]]
        center = mean_edit_queries(encoder, train_refs, anchors, rt.device).mean(0)
        unique, indexes = torch.unique(ri, return_inverse=True)
        bias = mean_edit_queries(encoder, im["global"][unique], anchors, rt.device) - center
        queries["center05"] = F.normalize(clean - 0.5 * bias[indexes])
        queries["center05_shuffled"] = F.normalize(
            clean - 0.5 * bias[torch.randperm(len(unique), generator=gen)][indexes]
        )
        empty_center = chunks_compose(
            encoder, train_refs, identity[:1].expand(len(train_refs), -1), rt.device
        ).mean(0)
        queries["empty05"] = F.normalize(clean - 0.5 * (q0 - empty_center))
        del train_images, train_queries, bank
    gallery_gpu = gallery.to(rt.device)
    pair_refs = np.array([qs[i]["reference"] for i in left.tolist()])
    color = torch.tensor([bool(COLOR.search(q["text"])) for q in qs])
    preserve = torch.tensor([bool(KEEP.search(q["text"])) for q in qs])
    first = gallery[ti[left]]
    second = gallery[ti[right]]
    delta = first - second
    p_ref = gallery[ri[left]]
    chord_direction = F.normalize(first - p_ref) - F.normalize(second - p_ref)

    def margins(q):
        return torch.stack(((q[left] * delta).sum(1), -(q[right] * delta).sum(1)), dim=1)

    clean_m = margins(clean)
    folder = ROOT / "results/paired_edits" / run_key(args) / args.split / category
    folder.mkdir(parents=True, exist_ok=True)
    out = {
        "queries": len(qs),
        "pairs": len(pairs),
        "view_energy_ratio_median": float(altered["ratio"].median()),
        "gray_unchanged_refs": int((altered["pixel_delta"] < 1e-5).sum()),
        "methods": {},
    }
    base_rank = None
    per_query = {}
    truth = read_lines(
        ROOT
        / "results/diagnosis"
        / run_key(args)
        / args.split
        / "base"
        / category
        / "rankings.jsonl"
    )
    for name in tqdm([*queries, "chord034"], desc=f"{category} paired rankings"):
        q = queries.get(name, clean)
        m = margins(q)
        if name == "chord034":
            dq = F.normalize(q - q0)
            m += 0.34 * torch.stack(
                ((dq[left] * chord_direction).sum(1), -(dq[right] * chord_direction).sum(1)), dim=1
            )
        rec = stats_for_margins(m, clean_m, pair_refs)
        ranks = []
        for a in range(0, len(q), 64):
            s = slice(a, a + 64)
            scores = q[s].to(rt.device) @ gallery_gpu.T
            if name == "chord034":
                # cos(q-q0, g-r) for unit g,r, without a [batch,gallery,dim] tensor.
                d = dq[s].to(rt.device)
                r = F.normalize(ref[s]).to(rt.device)
                denom = (2 - 2 * (r @ gallery_gpu.T)).clamp_min(1e-10).sqrt()
                extra = (d @ gallery_gpu.T - (d * r).sum(1, keepdim=True)) / denom
                extra.scatter_(1, ri[s, None].to(rt.device), 0.0)
                scores += 0.34 * extra
            if args.dataset == "cirr":
                scores.scatter_(1, ri[s, None].to(rt.device), -torch.inf)
            target = ti[s, None].to(rt.device)
            val = scores.gather(1, target)
            ranks.append(
                (
                    1
                    + (
                        (scores > val)
                        | ((scores == val) & (torch.arange(len(names), device=rt.device) < target))
                    ).sum(1)
                ).cpu()
            )
        ranks = torch.cat(ranks)
        if name == "base":
            base_rank = ranks
            gold = torch.tensor([r["rank"] for r in truth])
            if [r["id"] for r in truth] != [q["id"] for q in qs] or (gold - ranks).abs().max() > 1:
                raise ValueError("Independent ranking audit failed")
            out["rank_agreement"] = float((gold == ranks).float().mean())
        k = 1 if args.dataset == "cirr" else 10
        rec["recall"] = {str(x): float((ranks <= x).float().mean()) for x in (1, 5, 10, 50)}
        rec["recall_delta"] = cluster_stat(
            ((ranks <= k).float() - (base_rank <= k).float()).numpy(), [q["reference"] for q in qs]
        )
        out["methods"][name] = rec
        per_query[name] = ranks
        if name in ("gray", "luma", "random_direction"):
            n = risk(m) - risk(clean_m)
            selected = color[left] ^ color[right]
            contrast = torch.where(color[left], n[:, 0] - n[:, 1], n[:, 1] - n[:, 0])
            rec["color_mentioned_minus_unmentioned"] = cluster_stat(
                contrast[selected].numpy(), pair_refs[selected.numpy()]
            )
            rec["n_explicit_same_color"] = int(preserve.sum())
            rec["mean_risk_change"] = cluster_stat(n.mean(1).numpy(), pair_refs)
    write_json(folder / "summary.json", out)
    save(
        folder / "paired_data.pt",
        dict(
            pairs=pairs,
            ids=[x["id"] for x in qs],
            references=[x["reference"] for x in qs],
            targets=[x["target"] for x in qs],
            clean=clean_m,
            gray=margins(gray),
            luma=margins(luma),
            color=color,
            ranks=per_query,
        ),
    )
    if args.split == "train":
        # One fixed alternate per eligible query, selected without model scores.
        alts = defaultdict(list)
        for a, b in pairs.tolist():
            alts[a].append(b)
            alts[b].append(a)
        selected = sorted(alts)
        rng = np.random.default_rng(42)
        negative = ti[[rng.choice(alts[i]) for i in selected]]
        direction = gallery[ti[selected]] - gallery[negative]
        m0 = (clean[selected] * direction).sum(1)
        m1 = (gray[selected] * direction).sum(1)
        save(
            folder / "supervision.pt",
            dict(
                identity=meta["identity"],
                ids=selected,
                negative=negative,
                changed_base=gray[selected],
                changed_feature=altered["patch"][vi[selected]],
                target_risk=(risk(m1) - risk(m0)).clamp_min(0),
                temperature=0.07,
            ),
        )
    print(folder, "pairs", len(pairs), flush=True)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--heads", nargs="*", default=[])
    p.add_argument("--operator", action="store_true")
    p.add_argument("--center-reference", action="store_true")
    args = p.parse_args()
    if args.backbone != "vit_cir":
        p.error("Use the trained ViT Combiner")
    rt = Runtime()
    if rt.world != 1:
        raise ValueError("Run one analysis process per dataset")
    enc = Encoder(
        checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit), rt.device, args.layer
    )
    for cat in CATEGORIES[args.dataset]:
        analyze(args, rt, enc, cat)
    rt.close()
