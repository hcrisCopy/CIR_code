"""Full-gallery audit of direct modality contributions and counterfactual candidate priors."""

import argparse
import math
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES, DATA, ROOT, read
from baseline.model import Encoder, checkpoint_paths
from baseline.runtime import load, load_cache, run_key, save, write_json

OUT = ROOT / "results/breadth"
FAMILIES = {"branch": [1, 2], "prior": [3], "conditional": [3, 4, 5], "combined": [1, 2, 3, 4, 5]}
TEMP = 32.0


def cfg(ds):
    return SimpleNamespace(dataset=ds, backbone="vit_cir", layer=6, limit=0)


def null_scores(vectors, gallery):
    sims = (vectors @ gallery.T) * TEMP
    return (sims.logsumexp(-2) - math.log(vectors.shape[-2])) / TEMP


def crossed(enc, ref, text):
    batch, count, dim = ref.shape
    vectors = enc.compose(ref.reshape(-1, dim), text.reshape(-1, dim))
    return vectors.reshape(batch, count, dim)


def ranks(scores, target):
    v = scores.gather(1, target[:, None])
    before = torch.arange(scores.shape[1], device=scores.device) < target[:, None]
    return (1 + ((scores > v) | ((scores == v) & before)).sum(1)).cpu()


def bank(args, cat):
    meta, images, queries = load_cache(args, "train", cat)
    gen = torch.Generator().manual_seed(42)
    ix = {name: i for i, name in enumerate(meta["names"])}
    ids = torch.randperm(len(meta["queries"]), generator=gen)[:512].tolist()
    refs = torch.tensor([ix[meta["queries"][i]["reference"]] for i in ids])
    return images["global"][refs].cuda(), queries["text"][ids].cuda(), queries["base"][ids].cuda()


def train_fit(blocks):
    # Restricted same-budget linear calibrations. Each row contains the positive then mined foils.
    pool = torch.cat(blocks).cuda()
    output = {}
    for name, cols in FAMILIES.items():
        w = torch.zeros(len(cols), device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.05)
        for _ in range(100):
            coef = w.tanh()
            logits = pool[:, :, 0] + pool[:, :, cols] @ coef
            loss = -(logits / 0.04).log_softmax(1)[:, 0].mean() + 0.012 * coef.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weights = [1.0] + [0.0] * 5
        for col, value in zip(cols, w.tanh().detach().tolist()):
            weights[col] = value
        output[name] = weights
    return output


@torch.no_grad()
def run_category(args, cat, enc, split, banks, weights, count):
    meta, images, qf = load_cache(args, split, cat)
    index = {name: i for i, name in enumerate(meta["names"])}
    queries = meta["queries"]
    gen = torch.Generator().manual_seed(42)
    ids = (
        torch.randperm(len(queries), generator=gen)[:count].tolist()
        if split == "train"
        else list(range(len(queries)))
    )
    ri = torch.tensor([index[q["reference"]] for q in queries])
    ti = torch.tensor([index[q["target"]] for q in queries])
    raw = images["global"].cuda()
    gallery = F.normalize(raw)
    text = qf["text"].cuda()
    base = qf["base"].cuda()
    br, bt, bq = banks
    global_prior = null_scores(bq, gallery)
    kb = 32
    cr, ct = br[:kb], bt[:kb]
    global_shuffled = global_prior[torch.randperm(len(gallery), generator=gen).cuda()]
    groups = defaultdict(list)
    margin_records = []
    pools = []
    k = 1 if args.dataset == "cirr" else 10
    soft = {}
    if args.dataset == "cirr" and split == "train":
        ann = read(DATA / "cirr_dataset/cirr/captions/cap.rc2.train.json")
        soft = {
            str(q["pairid"]): [
                index[key] for key, v in q["target_soft"].items() if v > 0 and key in index
            ]
            for q in ann
        }
    for begin in tqdm(range(0, len(ids), 32), desc=f"{split}/{cat}"):
        batch = ids[begin : begin + 32]
        n = len(batch)
        tgt, refidx = ti[batch].cuda(), ri[batch].cuda()
        r, t = raw[refidx], text[batch]
        head = enc.head
        h = torch.cat(
            [head.text_projection_layer(t).relu(), head.image_projection_layer(r).relu()], 1
        )
        gate = head.dynamic_scalar(h)
        joint = head.output_layer(head.combiner_layer(h).relu())
        scale = (joint + gate * t + (1 - gate) * r).norm(dim=1, keepdim=True)
        full = base[batch] @ gallery.T
        direct_r = ((1 - gate) * r / scale) @ gallery.T
        direct_t = (gate * t / scale) @ gallery.T
        js = (joint / scale) @ gallery.T
        if (full - direct_r - direct_t - js).abs().max().item() > 2e-5:
            raise ValueError("Combiner branch scores do not reconstruct the cached baseline")
        rn = null_scores(
            crossed(enc, r[:, None].expand(-1, kb, -1), ct[None].expand(n, -1, -1)), gallery
        )
        tn = null_scores(
            crossed(enc, cr[None].expand(n, -1, -1), t[:, None].expand(-1, kb, -1)), gallery
        )
        parts = torch.stack(
            [
                full,
                direct_r,
                direct_t,
                global_prior.expand_as(full),
                rn - global_prior,
                tn - global_prior,
            ],
            -1,
        )

        def valid(score):
            score = score.clone()
            if args.dataset == "cirr":
                score.scatter_(1, refidx[:, None], -torch.inf)
            return score

        if split == "train":
            hard = valid(full)
            for j, idx in enumerate(batch):
                hard[j, soft.get(queries[idx]["id"], [])] = -torch.inf
            hard.scatter_(1, tgt[:, None], -torch.inf)
            neg = hard.topk(64).indices
            picks = torch.cat([tgt[:, None], neg], 1)
            pools.append(parts.gather(1, picks[:, :, None].expand(-1, -1, 6)).cpu())
            continue
        # K-th strongest other candidate is precisely the native pass/fail boundary.
        hard = valid(full)
        hard.scatter_(1, tgt[:, None], -torch.inf)
        rival = hard.topk(k).indices[:, -1]

        def margin(score):
            return score.gather(1, tgt[:, None]) - score.gather(1, rival[:, None])

        margin_records.append(
            torch.cat([margin(full), margin(direct_r), margin(direct_t), margin(js), gate], 1).cpu()
        )
        scores = {
            "base": full,
            "no_ref": full - direct_r,
            "no_text": full - direct_t,
            "joint_only": js,
            "direct_only": direct_r + direct_t,
            "ref_half": full - 0.5 * direct_r,
            "text_half": full - 0.5 * direct_t,
            "global_norm": full - global_prior,
            "global_shuffled": full - global_shuffled,
            "ref_null": full - 0.5 * (rn - global_prior),
            "text_null": full - 0.5 * (tn - global_prior),
            "both_null": full - 0.5 * (rn + tn - 2 * global_prior),
            "ref_plus_norm": full - 0.5 * (rn - global_prior) - global_prior,
            "clip_sum": F.normalize(r + t) @ gallery.T,
            "text_only": F.normalize(t) @ gallery.T,
        }
        for key, w in weights.items():
            scores[key] = parts @ full.new_tensor(w)
        for key, score in scores.items():
            groups[key].append(ranks(valid(score), tgt))
    if split == "train":
        return torch.cat(pools)
    result = {m: torch.cat(xs) for m, xs in groups.items()}
    prior = load(ROOT / "results/paired_edits" / run_key(args) / "val" / cat / "paired_data.pt")
    if prior["ids"] != [q["id"] for q in queries] or not torch.equal(
        prior["ranks"]["base"], result["base"]
    ):
        raise ValueError("Native query identities or baseline ranks differ from audited run")
    for name in ("chord034", "center05", "minus_ref", "plus_ref", "wrong_ref"):
        result[name] = prior["ranks"][name]
    folder = OUT / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    save(
        folder / f"{cat}.pt",
        dict(
            refs=[q["reference"] for q in queries],
            ids=prior["ids"],
            ranks=result,
            margins=torch.cat(margin_records),
            targets=prior["targets"],
        ),
    )
    print(
        cat,
        {m: round(100 * (v <= k).float().mean().item(), 2) for m, v in result.items()},
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-count", type=int, default=1200, help="Calibration queries per category")
    opt = p.parse_args()
    if opt.train_count < 1:
        p.error("Use a positive calibration sample")
    OUT.mkdir(parents=True, exist_ok=True)
    all_weights = {}
    for ds, cats in CATEGORIES.items():
        args = cfg(ds)
        enc = Encoder(checkpoint_paths(ds, "vit_cir"), "cuda", 6)
        pools = [
            run_category(args, c, enc, "train", bank(args, c), {}, opt.train_count) for c in cats
        ]
        all_weights[ds] = train_fit(pools)
        del enc
    write_json(
        OUT / "weights.json", dict(weights=all_weights, train_per_category=opt.train_count, seed=42)
    )
    for ds, cats in CATEGORIES.items():
        args = cfg(ds)
        enc = Encoder(checkpoint_paths(ds, "vit_cir"), "cuda", 6)
        weights = {
            ("own_" if source == ds else "transfer_") + m: w
            for source, ws in all_weights.items()
            for m, w in ws.items()
        }
        for c in cats:
            run_category(args, c, enc, "val", bank(args, c), weights, opt.train_count)
        del enc


if __name__ == "__main__":
    main()
