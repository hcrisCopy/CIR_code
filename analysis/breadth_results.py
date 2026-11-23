"""Paired full-gallery estimates, validated strata, and aggregate hits from public ranks."""

import csv
import json
import zipfile
from zlib import crc32
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline.data import CATEGORIES, DATA, layout
from baseline.runtime import load, write_json
from .breadth import OUT
from .validation_slices import normalize


def bootstrap(values, refs):
    _, group = np.unique(refs, return_inverse=True)
    size = np.bincount(group)
    totals = np.zeros((len(size), values.shape[1]))
    np.add.at(totals, group, values)
    gen = np.random.default_rng(20261123 + crc32(str(refs[0]).encode()))
    draws = []
    for _ in range(20):
        count = gen.multinomial(len(size), np.ones(len(size)) / len(size), size=50)
        draws.append((count @ totals) / (count @ size)[:, None])
    return np.concatenate(draws)


def metrics(dataset, scope, rank_key, k, cohorts, rows):
    methods = sorted(set.intersection(*(set(x[rank_key]) for x, _ in cohorts)))
    baseline = "base"
    base_parts, diff_parts, boot_parts = [], [], []
    for x, pick in cohorts:
        hits = np.stack([x[rank_key][m].numpy()[pick] <= k for m in methods], 1)
        base = x[rank_key][baseline].numpy()[pick] <= k
        diffs = hits.astype(float) - base[:, None]
        for method, alt in [("clip_sum_boost_text", "clip_sum")]:
            if method in methods:
                diffs[:, methods.index(method)] = (
                    hits[:, methods.index(method)].astype(float) - hits[:, methods.index(alt)]
                )
        boot_parts.append(bootstrap(diffs, np.array(x["refs"])[pick]))
        diff_parts.append(diffs)
        base_parts.append(hits.mean(0))
    boot = np.mean(boot_parts, 0)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    for i, method in enumerate(methods):
        d = np.concatenate([v[:, i] for v in diff_parts])
        rows.append(
            dict(
                dataset=dataset,
                scope=scope,
                k=k,
                n=len(d),
                method=method,
                comparator="clip_sum" if method == "clip_sum_boost_text" else baseline,
                recall=100 * np.mean(base_parts, 0)[i],
                delta=100 * np.mean([v[:, i].mean() for v in diff_parts]),
                lo=100 * lo[i],
                hi=100 * hi[i],
                fixed=int((d > 0).sum()),
                broken=int((d < 0).sum()),
            )
        )


def native(rows):
    diagnostics = {}
    for ds, cats in CATEGORIES.items():
        k = 1 if ds == "cirr" else 10
        all_data = []
        lookup = defaultdict(list)
        for c in cats:
            x = load(OUT / ds / f"{c}.pt")
            _, q = layout(ds, c, "val")
            assert [item["id"] for item in q] == x["ids"]
            for i, item in enumerate(q):
                lookup[(item["reference"], normalize(item["text"]))].append((c, i))
            ix = np.arange(len(q))
            all_data.append((x, ix))
            metrics(ds, c, "ranks", k, [(x, ix)], rows)
            m = x["margins"].numpy()
            fail = x["ranks"]["base"].numpy() > k
            diagnostics[c] = dict(
                n=len(q),
                misses=int(fail.sum()),
                neg_ref=int((fail & (m[:, 1] < 0)).sum()),
                neg_text=int((fail & (m[:, 2] < 0)).sum()),
                text_for_ref_against=int((fail & (m[:, 1] < 0) & (m[:, 2] > 0)).sum()),
                text_only_recovers=int((fail & (x["ranks"]["text_only"].numpy() <= k)).sum()),
                image_only_recovers=int((fail & (x["ranks"]["image_only"].numpy() <= k)).sum()),
            )
        for other_k in sorted({1, 10, 50, k}):
            metrics(ds, "native", "ranks", other_k, all_data, rows)
        if ds == "cirr":
            metrics(ds, "group", "subset_ranks", 1, all_data, rows)
        seen = set()
        selections = []
        for name in ["solved", "unsolved"]:
            selected = defaultdict(list)
            p = DATA / "circus_slices" / f"mbeir_{ds}_task7_test_validated_{name}.jsonl"
            for line in p.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                hits = lookup[(Path(item["query_img_path"]).stem, normalize(item["query_txt"]))]
                if len(hits) == 1 and hits[0] not in seen:
                    c, i = hits[0]
                    seen.add((c, i))
                    selected[c].append(i)
            selections.append(selected)
        # Pool these matched queries, regardless of which external model could solve them.
        for name, selections_for_scope in [
            ("clean", selections),
            ("clean_solved", selections[:1]),
            ("clean_unsolved", selections[1:]),
        ]:
            merged = {"refs": [], "ranks": defaultdict(list)}
            for c, (x, _) in zip(cats, all_data):
                ix = [i for v in selections_for_scope for i in v[c]]
                merged["refs"].extend(f"{c}/{x['refs'][i]}" for i in ix)
                for method, ranks in x["ranks"].items():
                    merged["ranks"][method].append(ranks[ix])
            import torch

            merged["ranks"] = {m: torch.cat(v) for m, v in merged["ranks"].items()}
            for cut in sorted({k, 10}):
                metrics(ds, name, "ranks", cut, [(merged, np.arange(len(merged["refs"])))], rows)
    return diagnostics


def external(archive):
    out = {}
    with zipfile.ZipFile(archive) as z:
        for f in z.namelist():
            if not f.endswith(".json"):
                continue
            d = json.loads(z.read(f))
            meta = d["metadata"]
            ds = meta["dataset"]
            nmod = len(meta["aggregation"]["retrievers"])
            support = np.array(
                [
                    [
                        s["aggregation"]["retriever_support_at_k"][key]
                        for key in ("multimodal", "text_only", "image_only", "either_unimodal")
                    ]
                    for s in d["samples"]
                ]
            )
            mm, t, img, uni = support.T
            # Use the published per-modality hit flags: support counts exceed the
            # number of named retrievers in some rows of this release.
            modes = ("multimodal", "text_only", "image_only")
            flags = np.array([[s["hits_at_k"][m] for m in modes] for s in d["samples"]], dtype=bool)
            raw = np.array([[s["ranks"][m] for m in modes] for s in d["samples"]])
            assert np.array_equal(flags, raw <= meta["k"])
            mh, th, ih = flags.T
            uh = th | ih
            out[ds] = dict(
                queries=len(mm),
                named_models=nmod,
                k=meta["k"],
                support_exceeds_named=int((support > nmod).any(axis=1).sum()),
                any_uni=int(uh.sum()),
                mm_no_uni=int((mh & ~uh).sum()),
                none=int((~mh & ~uh).sum()),
                uni_no_mm=int((uh & ~mh).sum()),
                text_no_mm=int((th & ~mh).sum()),
                image_no_mm=int((ih & ~mh).sum()),
            )
    return out


def plot(rows):
    methods = [
        "ref_half",
        "global_norm",
        "own_branch",
        "transfer_branch",
        "boost_text",
        "late_text_025",
    ]
    names = [
        "Halve reference branch",
        "Global score normalization",
        "Two weights, in-domain",
        "Two weights, transferred",
        "Double text branch",
        "Add text-only score (0.25)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, layout="constrained")
    for ax, ds, k in zip(axes, ["cirr", "fashioniq"], [1, 10]):
        for n, (method, label) in enumerate(zip(methods, names)):
            res = next(
                r
                for r in rows
                if (r["dataset"], r["scope"], r["k"], r["method"]) == (ds, "native", k, method)
            )
            ax.plot([res["lo"], res["hi"]], [n, n], color="#25626f", linewidth=2)
            ax.scatter(res["delta"], n, color="#234756", s=33)
        ax.axvline(0, color="0.7", linestyle="--")
        ax.set(
            xlabel="Change in recall (percentage points)",
            title=f"{ds.upper()}  /  R@{k}",
            yticks=range(len(names)),
            yticklabels=names,
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.savefig(OUT / "calibration.png", dpi=190)
    plt.close(fig)


def main():
    rows = []
    diagnostics = native(rows)
    write_json(OUT / "diagnostics.json", diagnostics)
    public = OUT / "circus_public_ranks.zip"
    if public.exists():
        write_json(OUT / "external.json", external(public))
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
