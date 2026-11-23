"""Fit frozen-feature readouts from within-reference edit differences and matched shuffled groups."""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES
from baseline.runtime import load, load_cache, save, write_json
from .breadth import OUT, cfg, ranks


def material(ds, cat, split):
    meta, images, qf = load_cache(cfg(ds), split, cat)
    g = F.normalize(images["global"].cuda())
    t = F.normalize(qf["text"].cuda())
    names = {name: i for i, name in enumerate(meta["names"])}
    qs = meta["queries"]
    ri, ti = [
        torch.tensor([names[q[field]] for q in qs], device="cuda")
        for field in ("reference", "target")
    ]
    return qs, g, t, ri, ti, qf["base"].cuda()


def design(t, r, interaction):
    return torch.cat([t, F.normalize(t * r)], 1) if interaction else t


def fit(ds, seed):
    blocks = defaultdict(list)
    targets = defaultdict(list)
    counts = {}
    gen = torch.Generator().manual_seed(seed)
    for cat in CATEGORIES[ds]:
        q, g, t, ri, ti, _ = material(ds, cat, "train")
        groups = defaultdict(dict)
        for i, item in enumerate(q):
            groups[item["reference"]].setdefault(item["target"], i)
        groups = [list(v.values()) for v in groups.values() if len(v) > 1]
        flat = [i for group in groups for i in group]
        n = len(flat)
        sizes = torch.tensor([len(v) for v in groups], device="cuda")
        group_index = torch.arange(len(groups), device="cuda").repeat_interleave(sizes)
        counts[cat] = dict(references=len(groups), queries=n)

        def center(v):
            totals = v.new_zeros(len(groups), v.shape[1]).index_add_(0, group_index, v)
            return v - (totals / sizes[:, None])[group_index]

        shuffled = torch.randperm(n, generator=gen).cuda()
        y = g[ti[flat]]
        for part in ("within", "shuffled"):
            ix = torch.arange(n, device="cuda") if part == "within" else shuffled
            targets[part].append(center(y[ix]))
            for mix in [False, True]:
                x = design(t[flat], g[ri[flat]], mix)
                blocks[part, mix].append(center(x[ix]))
    fitted = {}
    for (part, mix), xs in blocks.items():
        x = torch.cat(xs)
        y = torch.cat(targets[part])
        cov = x.T @ x
        cov.diagonal().add_(cov.diagonal().mean().clone())
        fitted[part, mix] = torch.linalg.solve(cov, x.T @ y)
    return fitted, counts


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    opt = p.parse_args()
    models, counts = {}, {}
    for ds in CATEGORIES:
        models[ds], counts[ds] = fit(ds, opt.seed)
    for ds, cats in CATEGORIES.items():
        for cat in tqdm(cats, desc=f"{ds} validation"):
            q, g, t, ri, ti, b = material(ds, cat, "val")
            prev = load(OUT / ds / f"{cat}.pt")
            assert prev["ids"] == [a["id"] for a in q]
            result = {}
            gen = torch.Generator().manual_seed(opt.seed)
            other_refs = ri[torch.randperm(len(q), generator=gen)]
            for src, reg in models.items():
                for (part, mix), w in reg.items():
                    for wrong in [False, True] if mix and src == ds else [False]:
                        vec = F.normalize(design(t, g[other_refs if wrong else ri], mix) @ w)
                        tag = f"{'own' if src == ds else 'cross'}_{part}_{'product' if mix else 'text'}"
                        if wrong:
                            tag += "_wrongref"
                        for alpha in (0.25, 0.5):
                            take = []
                            for start in range(0, len(q), 128):
                                sl = slice(start, start + 128)
                                s = (b[sl] + alpha * vec[sl]) @ g.T
                                if ds == "cirr":
                                    s.scatter_(1, ri[sl, None], -torch.inf)
                                take.append(ranks(s, ti[sl]))
                            result[f"{tag}_{int(alpha * 100)}"] = torch.cat(take)
            save(OUT / ds / f"{cat}_edits.pt", dict(ids=prev["ids"], ranks=result))
    write_json(
        OUT / "edit_training.json",
        dict(counts=counts, seed=opt.seed, ridge="mean covariance diagonal"),
    )


if __name__ == "__main__":
    main()
