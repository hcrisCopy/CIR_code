"""Exact cosine ranks; only CIRR excludes the reference from the gallery."""

import torch
import torch.nn.functional as F


def rank_batch(dataset, queries, combined, images, text, names, index, top_k):
    gallery = F.normalize(images)
    scores = F.normalize(combined) @ gallery.T
    ref_ids = [index[q["reference"]] for q in queries]
    if dataset == "cirr":
        scores[torch.arange(len(queries)), ref_ids] = -torch.inf
    targets = torch.tensor([index[q["target"]] for q in queries], device=scores.device)[:, None]
    target_scores = scores.gather(1, targets)
    ahead = (scores > target_scores) | (
        (scores == target_scores) & (torch.arange(len(names), device=scores.device) < targets)
    )
    ranks = (ahead.sum(1) + 1).tolist()
    eligible = len(names) - (dataset == "cirr")
    count = min(max(top_k, 2), eligible)
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :count].tolist()
    rows = []
    for i, query in enumerate(queries):
        ref = ref_ids[i]
        gt = index[query["target"]]
        ranked = order[i]
        neg = next(j for j in ranked if j != gt)
        top = ranked[:top_k]
        inspected = list(dict.fromkeys([gt, neg, *top]))
        values = torch.stack(
            (
                scores[i, inspected],
                gallery[inspected] @ gallery[ref],
                gallery[inspected] @ F.normalize(text[i], dim=0),
            ),
            dim=1,
        ).tolist()
        evidence = {
            names[j]: dict(score=s, reference_cos=r, text_cos=t)
            for j, (s, r, t) in zip(inspected, values)
        }
        row = dict(
            **query,
            rank=ranks[i],
            top=[names[j] for j in top],
            negative=names[neg],
            margin=evidence[names[gt]]["score"] - evidence[names[neg]]["score"],
            scores=evidence,
        )
        if dataset == "cirr":
            members = [index[n] for n in query["group"] if n != query["reference"]]
            row["subset_rank"] = int(ahead[i, members].sum()) + 1
        rows.append(row)
    return rows


def metrics(dataset, rows):
    keys = (
        {"rank": (1, 5, 10, 50), "subset_rank": (1, 2, 3)} if dataset == "cirr" else {"rank": (10, 50)}
    )
    return {
        ("Subset " if key == "subset_rank" else "") + f"R@{k}": 100
        * sum(row[key] <= k for row in rows)
        / len(rows)
        for key, cutoffs in keys.items()
        for k in cutoffs
    }
