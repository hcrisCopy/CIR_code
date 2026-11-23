"""Exact cosine ranks; only CIRR excludes the reference from the gallery."""

import torch
import torch.nn.functional as F


def rank_batch(dataset, queries, combined, images, text, names, index, top_k):
    gallery = F.normalize(images)
    scores = F.normalize(combined) @ gallery.T
    ref_ids = [index[q["reference"]] for q in queries]
    if dataset == "cirr":
        scores[torch.arange(len(queries)), ref_ids] = -torch.inf
    order = scores.argsort(dim=1, descending=True, stable=True).cpu()
    rows = []
    for i, query in enumerate(queries):
        ref = ref_ids[i]
        gt = index[query["target"]]
        ranked = order[i, : len(names) - (dataset == "cirr")].tolist()
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
            rank=ranked.index(gt) + 1,
            top=[names[j] for j in top],
            negative=names[neg],
            margin=evidence[names[gt]]["score"] - evidence[names[neg]]["score"],
            scores=evidence,
        )
        if dataset == "cirr":
            members = set(query["group"])
            subset = [names[j] for j in ranked if names[j] in members]
            row["subset_rank"] = subset.index(query["target"]) + 1
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
