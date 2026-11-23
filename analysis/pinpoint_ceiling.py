"""Using labels only for diagnosis, measure the value of perfectly filtering known negatives."""

import duckdb
import numpy as np
from tqdm import tqdm

from baseline.runtime import write_json
from .breadth import OUT
from .breadth_results import bootstrap
from .pinpoint_folds import MODELS, construct, files
from .pinpoint_ranks import FILES, PUBLIC, lookup, rank_evidence


def main():
    truth = duckdb.sql(
        f"SELECT query_id,query_image_signature,positive_candidates,negative_candidates FROM read_parquet('{PUBLIC}/pinpoint_licensed.parquet') ORDER BY query_id"
    ).fetchall()
    text, image = [
        {k: rank_evidence(v["retrieved_items"]) for k, v in files(FILES[m]).items()}
        for m in ("text", "image")
    ]
    official = lookup()
    output = {}
    for model in MODELS:
        _, queries, _, refs = construct(model, truth, text, image)
        values = []
        published = files(MODELS[model]).keys()
        coverage = np.zeros(4, dtype=int)
        for (items, x, _, pos, neg), qid in tqdm(
            zip(
                queries,
                [
                    str(q[0]).removeprefix("query_").zfill(5)
                    for q in truth
                    if str(q[0]).removeprefix("query_").zfill(5) in published
                ],
            ),
            total=len(queries),
            desc=model,
        ):
            ns = set(neg)
            known = np.array([c in ns for c in items])
            flag = (x[:, 3] == 0) & (x[:, 4] > 0)
            row = []
            for excluded in (np.zeros(len(items), bool), known & flag, known):
                top = [c for c, mask in zip(items, excluded) if not mask][:10]
                row.append(
                    [
                        official.calculate_metrics_at_k(top, pos, 10)["ap"],
                        official.calculate_neg_recall_at_k(top, neg, 10),
                    ]
                )
            ps = set(pos)
            prefix = set(items[:40])
            extra = [c for c in text[qid] if c not in prefix][:10]
            pool = (items[:40] + extra + [c for c in items[40:] if c not in extra])[:50]
            assert len(pool) == len(items) == 50 and len(set(pool)) == 50
            for j, seq in enumerate(
                [items[:10], items, pool, list(dict.fromkeys(items + list(text[qid])))]
            ):
                coverage[j] += bool(ps.intersection(seq))
            b = rank_evidence(items)
            for seq, oracle in [(items, True), (pool, True), (items, False), (pool, False)]:
                top = sorted(
                    seq,
                    key=lambda c: int(c in ps) if oracle else b.get(c, 0) + text[qid].get(c, 0),
                    reverse=True,
                )[:10]
                row.append(
                    [
                        official.calculate_metrics_at_k(top, pos, 10)["ap"],
                        official.calculate_neg_recall_at_k(top, neg, 10),
                    ]
                )
            values.append(row)
        v = np.array(values)
        d = v - v[:, :1]
        estimates = bootstrap(d.reshape(len(v), -1), refs).reshape(1000, v.shape[1], 2)
        lo, hi = np.quantile(estimates, [0.025, 0.975], axis=0)
        output[model] = {
            name: dict(
                value=v[:, j].mean(0).tolist(),
                delta=d[:, j].mean(0).tolist(),
                lo=lo[j].tolist(),
                hi=hi[j].tolist(),
                ap_improved=int((d[:, j, 0] > 0).sum()),
                n=len(v),
            )
            for j, name in enumerate(
                (
                    "base",
                    "oracle_flagged",
                    "oracle_all",
                    "oracle_positive50",
                    "oracle_mixed50",
                    "rank_fusion50",
                    "rank_fusion_mixed50",
                )
            )
        }
        output[model]["at_least_one_positive"] = dict(
            zip(["base10", "base50", "mixed50", "union"], coverage.tolist())
        )
    write_json(OUT / "pinpoint_ceiling.json", output)


if __name__ == "__main__":
    main()
