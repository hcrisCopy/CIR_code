"""Cross-fit selective use of the published BGE reranker on matched fixed candidate sets."""

import hashlib
import numpy as np
import duckdb

from .breadth import OUT
from .breadth_results import bootstrap
from .pinpoint_folds import files, fold, MODELS
from .pinpoint_ranks import FILES, PUBLIC, lookup, rank_evidence
from baseline.runtime import write_json


def main():
    truth = duckdb.sql(
        f"SELECT query_id,query_image_signature,positive_candidates,negative_candidates FROM read_parquet('{PUBLIC}/pinpoint_licensed.parquet') ORDER BY query_id"
    ).fetchall()
    base, teacher = map(files, [MODELS["BGE-VL-MLLM-S1"], MODELS["BGE-reranked"]])
    text, image = map(files, [FILES["text"], FILES["image"]])
    official = lookup()
    feature, delta, refs, ties = [], [], [], []
    for q, ref, pos, neg in truth:
        key = q.removeprefix("query_").zfill(5)
        b, h = [rows[key]["retrieved_items"] for rows in (base, teacher)]
        assert set(b) == set(h) and len(b) == 50
        x = []
        t, i = [rank_evidence(rows[key]["retrieved_items"]) for rows in (text, image)]
        for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50)]:
            tv, iv = [np.array([ss.get(c, 0) for c in b[lo:hi]]) for ss in (t, i)]
            x.extend(
                [
                    tv.mean(),
                    iv.mean(),
                    np.minimum(tv, iv).mean(),
                    (iv * (tv == 0)).mean(),
                    (tv * (iv == 0)).mean(),
                ]
            )
        feature.append(x)

        def measured(rank):
            return np.array(
                [
                    official.calculate_metrics_at_k(rank, pos, 10)["ap"],
                    official.calculate_neg_recall_at_k(rank, neg, 10),
                ]
            )

        delta.append(measured(h) - measured(b))
        refs.append(ref)
        ties.append(hashlib.sha256(key.encode()).hexdigest())
    x, y = np.array(feature), np.array(delta)
    ids = np.array([fold(ref) for ref in refs])
    utility = y[:, 0] - 0.25 * y[:, 1]
    result = {}
    for budget in (0.25, 0.5):
        methods = {
            n: np.zeros(len(x)) for n in ("random_expected", "image_no_text", "learned", "oracle")
        }
        for f in range(5):
            train = ids != f
            test = np.flatnonzero(~train)
            k = int(round(budget * len(test)))
            mu = x[train].mean(0)
            sd = x[train].std(0).clip(0.001)
            design = (x[train] - mu) / sd
            w = np.linalg.solve(
                design.T @ design + np.eye(x.shape[1]) * 0.05 * train.sum(),
                design.T @ (utility[train] - utility[train].mean()),
            )
            scores = {
                "learned": ((x[test] - mu) / sd) @ w,
                "image_no_text": (x[test, 3] + 4 * x[test, 8] + 5 * x[test, 13]) / 10,
                "oracle": utility[test],
            }
            methods["random_expected"][test] = k / len(test)
            for name, s in scores.items():
                order = np.lexsort((np.array(ties)[test], -s))
                methods[name][test[order[:k]]] = 1
        section = {}
        for name, mask in methods.items():
            gain = mask[:, None] * y
            control = (mask - methods["random_expected"])[:, None] * y
            est = bootstrap(np.column_stack([gain, control]), refs)
            lo, hi = np.quantile(est, [0.025, 0.975], axis=0)
            section[name] = dict(
                delta=gain.mean(0).tolist(),
                vs_random=control.mean(0).tolist(),
                ci_lo=lo.tolist(),
                ci_hi=hi.tolist(),
                coverage=float(mask.mean()),
            )
        result[str(budget)] = section
    write_json(
        OUT / "pinpoint_routing.json",
        dict(n=len(x), refs=len(set(refs)), full=y.mean(0).tolist(), budget=result),
    )


if __name__ == "__main__":
    main()
