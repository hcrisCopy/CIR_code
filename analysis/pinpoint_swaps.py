"""Check whether locally Pareto-improving swaps protect positives while rejecting negatives."""

import zlib

import duckdb
import numpy as np
from tqdm import tqdm

from baseline.runtime import write_json
from .breadth import OUT
from .breadth_results import bootstrap
from .pinpoint_folds import FEATURES, MODELS, construct, files, learn
from .pinpoint_ranks import FILES, PUBLIC, lookup, rank_evidence


def ordering(pred, rule, passes):
    order = list(range(len(pred)))
    # Disjoint adjacent pairs keep displacement to at most one position per pass.
    for sweep in range(passes):
        for j in range(sweep % 2, min(19, len(order) - 1), 2):
            a, b = order[j : j + 2]
            diff = pred[b] - pred[a]
            if rule == "both":
                yes = diff[0] > 0 and diff[1] < 0
            elif rule == "positive":
                yes = diff[0] > 0
            elif rule == "negative":
                yes = diff[1] < 0
            else:
                raise ValueError(rule)
            if yes:
                order[j : j + 2] = [b, a]
    return order


def main():
    truth = duckdb.sql(
        f"SELECT query_id,query_image_signature,positive_candidates,negative_candidates FROM read_parquet('{PUBLIC}/pinpoint_licensed.parquet') ORDER BY query_id"
    ).fetchall()
    official = lookup()
    text, image = [
        {k: rank_evidence(v["retrieved_items"]) for k, v in files(FILES[m]).items()}
        for m in ("text", "image")
    ]
    summary = {}
    for m in MODELS:
        stats, queries, _, refs = construct(m, truth, text, image)
        weights = {
            (family, f): learn(stats, FEATURES[family], f)
            for family in ("conditional", "additive")
            for f in range(5)
        }
        specs = [
            ("base", "conditional", "both", 0),
            ("both3", "conditional", "both", 3),
            ("both8", "conditional", "both", 8),
            ("positive3", "conditional", "positive", 3),
            ("negative3", "conditional", "negative", 3),
            ("additive3", "additive", "both", 3),
            ("shuffle3", "conditional", "both", 3),
        ]
        output = []
        for candidates, x, f, pos, neg in tqdm(queries, desc=m):
            pairs = []
            # Deterministic within-query evidence permutation has no access to annotations.

            gen = np.random.default_rng(zlib.crc32("|".join(candidates[:20]).encode()))
            for name, fam, rule, passes in specs:
                y = x[:20, FEATURES[fam]] @ weights[fam, f]
                if name == "shuffle3":
                    y = y[gen.permutation(len(y))]
                order = ordering(y, rule, passes)
                top = [candidates[j] for j in order[:10]]
                pairs.append(
                    [
                        official.calculate_metrics_at_k(top, pos, 10)["ap"],
                        official.calculate_neg_recall_at_k(top, neg, 10),
                        float(top != candidates[:10]),
                        float(set(top) != set(candidates[:10])),
                    ]
                )
            output.append(pairs)
        v = np.array(output)
        d = v - v[:, :1]
        boot = bootstrap(d.reshape(len(v), -1), refs).reshape(1000, len(specs), 4)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        summary[m] = {
            name: dict(
                mean=v[:, j].mean(0).tolist(),
                delta=d[:, j].mean(0).tolist(),
                lo=lo[j].tolist(),
                hi=hi[j].tolist(),
            )
            for j, (name, *_) in enumerate(specs)
        }
    write_json(OUT / "pinpoint_swaps.json", summary)


if __name__ == "__main__":
    main()
