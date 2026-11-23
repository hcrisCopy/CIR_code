"""Five reference-disjoint folds for learning the value of independent modality evidence."""

import hashlib
from collections import Counter
from itertools import product

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from baseline.runtime import write_json
from .breadth import OUT
from .breadth_results import bootstrap
from .pinpoint_ranks import PUBLIC, FILES, lookup, rank_evidence
import json

MODELS = {
    **{k: v for k, v in FILES.items() if k not in ("text", "image")},
    "BGE-reranked": "bge_vl_mllm_s1_retrieval_results_licensed_reranked",
    "GPT5-premerge": "gpt5_text_premerge_standardized",
}
FEATURES = {"rank": [0, 1, 2], "additive": [0, 1, 2, 3, 4], "conditional": list(range(9))}


def files(name):
    return json.loads((PUBLIC / (name + ".json")).read_text())


def fold(ref):
    return int.from_bytes(hashlib.sha256(ref.encode()).digest()[:4], "big") % 5


def construct(model, truth, text, image):
    ranking = files(MODELS[model])
    stats = np.zeros((5, 9, 9))
    outcomes = np.zeros((5, 9, 2))
    queries = []
    counts = Counter()
    for qid, ref, pos, neg in truth:
        k = str(qid).removeprefix("query_").zfill(5)
        candidates = ranking[k]["retrieved_items"][:50]
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate candidate for {model}/{k}")
        b = np.array(list(rank_evidence(candidates).values()))
        t, i = [np.array([values[k].get(c, 0) for c in candidates]) for values in (text, image)]
        x = np.column_stack(
            [np.ones(len(b)), b, b * b, t, i, np.minimum(t, i), t * i, i * (t == 0), t * (i == 0)]
        )
        ps, ns = set(pos), set(neg)
        y = np.array([[c in ps, c in ns] for c in candidates], dtype=float)
        f = fold(ref)
        stats[f] += x.T @ x
        outcomes[f] += x.T @ y
        queries.append((candidates, x, f, pos, neg))
        suspect = (i[:10] > 0) & (t[:10] == 0)
        counts["any_image_no_text"] += int(suspect.any())
        counts["any_explicit_negative"] += int(y[:10, 1].any())
        counts["any_image_no_text_negative"] += int((suspect & y[:10, 1].astype(bool)).any())
        counts["any_image_no_text_positive"] += int((suspect & y[:10, 0].astype(bool)).any())
        counts["sum_shortlist_sizes"] += len(candidates)
    assert len(queries) == len(ranking)
    return (stats, outcomes), queries, dict(counts)


def learn(stats, fs, fold_id):
    cov, out = stats
    mat = (cov.sum(0) - cov[fold_id])[np.ix_(fs, fs)]
    rhs = (out.sum(0) - out[fold_id])[fs]
    # Label columns estimate observed positive and explicit negative utility separately.
    reg = np.diag(np.maximum(np.diag(mat), 1)) * 0.001
    return np.linalg.solve(mat + reg, rhs)


def main():
    truth = duckdb.sql(
        f"SELECT query_id,query_image_signature,positive_candidates,negative_candidates FROM read_parquet('{PUBLIC}/pinpoint_licensed.parquet') ORDER BY query_id"
    ).fetchall()
    refs = [row[1] for row in truth]
    text, image = [
        {k: rank_evidence(v["retrieved_items"]) for k, v in files(FILES[m]).items()}
        for m in ("text", "image")
    ]
    fits = {}
    queries = {}
    counts = {}
    for m in tqdm(MODELS, desc="Prepare public rankings"):
        fits[m], queries[m], counts[m] = construct(m, truth, text, image)
    official = lookup()
    out = {}
    for model in MODELS:
        specs = [("base", None, None, None)]
        for src, family, penalty in product(
            dict.fromkeys([model, "MetaCLIP2"]), FEATURES, (0.0, 0.25)
        ):
            specs.append(
                (
                    f"{'own' if src == model else 'transfer'}_{family}_{int(100 * penalty)}",
                    src,
                    family,
                    penalty,
                )
            )
        w = {
            (src, fam, f): learn(fits[src], cols, f)
            for src in dict.fromkeys([model, "MetaCLIP2"])
            for fam, cols in FEATURES.items()
            for f in range(5)
        }
        measures = []
        for candidates, x, f, pos, neg in tqdm(queries[model], desc=model):
            pairs = []
            for _, src, fam, penalty in specs:
                if src is None:
                    top = candidates[:10]
                else:
                    pred = x[:, FEATURES[fam]] @ w[src, fam, f]
                    s = pred[:, 0] - penalty * pred[:, 1]
                    top = [candidates[j] for j in np.argsort(-s, kind="stable")[:10]]
                pairs.append(
                    [
                        official.calculate_metrics_at_k(top, pos, 10)["ap"],
                        official.calculate_neg_recall_at_k(top, neg, 10),
                    ]
                )
            measures.append(pairs)
        v = np.array(measures)
        diff = v - v[:, :1]
        n = len(v)

        def intervals(d):
            return np.quantile(
                bootstrap(d.reshape(n, -1), refs).reshape(1000, *d.shape[1:]),
                [0.025, 0.975],
                axis=0,
            )

        lo, hi = intervals(diff)
        contrasts = {}
        names = [s[0] for s in specs]
        for a, b in [
            ("own_conditional_25", "own_additive_25"),
            ("own_conditional_25", "own_conditional_0"),
        ]:
            diff_ab = v[:, names.index(a)] - v[:, names.index(b)]
            low, high = intervals(diff_ab)
            contrasts[a + " vs " + b] = dict(
                delta=diff_ab.mean(0).tolist(), lo=low.tolist(), hi=high.tolist()
            )
        out[model] = dict(
            n=n,
            refs=len(set(refs)),
            counts=counts[model],
            metrics=["mAP@10", "NegRecall@10"],
            contrasts=contrasts,
            methods={
                name: dict(
                    value=v[:, j].mean(0).tolist(),
                    delta=diff[:, j].mean(0).tolist(),
                    lo=lo[j].tolist(),
                    hi=hi[j].tolist(),
                )
                for j, name in enumerate(names)
            },
        )
    write_json(OUT / "pinpoint_folds.json", out)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for metric, ax in enumerate(axes):
        for j, (name, color, label) in enumerate(
            [
                ("own_additive_25", "#788794", "Additive"),
                ("own_conditional_25", "#137361", "With interactions"),
            ]
        ):
            for row, m in enumerate(MODELS):
                r = out[m]["methods"][name]
                y = row + (j - 0.5) * 0.18
                ax.plot([100 * r["lo"][metric], 100 * r["hi"][metric]], [y, y], color=color)
                ax.scatter(
                    100 * r["delta"][metric],
                    y,
                    color=color,
                    s=22,
                    label=label if row == 0 else None,
                )
        ax.axvline(0, color=".6", linestyle="--", linewidth=1)
        ax.set(
            yticks=range(len(MODELS)),
            yticklabels=list(MODELS) if metric == 0 else [],
            xlabel="Change vs. published ranking (points)",
            title=["mAP@10   (higher is better)", "NegRecall@10   (lower is better)"][metric],
        )
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "pinpoint_folds.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
