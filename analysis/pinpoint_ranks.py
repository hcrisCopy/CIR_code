"""Rerank fixed public PinPoint shortlists; annotations enter only after scoring."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
from tqdm import tqdm

from baseline.runtime import write_json
from .breadth import OUT
from .breadth_results import bootstrap

PUBLIC = Path("../CIR_data/raw/pinpoint_public")
FILES = {
    "text": "metaclip2_retrieval_results_text_only",
    "image": "metaclip2_retrieval_results_image_only",
    "MetaCLIP2": "metaclip2_retrieval_results_combined",
    "BGE-VL-MLLM-S1": "bge_vl_mllm_s1_retrieval_results_licensed",
}


def lookup():
    spec = importlib.util.spec_from_file_location(
        "pinpoint_official_metrics", PUBLIC / "metrics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rank_evidence(items):
    return {key: 11 / (11 + i) for i, key in enumerate(items)}


def main():
    official = lookup()
    truth = duckdb.sql(
        f"SELECT query_id,query_image_signature,positive_candidates,negative_candidates FROM read_parquet('{PUBLIC}/pinpoint_licensed.parquet')"
    ).fetchall()
    files = {key: json.loads((PUBLIC / (f + ".json")).read_text()) for key, f in FILES.items()}
    summary = {}
    for model in ["MetaCLIP2", "BGE-VL-MLLM-S1"]:
        outcomes = []
        refs = []
        counts = Counter()
        rng = np.random.default_rng(42)
        for id_, ref, pos, neg in tqdm(truth, desc=model):
            key = str(id_).removeprefix("query_").zfill(5)
            base = files[model][key]["retrieved_items"]
            if len(set(base)) != len(base):
                raise ValueError("Duplicate public candidates")

            def evidence(mode):
                mapping = rank_evidence(files[mode][key]["retrieved_items"])
                return np.array([mapping.get(c, 0) for c in base])

            b = np.array(list(rank_evidence(base).values()))
            t = evidence("text")
            i = evidence("image")
            scores = {
                "base": b,
                "text": b + 0.5 * t,
                "image": b + 0.5 * i,
                "subtract_image": b - 0.5 * i,
                "no_text_image_penalty": b - 0.5 * i * (t == 0),
                "combined": b + 0.5 * t - 0.5 * i * (t == 0),
                "linear": b + 0.5 * t - 0.5 * i,
                "shuffled": b + 0.5 * rng.permutation(t),
                "agreement": b + 0.5 * np.minimum(i, t),
            }
            pos_set, neg_set = set(pos), set(neg)
            for j, cand in enumerate(base[:10]):
                kind = "image_no_text" if i[j] > 0 and t[j] == 0 else "other"
                counts[kind + "_all"] += 1
                counts[kind + "_pos"] += int(cand in pos_set)
                counts[kind + "_neg"] += int(cand in neg_set)
            measurement = []
            for score in scores.values():
                top = [base[j] for j in np.argsort(-score, kind="stable")[:10]]
                ap = official.calculate_metrics_at_k(top, pos, 10)["ap"]
                nr = official.calculate_neg_recall_at_k(top, neg, 10)
                measurement.append((ap, nr))
            outcomes.append(measurement)
            refs.append(ref)
        values = np.array(outcomes)
        delta = values - values[:, :1]
        n = len(values)
        intervals = bootstrap(delta.reshape(n, -1), refs).reshape(1000, len(scores), 2)
        lo, hi = np.quantile(intervals, [0.025, 0.975], axis=0)
        summary[model] = dict(n=n, reference_groups=len(set(refs)), exposure=counts, methods={})
        for j, name in enumerate(scores):
            row = {}
            for h, metric in enumerate(["map10", "negative_recall10"]):
                v = delta[:, j, h]
                row[metric] = dict(
                    value=float(values[:, j, h].mean()),
                    delta=float(v.mean()),
                    lo=float(lo[j, h]),
                    hi=float(hi[j, h]),
                    up=int((v > 0).sum()),
                    down=int((v < 0).sum()),
                )
            summary[model]["methods"][name] = row
        assert len(files[model]) == n
    write_json(OUT / "pinpoint.json", summary)


if __name__ == "__main__":
    main()
