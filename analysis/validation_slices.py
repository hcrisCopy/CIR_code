"""Paired validation contrasts and a reference/text join to external CIRCUS strata."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baseline.data import CATEGORIES, DATA, ROOT, layout
from baseline.runtime import load, parser, run_key, write_json
from .paired_edits import cluster_stat


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


# Restrict this test to explicit destination-color phrases. Keep other queries unchanged.
HUE = r"(?:red|blue|green|yellow|purple|pink|orange|brown|black|white|gr[ae]y|navy|beige)"
DEST = re.compile(
    r"\b(?:make (?:it|the garment)|turn (?:it|the garment)|change (?:the |its )?colou?r to|should be|is (?:solid |plain )?)\s*"
    + HUE
    + r"\b",
    re.I,
)
EXCLUDE = re.compile(
    r"\b(?:same|keep|retain|not|less|more|instead|lighter|darker|striped|pattern|spots?|sleeves?|neckline)\b",
    re.I,
)


def native(args):
    ans = {}
    lookup = defaultdict(list)
    for cat in CATEGORIES[args.dataset]:
        folder = ROOT / "results/paired_edits" / run_key(args) / "val" / cat
        artifact = load(folder / "paired_data.pt")
        _, queries = layout(args.dataset, cat, "val")
        assert [q["id"] for q in queries] == artifact["ids"]
        for i, q in enumerate(queries):
            lookup[q["reference"], normalize(q["text"])].append((cat, i))
        rank = artifact["ranks"]
        base = rank["base"]
        select = torch.tensor(
            [bool(DEST.search(q["text"]) and not EXCLUDE.search(q["text"])) for q in queries]
        )
        rng = torch.Generator().manual_seed(42)
        rank["lexical_gray"] = torch.where(select, rank["gray"], base)
        rank["shuffled_gray"] = torch.where(
            select[torch.randperm(len(select), generator=rng)], rank["gray"], base
        )
        ans[cat] = (artifact, rank, select)
    return ans, lookup


def audit(args):
    directory = ROOT / "results/validation_slices" / run_key(args)
    directory.mkdir(parents=True, exist_ok=True)
    native_data, lookup = native(args)
    metrics = {}
    for cat, (artifact, rank, mask) in native_data.items():
        refs = np.array(artifact["references"])
        k = 1 if args.dataset == "cirr" else 10
        event = {n: (r.numpy() <= k).astype(float) for n, r in rank.items()}
        contrasts = [
            ("color_necessity", "color_shuffled"),
            ("color_necessity", "color_hard"),
            ("color_hard", "color_retrieval"),
            ("probe_patch", "probe_global"),
            ("retrieval_patch", "retrieval_global"),
            ("retrieval_cls", "retrieval_global"),
            ("chord034", "base"),
            ("lexical_gray", "shuffled_gray"),
        ]
        metrics[cat] = dict(
            n=len(refs),
            explicit_destination=int(mask.sum()),
            k=k,
            contrasts={
                f"{a}-{b}": cluster_stat(event[a] - event[b], refs)
                for a, b in contrasts
                if a in rank and b in rank
            },
            gray_on_selected=cluster_stat((event["gray"] - event["base"])[mask], refs[mask]),
        )
    circus = {}
    matched_all = set()
    for stratum in ["solved", "unsolved"]:
        p = DATA / "circus_slices" / f"mbeir_{args.dataset}_task7_test_validated_{stratum}.jsonl"
        records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        matched = []
        rejected = []
        for record in records:
            key = Path(record["query_img_path"]).stem, normalize(record["query_txt"])
            hits = lookup.get(key, [])
            if len(hits) == 1 and hits[0] not in matched_all:
                matched.append(hits[0])
                matched_all.add(hits[0])
            else:
                rejected.append(dict(qid=record["qid"], candidates=len(hits)))
        refs = [native_data[c][0]["references"][i] for c, i in matched]
        methods = list(next(iter(native_data.values()))[1])
        results = {}
        for k in (1, 10, 50):
            calls = {
                m: np.array([int(native_data[c][1][m][i]) <= k for c, i in matched], float)
                for m in methods
            }
            results[str(k)] = {
                m: dict(recall=float(x.mean()), delta=cluster_stat(x - calls["base"], refs))
                for m, x in calls.items()
            }
        circus[stratum] = dict(
            source=len(records),
            matched=len(matched),
            skipped=rejected,
            by_category={c: sum(k == c for k, _ in matched) for c in native_data},
            methods=results,
        )
    write_json(directory / "summary.json", dict(native=metrics, circus=circus))
    print(directory)


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    audit(args)
