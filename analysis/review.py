"""Draw seed-42 success/failure samples and create a human review sheet."""

import json
import random

from baseline.data import CATEGORIES, ROOT
from baseline.model import digest
from baseline.runtime import parser, run_key


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sheet_path(args):
    return ROOT / "reviews" / run_key(args) / args.split / f"{args.label}.jsonl"


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--label", default="base")
    p.add_argument(
        "--per-group", type=int, default=100, help="Per category and rank-1 success/failure stratum"
    )
    args = p.parse_args()
    if args.per_group < 1 or not args.label.isidentifier():
        p.error("Invalid sample size or label")
    path = sheet_path(args)
    rng = random.Random(42)
    selected = []
    for category in CATEGORIES[args.dataset]:
        source = ROOT / "results/diagnosis" / run_key(args) / args.split / args.label / category
        rows = read_lines(source / "rankings.jsonl")
        source_id = digest(source / "rankings.jsonl")
        for correct in (False, True):
            pool = [q for q in rows if (q["rank"] == 1) == correct]
            for q in rng.sample(pool, min(args.per_group, len(pool))):
                selected.append(
                    dict(
                        dataset=args.dataset,
                        split=args.split,
                        category=category,
                        id=q["id"],
                        reference=q["reference"],
                        target=q["target"],
                        text=q["text"],
                        negative=q["negative"],
                        rank=q["rank"],
                        top=q["top"],
                        margin=q["margin"],
                        scores=q["scores"],
                        source=source_id,
                        stratum_size=len(pool),
                        reviewed=False,
                        use=None,
                        error_type="",
                        modification_type="",
                        factor="",
                        condition="",
                        donor="",
                        box=None,
                        donor_box=None,
                    )
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():

        def identities(rows):
            return [(r["category"], r["id"], r["source"]) for r in rows]

        if identities(read_lines(path)) != identities(selected):
            raise ValueError(
                f"Sampling inputs changed. Archive the previous review sheet at {path} first."
            )
        print(f"Reusing existing manual review at {path}")
        raise SystemExit(0)
    temp = path.with_suffix(".partial")
    with temp.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    print(f"{len(selected)} queries -> {path}")
