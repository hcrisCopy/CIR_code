"""Paired rank changes on the held-out human-reviewed validation samples."""

from collections import defaultdict
from statistics import mean

from baseline.data import CATEGORIES, ROOT
from baseline.runtime import parser, run_key, write_json

from .review import read_lines, sheet_path

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--models", nargs="+", required=True)
    args = p.parse_args()
    args.label, args.split = "base", "val"
    folder = ROOT / "results/diagnosis" / run_key(args) / "val"
    sheet = read_lines(sheet_path(args))
    if not sheet or any(r["reviewed"] is not True or not isinstance(r["use"], bool) for r in sheet):
        raise ValueError("Finish the validation review sheet first")
    output = {}
    for model in args.models:
        groups = defaultdict(list)
        for category in CATEGORIES[args.dataset]:
            base = {r["id"]: r for r in read_lines(folder / "base" / category / "rankings.jsonl")}
            current = {r["id"]: r for r in read_lines(folder / model / category / "rankings.jsonl")}
            for r in sheet:
                if r["category"] != category or not r["use"]:
                    continue
                a, b = base[r["id"]], current[r["id"]]
                if (
                    a["reference"] != r["reference"]
                    or a["target"] != r["target"]
                    or a["rank"] != r["rank"]
                ):
                    raise ValueError(f"Review sheet and baseline differ for {r['id']}")
                pair = (a["rank"], b["rank"])
                for key in (
                    "all",
                    f"factor/{r['factor']}/{r['condition']}",
                    f"error/{r['error_type']}",
                    f"baseline/{'success' if a['rank'] == 1 else 'failure'}",
                ):
                    groups[key].append(pair)
        if not groups:
            raise ValueError("No included validation reviews")
        output[model] = {
            key: dict(
                n=len(pairs),
                fixed=sum(a > 1 and b == 1 for a, b in pairs),
                broken=sum(a == 1 and b > 1 for a, b in pairs),
                mean_rank_gain=mean(a - b for a, b in pairs),
            )
            for key, pairs in groups.items()
        }
    write_json(folder / "paired_review.json", output)
    print(folder / "paired_review.json")
