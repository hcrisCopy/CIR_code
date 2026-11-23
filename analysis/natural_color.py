"""Blinded audit of unaltered FashionIQ same-reference examples with different named colors."""

import argparse
import csv
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from baseline.data import CATEGORIES, ROOT, layout
from baseline.runtime import load, write_json
from .paired_edits import pairs_for, cluster_stat

WORD = re.compile(
    r"\b(red|blue|green|purple|pink|orange|yellow|black|white|brown|grey|gray)\b", re.I
)
OUT = ROOT / "results/natural_color"


def keywords(q):
    return {w.lower().replace("grey", "gray") for w in WORD.findall(q["text"])}


def select(n):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for cat in CATEGORIES["fashioniq"]:
        paths, qs = layout("fashioniq", cat, "val")
        pool = []
        for pair_idx, (a, b) in enumerate(pairs_for("fashioniq", "val", qs).tolist()):
            left, right = qs[a], qs[b]
            lc, rc = keywords(left), keywords(right)
            if len(lc) == len(rc) == 1 and lc != rc:
                key = hashlib.sha256(("42/" + left["id"] + "/" + right["id"]).encode()).hexdigest()
                pool.append((key, pair_idx, left, right))
        chosen = []
        used = set()
        for _, pair_idx, l, r in sorted(pool):
            if l["reference"] in used:
                continue
            used.add(l["reference"])
            chosen.append((pair_idx, l, r))
            if len(chosen) == n:
                break
        sheet = Image.new("RGB", (3 * 382, ((len(chosen) + 2) // 3) * 147), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (j, l, r) in enumerate(chosen):
            x, y = (i % 3) * 382, (i // 3) * 147
            draw.text(
                (x + 4, y + 2),
                f"{cat}-{i + 1:02d}      source             A               B",
                fill="black",
            )
            for k, name in enumerate([l["reference"], l["target"], r["target"]]):
                with Image.open(paths[name]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (122, 121))
                    sheet.paste(
                        thumb,
                        (
                            x + 126 * k + (126 - thumb.width) // 2,
                            y + 18 + (121 - thumb.height) // 2,
                        ),
                    )
            records.append(
                dict(
                    key=f"{cat}-{i + 1:02d}",
                    category=cat,
                    pair_index=j,
                    reference=l["reference"],
                    left=l["id"],
                    right=r["id"],
                    left_text=l["text"],
                    right_text=r["text"],
                )
            )
        sheet.save(OUT / f"{cat}.jpg", quality=92)
        print(cat, len(pool), len(chosen))
    write_json(OUT / "manifest.json", records)


def score(labels):
    import json

    rows = json.loads((OUT / "manifest.json").read_text())
    by_id = {r["key"]: r for r in rows}
    ratings = list(csv.DictReader(labels.open(encoding="utf-8-sig")))
    if set(by_id) != {r["key"] for r in ratings} or len(by_id) != len(ratings):
        raise ValueError("Rate each blinded item exactly once")
    saved = {}
    groups = []
    entries = []
    for row in ratings:
        x = by_id[row["key"]]
        if row["same"] not in ("A", "B", "neither", "unclear"):
            raise ValueError(row)
        if row["same"] not in ("A", "B"):
            continue
        cat = x["category"]
        if cat not in saved:
            saved[cat] = load(
                ROOT
                / "results/paired_edits/fashioniq/vit_cir-layer6/all/val"
                / cat
                / "paired_data.pt"
            )
        d = saved[cat]
        j = x["pair_index"]
        a, b = d["pairs"][j].tolist()
        assert [d["ids"][a], d["ids"][b]] == [x["left"], x["right"]]
        s = 0 if row["same"] == "A" else 1
        groups.append(x["reference"])
        # Align both instructions on the same visual target axis before differencing.
        g = (d["clean"][j] - d["gray"][j]).numpy()
        l = (d["clean"][j] - d["luma"][j]).numpy()
        entries.append([g[s], -g[1 - s], g.sum(), l.sum(), g.sum() - l.sum()])
    values = np.asarray(entries)
    write_json(
        OUT / "audit.json",
        dict(
            screened=len(rows),
            counts={
                s: sum(r["same"] == s for r in ratings) for s in ("A", "B", "neither", "unclear")
            },
            effects={
                key: cluster_stat(v, groups)
                for key, v in zip(
                    [
                        "same_target_chroma",
                        "different_target_chroma",
                        "instruction_interaction",
                        "luma_interaction",
                        "chroma_minus_luma_interaction",
                    ],
                    values.T,
                )
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-category", type=int, default=12)
    p.add_argument("--labels", type=Path)
    args = p.parse_args()
    if args.labels:
        score(args.labels)
    else:
        select(args.per_category)
