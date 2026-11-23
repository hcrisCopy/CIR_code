"""Mirror consistency on original CIRR annotations with relative and absolute direction."""

import argparse
import csv
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F

from baseline.data import ROOT, layout
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, write_json
from .paired_edits import cluster_stat

OUT = ROOT / "results/natural_direction"
REL = re.compile(
    r"\b(?:(?:opposite|other|same) (?:direction|side|way)|(?:flipp?ed|mirrored|mirror image))\b",
    re.I,
)
ABS = re.compile(
    r"\b(?:fac(?:e|es|ing)|look(?:s|ing)?|point(?:s|ing)?|turn(?:ed|s|ing)?)\s+(?:\w+\s+){0,3}(?:left|right)\b",
    re.I,
)
LR = re.compile(r"\b(?:left|right)\b", re.I)


def sample():
    paths, qs = layout("cirr", "cirr", "val")
    groups = {
        "relative": [q for q in qs if REL.search(q["text"]) and not LR.search(q["text"])],
        "absolute": [q for q in qs if ABS.search(q["text"]) and not REL.search(q["text"])],
    }
    return paths, groups


def preview():
    paths, groups = sample()
    rows = groups["relative"]
    for page in range(0, len(rows), 15):
        section = rows[page : page + 15]
        canvas = Image.new("RGB", (960, ((len(section) + 2) // 3) * 130), "white")
        pen = ImageDraw.Draw(canvas)
        for i, q in enumerate(section):
            x, y = (i % 3) * 320, (i // 3) * 130
            pen.text((x + 5, y), q["id"] + "     source                 target", fill="black")
            for col, field in enumerate(["reference", "target"]):
                with Image.open(paths[q[field]]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (154, 109))
                    canvas.paste(
                        thumb,
                        (
                            x + col * 158 + (158 - thumb.width) // 2,
                            y + 16 + (109 - thumb.height) // 2,
                        ),
                    )
        canvas.save(OUT / f"sheet{page // 15 + 1}.jpg", quality=92)
    write_json(OUT / "queries.json", groups)
    print({k: len(v) for k, v in groups.items()})


@torch.inference_mode()
def measure(labels):
    annotations = {
        r["id"]: r["horizontal"] for r in csv.DictReader(labels.open(encoding="utf-8-sig"))
    }
    paths, groups = sample()
    if set(annotations) != {q["id"] for q in groups["relative"]} or not set(
        annotations.values()
    ) <= {"yes", "no", "unclear"}:
        raise ValueError("Audit all relative cases before measurement")
    rt = Runtime()
    enc = Encoder(checkpoint_paths("cirr", "vit_cir", 6, 0), rt.device, 6)
    output = {}
    for group, rows in groups.items():
        records = []
        for start in range(0, len(rows), 16):
            qs = rows[start : start + 16]
            n = len(qs)

            def views(field):
                x = torch.stack([load_image(paths[q[field]], enc.preprocess) for q in qs]).to(
                    rt.device
                )
                return enc.clip.encode_image(torch.cat([x, x.flip([-1])]))

            src = views("reference")
            t0, t1 = F.normalize(views("target")).chunk(2)
            delta = (t0 - t1).repeat(2, 1)
            caps = [q["text"] for q in qs]
            mirror = [
                LR.sub(lambda m: "right" if m[0].lower() == "left" else "left", t) for t in caps
            ]
            txt = enc.texts(caps + mirror)
            q = enc.compose(src, txt)
            opposite = enc.compose(src.roll(n, 0), txt)
            model_oracle = (
                F.normalize(q + opposite)
                if group == "absolute"
                else torch.where(
                    torch.tensor(
                        [bool(re.search(r"\bsame\b", c, re.I)) for c in caps] * 2, device=rt.device
                    )[:, None],
                    q,
                    opposite,
                )
            )
            for name, pred in {
                "base": q,
                "reference": F.normalize(src),
                "text": F.normalize(txt),
                "operator_rule": model_oracle,
            }.items():
                a, b = (pred * delta).sum(1).chunk(2)
                for row, u, v in zip(qs, a.tolist(), b.tolist()):
                    records.append(
                        dict(
                            id=row["id"],
                            ref=row["reference"],
                            method=name,
                            accuracy=(float(u > 0) + float(v < 0)) / 2,
                            joint=float(u > 0 and v < 0),
                            margin=(u - v) / 2,
                        )
                    )
        for verified in [False, True] if group == "relative" else [False]:
            keep = [x for x in records if not verified or annotations[x["id"]] == "yes"]
            result = {}
            for name in ["base", "reference", "text", "operator_rule"]:
                xs = [x for x in keep if x["method"] == name]
                refs = [v["ref"] for v in xs]
                result[name] = {
                    m: cluster_stat([v[m] for v in xs], refs)
                    for m in ["accuracy", "joint", "margin"]
                }
            xs = [x for x in keep if x["method"] == "operator_rule"]
            base = [x for x in keep if x["method"] == "base"]
            assert [x["id"] for x in xs] == [x["id"] for x in base]
            result["rule_gain"] = cluster_stat(
                [x["accuracy"] - b["accuracy"] for x, b in zip(xs, base)], [x["ref"] for x in xs]
            )
            output[group + ("_reviewed" if verified else "_all")] = result
        write_json(OUT / f"{group}_records.json", records)
    write_json(OUT / "summary.json", output)
    rt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.labels:
        measure(args.labels)
    else:
        preview()
