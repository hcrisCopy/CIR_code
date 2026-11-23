"""Audit natural-image transfer of a position probe fitted only on rendered train garments."""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F
from baseline.data import CATEGORIES, ROOT, layout
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, write_json
from .controlled_color import examples
from .factor_readout import fit, encode
from .paired_edits import cluster_stat


OUT = ROOT / "results/spatial_transfer"


def sample():
    paths, _ = layout("cirr", "cirr", "val")
    chosen = sorted(
        paths, key=lambda name: hashlib.sha256(("natural-position/42/" + name).encode()).digest()
    )[:72]
    return paths, chosen


def preview():
    paths, ids = sample()
    for page in range(0, len(ids), 24):
        sheet = Image.new("RGB", (1020, 600), "white")
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(ids[page : page + 24]):
            x, y = i % 6 * 170, i // 6 * 150
            draw.text((x + 5, y), str(page + i + 1), fill="black")
            with Image.open(paths[name]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (162, 128))
                sheet.paste(
                    thumb, (x + (170 - thumb.width) // 2, y + 16 + (128 - thumb.height) // 2)
                )
        sheet.save(OUT / f"sheet{page // 24 + 1}.jpg", quality=92)
    write_json(OUT / "images.json", ids)


@torch.inference_mode()
def test(labels):
    paths, ids = sample()
    rows = list(csv.DictReader(labels.open(encoding="utf-8-sig")))
    if [int(r["i"]) for r in rows] != list(range(1, 73)) or any(
        r["side"] not in ("left", "right", "unclear") for r in rows
    ):
        raise ValueError("Label all 72 images")
    valid = [
        (ids[i], 1 if r["side"] == "left" else -1)
        for i, r in enumerate(rows)
        if r["side"] != "unclear"
    ]
    rt = Runtime()
    enc = Encoder(checkpoint_paths("fashioniq", "vit"), rt.device, 6)
    train = defaultdict(list)
    for cat in CATEGORIES["fashioniq"]:
        pp, qq = examples(cat, "train", 64)
        for k, v in encode(enc, rt, pp, qq, "target", "position", None).items():
            train[k].append(v)
    weights = {k: fit(torch.cat(v)) for k, v in train.items()}
    proto = F.normalize(
        enc.texts(
            [
                "a photo of an object on the left side of the frame",
                "a photo of an object on the right side of the frame",
            ]
        )
    ).cpu()
    rec = defaultdict(list)
    for a in range(0, len(valid), 16):
        subset = valid[a : a + 16]
        x = torch.stack([load_image(paths[k], enc.preprocess) for k, _ in subset]).to(rt.device)
        features = {
            k: F.normalize(v).cpu() for k, v in enc.images(torch.cat([x, x.flip([-1])])).items()
        }
        truth = torch.tensor([y for _, y in subset])
        for method in [*weights, "zero_shot"]:
            if method == "zero_shot":
                scores = features["global"] @ (proto[0] - proto[1])
            else:
                center, w = weights[method]
                scores = (features[method] - center) @ w
            orig, mirror = scores.chunk(2)
            ok = torch.stack([orig * truth > 0, mirror * truth < 0], dim=1)
            for metric, t in {
                "original": ok[:, 0],
                "mean": ok.float().mean(1),
                "joint": ok.all(1),
                "relative_order": (orig - mirror) * truth > 0,
            }.items():
                rec[method + "/" + metric].extend(t.float().tolist())
    result = {k: cluster_stat(v, [name for name, _ in valid]) for k, v in rec.items()}
    write_json(
        OUT / "summary.json",
        dict(
            screened=72,
            n=len(valid),
            counts={s: sum(r["side"] == s for r in rows) for s in ("left", "right", "unclear")},
            methods=result,
        ),
    )
    rt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.labels:
        test(args.labels)
    else:
        preview()
