"""Test whether the color result transfers to left/right placement on fixed validation products."""

from collections import defaultdict
import gzip
import json
import torch
from torch.nn import functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES, ROOT
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, parser, run_key, write_json
from .controlled_color import MEAN, STD, examples
from .paired_edits import cluster_stat


def position_views(images):
    size = images.shape[-1]
    side = size // 2
    tile = F.interpolate(
        images, size=(side, side), mode="bilinear", align_corners=False, antialias=True
    )
    white = (1 - images.new_tensor(MEAN)) / images.new_tensor(STD)
    outputs = []
    for x in (0, size - side, (size - side) // 2):
        canvas = white[None, :, None, None].expand_as(images).clone()
        y = (size - side) // 2
        canvas[:, :, y : y + side, x : x + side] = tile
        outputs.append(canvas)
    return outputs


TEMPLATES = (
    {
        "keep": "keep it on the same side of the frame",
        "switch": "move it to the opposite side of the frame",
        "set": "put it on the {side} side of the frame",
    },
    {
        "keep": "leave the horizontal position unchanged",
        "switch": "swap left and right positions",
        "set": "the item should appear on the {side}",
    },
)


@torch.inference_mode()
def run(args, rt, enc, cat):
    paths, queries = examples(cat, "val", args.per_category)
    records = []
    for i in tqdm(range(0, len(queries), 16), desc=cat):
        rows = queries[i : i + 16]
        n = len(rows)

        def views(field):
            return position_views(
                torch.stack([load_image(paths[q[field]], enc.preprocess) for q in rows]).to(
                    rt.device
                )
            )

        all_src = enc.clip.encode_image(torch.cat(views("reference")))
        a, b, center = all_src.chunk(3)
        src = torch.cat([a, b])
        near, far = F.normalize(enc.clip.encode_image(torch.cat(views("target")[:2]))).chunk(2)
        delta = (near - far).repeat(2, 1)

        def absolute(side):
            text = enc.texts(
                [q["text"] + ", and the target item is on the " + side for q in rows]
            ).repeat(2, 1)
            return enc.compose(src, text)

        abs_left, abs_right = absolute("left"), absolute("right")
        for template_id, template in enumerate(TEMPLATES):
            for mode, signs in [
                ("keep", [1, -1]),
                ("set0", [1, 1]),
                ("set1", [-1, -1]),
                ("switch", [-1, 1]),
            ]:
                phrase = (
                    template["set"].format(side="left" if mode == "set0" else "right")
                    if mode.startswith("set")
                    else template[mode]
                )
                text = enc.texts([q["text"] + ", and " + phrase for q in rows]).repeat(2, 1)
                q = enc.compose(src, text)
                null = enc.compose(center.repeat(2, 1), text)
                target = torch.tensor(signs, device=rt.device).repeat_interleave(n)
                known = {
                    "keep": q,
                    "switch": F.normalize(2 * null - q),
                    "set0": null,
                    "set1": null,
                }[mode]
                methods = {
                    "base": q,
                    "reference": F.normalize(src),
                    "text": F.normalize(text),
                    "operator_oracle": known,
                    "absolute_oracle": torch.where(target[:, None] > 0, abs_left, abs_right),
                }
                for model, vec in methods.items():
                    scores = (vec * delta).sum(1)
                    m0, m1 = scores.chunk(2)
                    acc = ((scores * target) > 0).reshape(2, n).float().mean(0)
                    for row, accuracy, slope in zip(rows, acc.tolist(), (m0 - m1).tolist()):
                        records.append(
                            dict(
                                ref=row["reference"],
                                model=model,
                                mode=mode,
                                template=template_id,
                                accuracy=accuracy,
                                slope=slope,
                            )
                        )
    out = ROOT / "results/controlled_position" / run_key(args) / cat
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "products.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    by = defaultdict(list)
    for row in records:
        by[row["model"], row["mode"]].append(row)
    write_json(
        out / "summary.json",
        {
            "/".join(k): {
                metric: cluster_stat([r[metric] for r in rows], [r["ref"] for r in rows])
                for metric in ("slope", "accuracy")
            }
            for k, rows in by.items()
        },
    )
    print(out)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--per-category", type=int, default=128)
    args = p.parse_args()
    if args.dataset != "fashioniq" or args.backbone != "vit_cir":
        p.error("Use the FashionIQ baseline")
    rt = Runtime()
    enc = Encoder(
        checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit), rt.device, args.layer
    )
    for cat in CATEGORIES["fashioniq"]:
        run(args, rt, enc, cat)
    rt.close()
