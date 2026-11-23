"""Factorial color stress test on distinct FashionIQ reference and target photos.

Derived evaluation only. The color of each image and each instruction is controlled;
the garment identity and luminance pattern are fixed between the two color alternatives.
"""

import hashlib
import re

import torch
from torch.nn import functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES, ROOT, layout
from baseline.heads import read_head
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, parser, run_key, write_json

from .color_views import perturb
from .paired_edits import COLOR, cluster_stat

PALETTES = {
    "red_blue": ("red", "blue", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    "green_purple": ("green", "purple", [0.0, 1.0, 0.0], [0.7, 0.0, 1.0]),
}
PROMPTS = (
    {
        "keep": "keep the original color",
        "set": "make it {color}",
        "switch": "switch to the other color, {a} if it was {b} and {b} if it was {a}",
    },
    {
        "keep": "the color stays the same",
        "set": "the garment should be {color}",
        "switch": "exchange {a} for {b}, or {b} for {a}, whichever applies",
    },
    {
        "keep": "given the options {a} and {b}, use the color in the photo",
        "set": "given the options {a} and {b}, choose {color} for both reference photos",
        "switch": "given the options {a} and {b}, use the color absent from the photo",
    },
)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def colorize(x, palette):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    rgb = (x * std + mean).clamp(0, 1)
    weights = x.new_tensor([0.299, 0.587, 0.114])
    y = (rgb * weights[None, :, None, None]).sum(1, keepdim=True)
    # Equal chromatic displacement and equal luminance for both rendered colors.
    # White/black pixels stay neutral, including white studio backgrounds.
    strength = 0.88 * torch.minimum(y, 1 - y)
    result = []
    for primary in palette[2:]:
        d = x.new_tensor(primary)
        d = F.normalize(d - (weights * d).sum(), dim=0)
        result.append((y + strength * d[None, :, None, None] - mean) / std)
    return result


def examples(category, split, limit):
    paths, rows = layout("fashioniq", category, split)
    stronger_color = re.compile(
        r"\b(?:lighter|darker|shade|shades|hue|multicolor\w*|pastel|bright\w*|cream|ivory|tan|khaki|beige|floral colors|monochrome)\b",
        re.I,
    )
    ordered = sorted(
        rows, key=lambda q: hashlib.sha256(("42/" + category + "/" + q["id"]).encode()).digest()
    )
    result = []
    used = set()
    for q in ordered:
        if (
            COLOR.search(q["text"])
            or stronger_color.search(q["text"])
            or q["reference"] in used
            or q["target"] in used
        ):
            continue
        result.append(q)
        used.update((q["reference"], q["target"]))
        if limit and len(result) == limit:
            break
    return paths, result


@torch.inference_mode()
def run(args, rt, enc, category):
    paths, rows = examples(category, "val", args.per_category)
    folder = ROOT / "results/controlled_color" / run_key(args) / category
    folder.mkdir(parents=True, exist_ok=True)
    files = checkpoint_paths(args.dataset, "vit_cir", args.layer, args.limit)
    from baseline.model import digest

    weights = [digest(p) for p in files]
    heads = {
        name: read_head(
            ROOT / "checkpoints/diagnosis" / run_key(args) / name / "last.pt",
            args,
            weights,
            rt.device,
        )[:2]
        for name in args.heads
    }
    summaries = {}
    records = []
    from .color_operator import checkpoint, compose_signed, coefficient
    from baseline.runtime import load

    op = load(checkpoint(args))
    if op["encoder"] != weights[0] or op["dataset"] != args.dataset:
        raise ValueError("Incompatible color operator")
    for pname, pal in PALETTES.items():
        for a in tqdm(range(0, len(rows), args.batch_size), desc=f"{category}/{pname}"):
            batch = rows[a : a + args.batch_size]
            src = torch.stack(
                [load_image(paths[q["reference"]], enc.preprocess) for q in batch]
            ).to(rt.device)
            dst = torch.stack([load_image(paths[q["target"]], enc.preprocess) for q in batch]).to(
                rt.device
            )
            ref = enc.images(torch.cat(colorize(src, pal)))
            neutral = enc.clip.encode_image(perturb(src)[0]).repeat(2, 1)
            candidates = F.normalize(enc.clip.encode_image(torch.cat(colorize(dst, pal))))
            v0, v1 = candidates.chunk(2)
            direction = v0 - v1
            for template, prompt in enumerate(PROMPTS):
                for mode in ("keep", "set0", "set1", "switch"):
                    phrase = (
                        prompt["set"].format(color=pal[int(mode[-1])], a=pal[0], b=pal[1])
                        if mode.startswith("set")
                        else prompt[mode].format(a=pal[0], b=pal[1])
                    )
                    text = enc.texts([q["text"] + ", and " + phrase for q in batch]).repeat(2, 1)
                    base = enc.compose(ref["global"], text)
                    neutral_q = enc.compose(neutral, text)
                    inverse_q = F.normalize(2 * neutral_q - base)
                    # The oracle knows only the instruction operator, never the target or palette.
                    oracle = {
                        "keep": base,
                        "set0": neutral_q,
                        "set1": neutral_q,
                        "switch": inverse_q,
                    }
                    preds = {
                        "base": base,
                        "gray": neutral_q,
                        "inverse": inverse_q,
                        "operator_oracle": oracle[mode],
                        "text_operator": compose_signed(text, base, neutral_q, op),
                        "reference": F.normalize(ref["global"]),
                        "text": F.normalize(text),
                    }
                    preds.update({k: h(base, ref[feat], text) for k, (h, feat) in heads.items()})
                    for name, vec in preds.items():
                        m0, m1 = (vec * direction.repeat(2, 1)).sum(1).chunk(2)
                        expected = {
                            "keep": (1, -1),
                            "set0": (1, 1),
                            "set1": (-1, -1),
                            "switch": (-1, 1),
                        }[mode]
                        success = torch.stack((m0 * expected[0] > 0, m1 * expected[1] > 0), dim=1)
                        stats = torch.stack(
                            (success.float().mean(1), success.all(1).float(), m0 - m1), dim=1
                        ).tolist()
                        coefs = (
                            coefficient(text[: len(batch)], op).tolist()
                            if name == "text_operator"
                            else [None] * len(batch)
                        )
                        for q, (acc, joint, slope), coef in zip(batch, stats, coefs):
                            records.append(
                                dict(
                                    id=q["id"],
                                    ref=q["reference"],
                                    target=q["target"],
                                    palette=pname,
                                    template=template,
                                    mode=mode,
                                    model=name,
                                    accuracy=acc,
                                    joint=joint,
                                    slope=slope,
                                    coefficient=coef,
                                )
                            )
    from collections import defaultdict

    groups = defaultdict(list)
    for row in records:
        groups[row["model"], row["mode"]].append(row)
    for (model, mode), values in groups.items():
        refs = [r["ref"] for r in values]
        summaries.setdefault(model, {})[mode] = {
            k: cluster_stat([r[k] for r in values], refs) for k in ("accuracy", "joint", "slope")
        }
    write_json(
        folder / "summary.json",
        dict(
            n_products=len(rows), settings=list(PALETTES), templates=len(PROMPTS), methods=summaries
        ),
    )
    import gzip
    import json

    with gzip.open(folder / "products.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(category, len(rows), folder)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--per-category", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--heads", nargs="*", default=[])
    args = p.parse_args()
    if args.dataset != "fashioniq" or args.backbone != "vit_cir" or args.per_category < 1:
        p.error("Use trained FashionIQ ViT and a positive product-pair count")
    rt = Runtime()
    if rt.world != 1:
        raise ValueError("Use one GPU for this diagnostic")
    enc = Encoder(
        checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit), rt.device, args.layer
    )
    for cat in CATEGORIES["fashioniq"]:
        run(args, rt, enc, cat)
    rt.close()
