"""Measure conditional ranking risk on human-checked pairs and matched controls."""

import math
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from baseline.data import CATEGORIES, ROOT, layout, read
from baseline.heads import compose, read_head
from baseline.model import Encoder, checkpoint_paths, digest
from baseline.runtime import (
    Runtime,
    cache_dir,
    load,
    load_cache,
    parser,
    prepare_run,
    run_key,
    save,
    signature,
    write_json,
)
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from .review import read_lines, sheet_path


def box(values, size):
    if (
        not isinstance(values, list)
        or len(values) != 4
        or not all(isinstance(v, (int, float)) for v in values)
    ):
        raise ValueError("Expected normalized box [left, top, right, bottom]")
    left, top, right, bottom = values
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"Invalid box {values}")
    pixels = tuple(round(v * size[i % 2]) for i, v in enumerate(values))
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise ValueError(f"Box smaller than one pixel: {values}")
    return pixels


def intervention_images(ref_path, donor_path, row, rng, transform):
    with Image.open(ref_path) as stream:
        original = stream.convert("RGB")
    with Image.open(donor_path) as stream:
        donor = stream.convert("RGB")
    region = box(row["box"], original.size)
    donor_region = box(row["donor_box"], donor.size)
    w, h = region[2] - region[0], region[3] - region[1]
    patch = donor.crop(donor_region).resize((w, h), Image.Resampling.BICUBIC)
    # Uniformly sample a different legal location with the same area and aspect ratio.
    cols = original.width - w + 1
    count = cols * (original.height - h + 1)
    if count <= 1:
        raise ValueError("Use a local region; no alternative control location available")
    index = rng.randrange(count - 1)
    index += index >= region[1] * cols + region[0]
    locations = [region[:2], (index % cols, index // cols)]
    variants = []
    for location in locations:
        variant = original.copy()
        variant.paste(patch, location)
        variants.append(transform(variant))
    if torch.equal(variants[0], transform(original)):
        raise ValueError(
            f"Replacement is invisible after preprocessing for {row['category']}/{row['id']}"
        )
    masked = original.copy()
    masked.paste((0, 0, 0), region)
    return torch.stack([*variants, transform(masked)])


def effect(q, direction, original, temperature):
    margin = q @ direction
    loss = F.softplus(-margin / temperature)
    return {
        "margin": float(margin),
        "delta_margin": float(margin - original),
        "necessity": float(loss - F.softplus(-original / temperature)),
    }


@torch.inference_mode()
def run(args, rt):
    review_file = sheet_path(
        Namespace(**(vars(args) | {"backbone": args.review_backbone or args.backbone}))
    )
    sheet = read_lines(review_file)
    if not sheet or any(
        r["dataset"] != args.dataset
        or r["split"] != args.split
        or r["reviewed"] is not True
        or not isinstance(r["use"], bool)
        for r in sheet
    ):
        raise ValueError("Every sampled row needs a human decision: reviewed=true and use=true/false")
    rows = [r for r in sheet if r["use"]]
    keys = [(r["category"], r["id"]) for r in rows]
    if (
        not rows
        or len(keys) != len(set(keys))
        or any(cat not in CATEGORIES[args.dataset] for cat, _ in keys)
    ):
        raise ValueError("Empty, duplicate or invalid reviewed query IDs")
    files = checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit)
    weights = [digest(f) for f in files]
    encoder = Encoder(files, rt.device, args.layer)
    head, source, head_id = read_head(args.head, args, weights, rt.device)
    folder = ROOT / "results/interventions" / run_key(args) / args.split / args.label
    identities = {}
    for category in CATEGORIES[args.dataset]:
        identities[category] = read(cache_dir(args, args.split, category) / "complete.json")["identity"]
    config = dict(
        sheet=digest(review_file),
        weights=weights,
        head=head_id,
        temperature=args.temperature,
        caches=identities,
        version=1,
    )
    prepare_run(folder, config, rt, args.clean)
    for category in CATEGORIES[args.dataset]:
        meta, image, query = load_cache(args, args.split, category)
        if meta["weights"] != weights:
            raise ValueError("Image cache and current encoder weights differ")
        paths, _ = layout(args.dataset, category, args.split)
        idx = {name: i for i, name in enumerate(meta["names"])}
        lookup = {q["id"]: i for i, q in enumerate(meta["queries"])}
        indices = [i for i, row in enumerate(rows) if row["category"] == category]
        for i in tqdm(
            indices[rt.rank :: rt.world], desc=f"{category} interventions", disable=rt.rank != 0
        ):
            dest = folder / f"pair-{i}.pt"
            if dest.is_file():
                continue
            row = rows[i]
            j = lookup[row["id"]]
            q = meta["queries"][j]
            if (
                any(row[k] != q[k] for k in ("reference", "text", "target"))
                or row["negative"] == q["target"]
                or row["donor"] in {row["reference"], row["target"], row["negative"]}
                or (args.dataset == "cirr" and row["negative"] == q["reference"])
                or row["condition"] not in ("keep", "change")
                or not all(row[k].strip() for k in ("factor", "modification_type", "error_type"))
            ):
                raise ValueError(f"Invalid reviewed semantics or identities at {category}/{q['id']}")
            key = signature([42, args.dataset, args.split, category, q["id"]])
            batch = intervention_images(
                paths[row["reference"]], paths[row["donor"]], row, random.Random(key), encoder.preprocess
            ).to(rt.device)
            changed = encoder.images(batch)
            text = query["text"][j : j + 1].to(rt.device)
            ref = idx[q["reference"]]
            clean = compose(
                head,
                query["base"][j : j + 1].to(rt.device),
                image[source][ref : ref + 1].to(rt.device),
                text,
            )[0]
            pos, neg = idx[q["target"]], idx[row["negative"]]
            diff = F.normalize(image["global"][[pos, neg]].to(rt.device))
            direction = diff[0] - diff[1]
            margin = clean @ direction
            base = encoder.compose(changed["global"], text.expand(3, -1))
            altered = compose(head, base, changed[source], text.expand(3, -1))
            stats = {
                name: effect(vec, direction, margin, args.temperature)
                for name, vec in zip(("replacement", "random_region", "black_region"), altered)
            }
            gen = torch.Generator().manual_seed(int(key[:15], 16))
            noise = torch.randn(clean.shape, generator=gen).to(rt.device)
            # Random query direction with exactly the same chord length on the unit sphere.
            tangent = F.normalize(noise - (noise @ clean) * clean, dim=0)
            cosine = torch.dot(altered[0], clean).clamp(-1, 1)
            random_q = cosine * clean + (1 - cosine.square()).clamp_min(0).sqrt() * tangent
            stats["random_direction"] = effect(random_q, direction, margin, args.temperature)
            zeros = torch.zeros_like(image["global"][ref : ref + 1]).to(rt.device)
            zero_q = compose(
                head, encoder.compose(zeros, text), torch.zeros_like(changed[source][:1]), text
            )[0]
            stats["zero_reference"] = effect(zero_q, direction, margin, args.temperature)
            save(
                dest,
                dict(
                    id=q["id"],
                    category=category,
                    negative=row["negative"],
                    factor=row["factor"],
                    condition=row["condition"],
                    modification_type=row["modification_type"],
                    error_type=row["error_type"],
                    base_margin=margin.item(),
                    effects=stats,
                    changed_base=base[0].cpu(),
                    changed={k: v[0].cpu() for k, v in changed.items()},
                ),
            )
    rt.sync()
    if rt.rank == 0:
        result = [load(folder / f"pair-{i}.pt") for i in range(len(rows))]
        groups = defaultdict(list)
        for record in result:
            groups[(record["factor"], record["condition"])].append(record)
        summary = [
            dict(
                factor=f,
                condition=c,
                n=len(group),
                mean_N={
                    k: mean(r["effects"][k]["necessity"] for r in group) for k in group[0]["effects"]
                },
            )
            for (f, c), group in sorted(groups.items())
        ]
        means = {(r["factor"], r["condition"]): r["mean_N"] for r in summary}
        contrasts = {}
        for factor, condition in means:
            if condition == "keep" and (factor, "change") in means:
                contrasts[factor] = {
                    k: value - means[factor, "change"][k] for k, value in means[factor, "keep"].items()
                }
        write_json(
            folder / "effects.json",
            [{k: v for k, v in r.items() if k not in ("changed", "changed_base")} for r in result],
        )
        write_json(
            folder / "summary.json",
            dict(sampled=len(sheet), included=len(rows), groups=summary, keep_minus_change=contrasts),
        )
        print(summary, folder)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--label", default="base")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--head", type=Path)
    p.add_argument("--review-backbone", choices=["vit", "vit_cir", "release"])
    args = p.parse_args()
    if not args.label.isidentifier() or not math.isfinite(args.temperature) or args.temperature <= 0:
        p.error("Invalid label or temperature")
    rt = Runtime()
    run(args, rt)
    rt.close()
