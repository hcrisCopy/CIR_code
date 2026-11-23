"""Cache reference interventions. Candidates and captions always stay unchanged."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.cache import Images
from baseline.data import CATEGORIES, ROOT, layout, read
from baseline.model import Encoder, checkpoint_paths, digest
from baseline.runtime import (
    Runtime,
    cache_dir,
    load,
    parser,
    prepare_run,
    run_key,
    save,
    write_json,
)


def view_dir(args, split, category):
    return ROOT / "embeddings/interventions" / run_key(args) / split / category


def perturb(normalized):
    mean = normalized.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = normalized.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    rgb = (normalized * std + mean).clamp(0, 1)
    luminance = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    gray = luminance.expand_as(rgb)
    # Shift all channels equally, choosing the direction with more headroom at each pixel.
    up = 1 - rgb.amax(1, keepdim=True)
    down = rgb.amin(1, keepdim=True)
    shift = torch.where(up > down, up, -down).expand_as(rgb)
    norm = lambda x: x.flatten(1).norm(dim=1)
    size = norm(gray - rgb)
    scale = (size / norm(shift).clamp_min(1e-8)).clamp_max(1)
    control = rgb + scale[:, None, None, None] * shift
    ratio = norm(control - rgb) / size.clamp_min(1e-8)
    return (gray - mean) / std, (control - mean) / std, ratio, size


@torch.inference_mode()
def run(args, rt):
    files = checkpoint_paths(args.dataset, "vit")
    encoder = Encoder(files, rt.device, args.layer)
    for split in args.splits:
        for category in CATEGORIES[args.dataset]:
            # Interventions depend only on the frozen vision encoder.
            meta = read(cache_dir(args, split, category) / "complete.json")
            if meta["weights"] != [digest(files[0])]:
                raise ValueError("Mismatched frozen ViT cache")
            names = sorted({q["reference"] for q in meta["queries"]})
            paths, _ = layout(args.dataset, category, split)
            out = view_dir(args, split, category)
            config = dict(cache=meta["identity"], batch=args.batch_size, version=1)
            prepare_run(out, config, rt, args.clean)
            n = (len(names) + args.batch_size - 1) // args.batch_size
            todo = [i for i in range(rt.rank, n, rt.world) if not (out / f"batch-{i}.pt").exists()]
            batches = [
                range(i * args.batch_size, min(len(names), (i + 1) * args.batch_size)) for i in todo
            ]
            loader = DataLoader(
                Images([paths[k] for k in names], encoder.preprocess),
                batch_sampler=batches,
                num_workers=args.workers,
                pin_memory=True,
            )
            for i, pixels in zip(
                todo,
                tqdm(loader, desc=f"{category}/{split} color controls", disable=rt.rank != 0),
                strict=True,
            ):
                gray, luma, ratio, size = perturb(pixels.to(rt.device))
                outputs = {k: v.cpu() for k, v in encoder.images(gray).items()}
                outputs["luma"] = encoder.clip.encode_image(luma).cpu()
                outputs["ratio"] = ratio.cpu()
                outputs["pixel_delta"] = size.cpu()
                save(out / f"batch-{i}.pt", outputs)
            rt.sync()
            if rt.rank == 0:
                write_json(out / "complete.json", dict(names=names, batches=n, **config))


def load_views(args, split, category):
    from argparse import Namespace

    raw = Namespace(**(vars(args) | {"backbone": "vit"}))
    folder = view_dir(raw, split, category)
    meta = read(folder / "complete.json")
    if meta["cache"] != read(cache_dir(raw, split, category) / "complete.json")["identity"]:
        raise ValueError("Stale reference views")
    batches = [load(folder / f"batch-{i}.pt") for i in range(meta["batches"])]
    return meta, {k: torch.cat([b[k] for b in batches]) for k in batches[0]}


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.backbone != "vit" or args.batch_size < 1:
        p.error("Use frozen ViT and a positive batch size")
    rt = Runtime()
    run(args, rt)
    rt.close()
