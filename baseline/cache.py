"""Cache the complete gallery and a reproducible subset of queries per category."""

import math
import os
import time
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import CATEGORIES, layout, read
from .model import VIT_SHA, Encoder, checkpoint_paths, digest, load_image
from .runtime import (
    Runtime,
    cache_dir,
    load,
    parser,
    prepare_run,
    save,
    signature,
    write_json,
)


class Images(Dataset):
    def __init__(self, files, preprocess):
        self.files, self.preprocess = files, preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return load_image(self.files[i], self.preprocess)


@torch.inference_mode()
def cache(args, rt, encoder, weights, split, category):
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(rt.device)
    path = cache_dir(args, split, category)
    paths, records = layout(args.dataset, category, split)
    queries = records[: args.limit or None]
    names, files = list(paths), list(paths.values())
    index = {name: i for i, name in enumerate(names)}
    image_info = [
        (name, str(file), file.stat().st_size, file.stat().st_mtime_ns)
        for name, file in zip(names, files)
    ]
    cfg = dict(
        version=1,
        dataset=args.dataset,
        split=split,
        category=category,
        layer=args.layer if args.backbone != "release" else None,
        weights=weights,
        image_identity=signature(image_info),
        queries=signature(queries),
        batch_size=args.batch_size,
        limit=args.limit,
        transform="targetpad1.25",
    )
    prepare_run(path, cfg, rt, args.clean)
    size = args.batch_size
    ni, nq = math.ceil(len(files) / size), math.ceil(len(queries) / size)
    raw = None
    if args.backbone == "vit_cir":
        raw = cache_dir(Namespace(**(vars(args) | {"backbone": "vit"})), split, category)
        raw_config = read(raw / "config.json")
        expected = dict(cfg, weights=weights[:1])
        if raw_config != expected or read(raw / "complete.json")["identity"] != signature(expected):
            raise ValueError(
                "Rebuild the matching ViT image/text cache before composing trained queries"
            )
    batches = [i for i in range(rt.rank, ni, rt.world) if not (path / f"images-{i}.pt").exists()]
    if raw:
        for i in batches:
            os.link(raw / f"images-{i}.pt", path / f"images-{i}.pt")
        batches = []
    samples = [range(i * size, min((i + 1) * size, len(files))) for i in batches]
    loader = DataLoader(
        Images(files, encoder.preprocess),
        batch_sampler=samples,
        num_workers=args.workers,
        pin_memory=True,
    )
    for i, images in zip(
        batches, tqdm(loader, desc=f"{split}/{category} gallery", disable=rt.rank != 0), strict=True
    ):
        features = encoder.images(images.to(rt.device, non_blocking=True))
        if not all(value.isfinite().all() for value in features.values()):
            raise ValueError(f"Non-finite image embeddings at {split}/{category}/{i}")
        save(path / f"images-{i}.pt", {k: v.cpu() for k, v in features.items()})
    rt.sync()
    # Preserve the raw reference norms expected by the released Combiner.
    gallery = torch.cat([load(path / f"images-{i}.pt")["global"] for i in range(ni)])
    for i in tqdm(
        range(rt.rank, nq, rt.world), desc=f"{split}/{category} queries", disable=rt.rank != 0
    ):
        out = path / f"queries-{i}.pt"
        if out.exists():
            continue
        group = queries[i * size : (i + 1) * size]
        text = (
            load(raw / f"queries-{i}.pt")["text"].to(rt.device)
            if raw
            else encoder.texts([q["text"] for q in group])
        )
        reference = gallery[[index[q["reference"]] for q in group]].to(rt.device)
        combined = encoder.compose(reference, text)
        if not (text.isfinite().all() and combined.isfinite().all()):
            raise ValueError(f"Non-finite query embeddings at {split}/{category}/{i}")
        save(out, {"text": text.cpu(), "base": combined.cpu()})
    rt.sync()
    if rt.rank == 0:
        write_json(
            path / "complete.json",
            dict(
                identity=signature(cfg),
                names=names,
                queries=queries,
                weights=weights,
                image_batches=ni,
                query_batches=nq,
                seconds_this_run=time.perf_counter() - start,
                rank0_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            ),
        )
        print(f"{split}/{category}: {len(names)} images, {len(queries)} queries -> {path}")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error("Invalid batch size or worker count")
    rt = Runtime()
    files = checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit)
    weights = [digest(f) for f in files]
    if args.backbone != "release" and weights[0] != VIT_SHA:
        raise ValueError("ViT-B-16.pt does not match the official OpenAI checkpoint")
    encoder = Encoder(files, rt.device, args.layer)
    for split in args.splits:
        for category in CATEGORIES[args.dataset]:
            cache(args, rt, encoder, weights, split, category)
    rt.close()
