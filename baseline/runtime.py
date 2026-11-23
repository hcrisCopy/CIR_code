import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import CATEGORIES, ROOT


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=CATEGORIES, required=True)
    p.add_argument("--backbone", choices=["release", "vit", "vit_cir"], default="vit")
    p.add_argument("--layer", type=int, default=6, help="1-based ViT block")
    p.add_argument("--limit", type=int, default=0, help="Queries per split/category; 0 uses all")
    p.add_argument("--clean", action="store_true")
    return p


class Runtime:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group("nccl", device_id=self.device)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def sync(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save(path, obj):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def prepare_run(path, config, rt, clean):
    if rt.rank == 0:
        if clean and path.is_dir():
            if not path.resolve().is_relative_to(ROOT.resolve()) or not (path / "config.json").is_file():
                raise ValueError(f"Refusing to clean unrecognized run {path}")
            shutil.rmtree(path)
        path.mkdir(exist_ok=True, parents=True)
        marker = path / "config.json"
        if marker.exists():
            if json.loads(marker.read_text()) != config:
                raise ValueError(f"Inputs changed for {path}. Select --clean to rebuild.")
        elif any(path.iterdir()):
            raise ValueError(f"Unidentified files at {path}")
        else:
            write_json(marker, config)
    rt.sync()


def run_key(args):
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    return (
        Path(args.dataset)
        / (f"{args.backbone}-layer{args.layer}" if args.backbone != "release" else "release")
        / (f"first{args.limit}" if args.limit else "all")
    )


def cache_dir(args, split, category):
    return ROOT / "embeddings" / run_key(args) / split / category


def load_cache(args, split, category):
    path = cache_dir(args, split, category)
    meta = json.loads((path / "complete.json").read_text())
    cfg = json.loads((path / "config.json").read_text())
    if meta["identity"] != signature(cfg):
        raise ValueError(f"Cache identity mismatch: {path}")

    def collect(kind, count):
        pieces = [load(path / f"{kind}-{i}.pt") for i in range(count)]
        return {key: torch.cat([part[key] for part in pieces]) for key in pieces[0]}

    return meta, collect("images", meta["image_batches"]), collect("queries", meta["query_batches"])
