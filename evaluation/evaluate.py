"""Rank a complete gallery, save query-level evidence and summarize official validation metrics."""

import json
import math
import time
from pathlib import Path
from statistics import mean

import torch
from baseline.data import CATEGORIES, ROOT
from baseline.heads import compose, read_head
from baseline.runtime import (
    Runtime,
    load_cache,
    parser,
    prepare_run,
    run_key,
    write_json,
)
from tqdm import tqdm

from .ranking import metrics, rank_batch


@torch.inference_mode()
def evaluate(args, rt, category):
    torch.cuda.reset_peak_memory_stats(rt.device)
    meta, images, queries = load_cache(args, args.split, category)
    names, records = meta["names"], meta["queries"]
    index = {name: i for i, name in enumerate(names)}
    head, feature, head_id = read_head(args.head, args, meta["weights"], rt.device)
    folder = ROOT / "results/diagnosis" / run_key(args) / args.split / args.label / category
    config = dict(
        identity=meta["identity"], head=head_id, top_k=args.top_k, batch_size=args.batch_size, version=1
    )
    prepare_run(folder, config, rt, args.clean)
    start = time.perf_counter()
    gallery = images["global"].to(rt.device)
    batches = math.ceil(len(records) / args.batch_size)
    for i in tqdm(range(rt.rank, batches, rt.world), desc=f"{category} rank", disable=rt.rank != 0):
        out = folder / f"rank-{i}.json"
        if out.exists():
            continue
        a, b = i * args.batch_size, (i + 1) * args.batch_size
        batch = records[a:b]
        ref = [index[q["reference"]] for q in batch]
        text = queries["text"][a:b].to(rt.device)
        combined = compose(
            head, queries["base"][a:b].to(rt.device), images[feature][ref].to(rt.device), text
        )
        rows = rank_batch(args.dataset, batch, combined, gallery, text, names, index, args.top_k)
        write_json(out, rows)
    rt.sync()
    if rt.rank == 0:
        rows = []
        for i in range(batches):
            rows.extend(json.loads((folder / f"rank-{i}.json").read_text(encoding="utf-8")))
        out = folder / "rankings.jsonl"
        tmp = out.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
        result = metrics(args.dataset, rows)
        write_json(
            folder / "metrics.json",
            dict(
                queries=len(rows),
                metrics=result,
                seconds_this_run=time.perf_counter() - start,
                rank0_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            ),
        )
        print(category, result, folder)
        return result


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--head", type=Path, help="Trained diagnostic head; defaults to the original query")
    p.add_argument("--label", default="base")
    args = p.parse_args()
    if min(args.top_k, args.batch_size) < 1 or not args.label.isidentifier():
        p.error("Use positive sizes and a simple output label")
    rt = Runtime()
    results = [evaluate(args, rt, category) for category in CATEGORIES[args.dataset]]
    if rt.rank == 0:
        macro = {key: mean(result[key] for result in results) for key in results[0]}
        folder = ROOT / "results/diagnosis" / run_key(args) / args.split / args.label
        write_json(
            folder / "summary.json",
            dict(categories=dict(zip(CATEGORIES[args.dataset], results)), macro=macro),
        )
        print("Macro (%)", macro)
    rt.close()
