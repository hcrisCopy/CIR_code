"""Matched frozen-encoder experiments for information readout and necessity supervision."""

import math
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from baseline.data import CATEGORIES, ROOT, read
from baseline.heads import make_head
from baseline.model import digest
from baseline.runtime import (
    Runtime,
    load,
    load_cache,
    parser,
    prepare_run,
    run_key,
    save,
    write_json,
)

from .review import read_lines


def make_tables(args, split):
    parts = defaultdict(list)
    banks, lineage = [], {}
    reviewed = {}
    if split == "train" and args.review_label:
        folder = ROOT / "results/interventions" / run_key(args) / split / args.review_label
        summary = read(folder / "summary.json")
        config = read(folder / "config.json")
        if config["head"] != "base":
            raise ValueError("Generate training supervision with the frozen base query")
        if args.temperature != config["temperature"]:
            raise ValueError("Use the same ranking-risk temperature for measurement and training")
        lineage["review"] = digest(folder / "config.json")
        for i in range(summary["included"]):
            pair = load(folder / f"pair-{i}.pt")
            reviewed[(pair["category"], pair["id"])] = pair
    for c, category in enumerate(CATEGORIES[args.dataset]):
        meta, images, queries = load_cache(args, split, category)
        result = ROOT / "results/diagnosis" / run_key(args) / split / "base" / category
        cfg = read(result / "config.json")
        if cfg["identity"] != meta["identity"] or cfg["head"] != "base":
            raise ValueError(f"Rebuild frozen rankings for {category}/{split}")
        if reviewed and config["caches"][category] != meta["identity"]:
            raise ValueError(f"Reviewed pairs use a different cache for {category}")
        rows = read_lines(result / "rankings.jsonl")
        if [r["id"] for r in rows] != [q["id"] for q in meta["queries"]]:
            raise ValueError(f"Ranking and cache query order differ for {category}/{split}")
        index = {n: i for i, n in enumerate(meta["names"])}
        banks.append(F.normalize(images["global"]))
        lineage[category] = [meta["identity"], digest(result / "rankings.jsonl")]
        ids = [i for i, r in enumerate(rows) if not reviewed or (category, r["id"]) in reviewed]
        natural = None
        if split == "train" and args.paired_color:
            path = ROOT / "results/paired_edits" / run_key(args) / split / category / "supervision.pt"
            natural = load(path)
            if natural["identity"] != meta["identity"] or natural["temperature"] != args.temperature:
                raise ValueError("Incompatible natural-pair chroma supervision")
            lineage[f"{category}_paired_color"] = digest(path)
            ids = natural["ids"]
        if not ids:
            continue
        selected = [rows[i] for i in ids]
        refs, pos, neg = [], [], []
        for r in selected:
            pair = reviewed.get((category, r["id"]))
            refs.append(index[r["reference"]])
            pos.append(index[r["target"]])
            neg.append(index[pair["negative"] if pair else r["negative"]])
            if pair:
                parts["changed_base"].append(pair["changed_base"][None])
                parts["changed_feature"].append(pair["changed"][args.feature][None])
                necessity = pair["effects"]["replacement"]["necessity"]
                parts["target_risk"].append(torch.tensor([max(0.0, necessity)]))
        parts["base"].append(queries["base"][ids])
        parts["text"].append(queries["text"][ids])
        parts["feature"].append(images[args.feature][refs])
        parts["positive"].append(banks[c][pos])
        if natural is not None:
            neg = natural["negative"]
            for field in ("changed_base", "changed_feature", "target_risk"):
                parts[field].append(natural[field])
        parts["negative"].append(banks[c][neg])
        parts["pos_id"].append(torch.tensor(pos))
        parts["ref_id"].append(torch.tensor(refs))
        parts["category"].append(torch.full((len(ids),), c))
    table = {k: torch.cat(v) for k, v in parts.items()}
    if reviewed and len(table["base"]) != len(reviewed):
        raise ValueError("Unmatched training annotations")
    return table, banks, lineage, meta["weights"]


def batch_loss(model, table, indices, args, rt, permutation):
    # Each rank sees the same global candidate set, including duplicate target IDs as positives.
    local = indices[rt.rank :: rt.world]
    real_count = len(local)
    if not real_count:
        local = indices[:1]
    batch = {k: v[local] for k, v in table.items()}
    paired = args.loss in ("necessity", "shuffled")
    if paired:
        features = torch.cat((batch["feature"], batch["changed_feature"]))
        bases = torch.cat((batch["base"], batch["changed_base"]))
        clean, altered = model(bases, features, batch["text"].repeat(2, 1)).chunk(2)
    else:
        clean = model(batch["base"], batch["feature"], batch["text"])
    margin = (clean * (batch["positive"] - batch["negative"])).sum(-1)
    pair_loss = F.softplus(-margin / args.temperature)
    if args.loss == "probe":
        loss = pair_loss
    else:
        logits = clean @ table["positive"][indices].T / args.temperature
        same_category = batch["category"][:, None] == table["category"][indices]
        positives = same_category & (batch["pos_id"][:, None] == table["pos_id"][indices])
        retrieval = logits.masked_fill(~same_category, -torch.inf).logsumexp(1)
        loss = retrieval - logits.masked_fill(~positives, -torch.inf).logsumexp(1)
        if args.loss in ("hard", "necessity", "shuffled"):
            loss = loss + pair_loss
        if paired:
            alt_margin = (altered * (batch["positive"] - batch["negative"])).sum(-1)
            risk_change = F.softplus(-alt_margin / args.temperature) - pair_loss
            target = batch["target_risk"]
            if args.loss == "shuffled":
                target = table["target_risk"][permutation[local]]
            loss = loss + args.weight * F.smooth_l1_loss(risk_change, target, reduction="none")
    total = loss.sum() if real_count else loss.sum() * 0
    return total * rt.world / len(indices), total.detach()


@torch.no_grad()
def validate(model, table, banks, args, rt):
    # Compare each target to its complete category gallery. Ties follow the split-file order.
    model.eval()
    stats = torch.zeros(len(banks), 4, dtype=torch.float64, device=rt.device)
    for c, bank in enumerate(banks):
        all_ids = torch.where(table["category"] == c)[0]
        local = all_ids[rt.rank :: rt.world]
        for ids in local.split(128):
            if not len(ids):
                continue
            query = model(table["base"][ids], table["feature"][ids], table["text"][ids])
            scores = query @ bank.T
            target = table["pos_id"][ids]
            if args.dataset == "cirr":
                scores.scatter_(1, table["ref_id"][ids, None], -torch.inf)
            target_scores = scores.gather(1, target[:, None])
            ahead = (scores > target_scores) | (
                (scores == target_scores) & (torch.arange(len(bank), device=rt.device) < target[:, None])
            )
            ranks = 1 + ahead.sum(1)
            win = (query * (table["positive"][ids] - table["negative"][ids])).sum(1) > 0
            k1, k2 = (1, 5) if args.dataset == "cirr" else (10, 50)
            stats[c] += torch.stack(
                [
                    (ranks <= k1).sum(),
                    (ranks <= k2).sum(),
                    win.sum(),
                    torch.tensor(len(ids), device=rt.device),
                ]
            )
    if rt.world > 1:
        dist.all_reduce(stats)
    rates = (100 * stats[:, :3] / stats[:, 3:]).mean(0).tolist()
    return dict(recall=rates[0], recall_secondary=rates[1], pair_accuracy=rates[2])


def artifacts(folder, state):
    best = folder / "best.pt"
    if not best.exists() or load(best)["epoch"] != state["best_epoch"]:
        save(best, dict(config=state["config"], head=state["best_head"], epoch=state["best_epoch"]))
    history = state["history"]
    write_json(folder / "history.json", history)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Train loss")
    for k in ("recall", "pair_accuracy"):
        axes[1].plot(epochs, [r[k] for r in history], label=k)
    axes[1].set(xlabel="Epoch", ylabel="Validation (%)")
    axes[1].legend()
    fig.savefig(folder / "curves.png", dpi=150)
    plt.close(fig)


def train(args, rt):
    if args.backbone == "release" and args.feature != "global":
        raise ValueError("CLS and patch readouts require ViT")
    if args.loss in ("necessity", "shuffled") and not (args.review_label or args.paired_color):
        raise ValueError("Necessity supervision needs reviewed training interventions")
    train_data, _, train_id, weights = make_tables(args, "train")
    val_data, banks, val_id, val_weights = make_tables(args, "val")
    if weights != val_weights:
        raise ValueError("Train and val encoder weights differ")
    n, dim = train_data["base"].shape
    if n < 2:
        raise ValueError("Training requires at least two queries")
    cfg = dict(
        version=1,
        seed=args.seed,
        dim=dim,
        width=args.width,
        gate=args.gate,
        feature=args.feature,
        dataset=args.dataset,
        backbone=args.backbone,
        layer=args.layer,
        weights=weights,
        train=train_id,
        val=val_id,
        global_batch=args.batch_size,
        lr=args.lr,
        loss=args.loss,
        weight=args.weight,
        temperature=args.temperature,
    )
    folder = ROOT / "checkpoints/diagnosis" / run_key(args) / args.name
    prepare_run(folder, cfg, rt, args.clean)
    torch.manual_seed(args.seed)
    model = make_head(cfg).to(rt.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    checkpoint = folder / "last.pt"
    history, best_score, best_head, best_epoch = [], -math.inf, None, 0
    if checkpoint.exists():
        state = load(checkpoint)
        model.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_score, best_head = state["history"], state["best_score"], state["best_head"]
        best_epoch = state["best_epoch"]
        if rt.rank == 0:
            artifacts(folder, state)
    if rt.world > 1:
        model_ddp = DistributedDataParallel(model, device_ids=[rt.device.index])
    else:
        model_ddp = model
    tr = {k: v.to(rt.device) for k, v in train_data.items()}
    va = {k: v.to(rt.device) for k, v in val_data.items()}
    banks = [bank.to(rt.device) for bank in banks]
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).to(rt.device)
    for epoch in range(len(history), args.epochs):
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(rt.device)
        torch.manual_seed(args.seed + epoch * rt.world + rt.rank)
        model_ddp.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).to(rt.device)
        losses = torch.zeros((), device=rt.device)
        chunks = order.tensor_split(math.ceil(n / args.batch_size))
        for indices in tqdm(chunks, desc=f"Epoch {epoch + 1}", disable=rt.rank != 0):
            optimizer.zero_grad(set_to_none=True)
            loss, total = batch_loss(model_ddp, tr, indices, args, rt, permutation)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            losses += total
        if rt.world > 1:
            dist.all_reduce(losses)
        metrics = validate(model, va, banks, args, rt)
        if rt.rank == 0:
            row = dict(
                epoch=epoch + 1,
                loss=float(losses / n),
                **metrics,
                seconds=time.perf_counter() - start,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                rank0_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            history.append(row)
            score = metrics["pair_accuracy" if args.loss == "probe" else "recall"]
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_head = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state = dict(
                config=cfg,
                head=model.state_dict(),
                optimizer=optimizer.state_dict(),
                history=history,
                best_score=best_score,
                best_head=best_head,
                best_epoch=best_epoch,
            )
            save(checkpoint, state)
            artifacts(folder, state)
            print(row, flush=True)
        rt.sync()


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--feature", choices=["global", "cls", "patch"], default="global")
    p.add_argument("--gate", action="store_true")
    p.add_argument(
        "--loss",
        choices=["combiner", "retrieval", "probe", "hard", "necessity", "shuffled"],
        default="retrieval",
    )
    p.add_argument("--review-label", help="Train on the same reviewed pairs for all matched controls")
    p.add_argument(
        "--paired-color",
        action="store_true",
        help="Use annotated sibling targets and automated reference desaturation",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256, help="Global queries per optimizer step")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--weight", type=float, default=1.0)
    args = p.parse_args()
    if args.paired_color and (
        args.review_label or args.feature != "patch" or args.backbone != "vit_cir"
    ):
        p.error("--paired-color requires a ViT-CIR patch readout and an independent automated pair pool")
    if args.loss == "combiner" and (
        args.backbone != "vit"
        or args.name != "combiner"
        or args.gate
        or args.feature != "global"
        or args.review_label
    ):
        p.error(
            "The task Combiner requires --name combiner --backbone vit --feature global with all training pairs"
        )
    if (
        not args.name.isidentifier()
        or (args.review_label and not args.review_label.isidentifier())
        or min(args.width, args.epochs, args.batch_size) < 1
        or min(args.lr, args.temperature) <= 0
    ):
        p.error("Invalid run name or hyperparameters")
    rt = Runtime()
    train(args, rt)
    rt.close()
