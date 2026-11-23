"""Run the diagnostic experiments from CIR_code on a single Linux GPU server."""

import argparse
import subprocess
import sys

from baseline.data import CATEGORIES, ROOT


def invoke(module, options, gpus=0):
    command = [sys.executable, "-m"]
    if gpus:
        command += ["torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", "--module"]
    command += [module, *map(str, options)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def run(args, dataset):
    common = ["--dataset", dataset, "--limit", args.limit, "--layer", args.layer]
    clean = ["--clean"] if args.clean else []

    def cache(backbone, splits):
        invoke(
            "baseline.cache",
            [
                *common,
                "--backbone",
                backbone,
                "--splits",
                *splits,
                "--batch-size",
                args.encode_batch,
                "--workers",
                args.workers,
                *clean,
            ],
            args.gpus,
        )

    def evaluate(backbone, split, name="base", head=None):
        flags = ["--head", head] if head else []
        invoke(
            "evaluation.evaluate",
            [
                *common,
                "--backbone",
                backbone,
                "--split",
                split,
                "--label",
                name,
                "--batch-size",
                args.encode_batch,
                "--top-k",
                args.top_k,
                *flags,
                *clean,
            ],
            args.gpus,
        )

    def fit(backbone, name, loss, feature, flags=()):
        invoke(
            "analysis.train",
            [
                *common,
                "--backbone",
                backbone,
                "--name",
                name,
                "--loss",
                loss,
                "--feature",
                feature,
                "--width",
                args.width,
                "--epochs",
                args.epochs,
                "--batch-size",
                args.global_batch,
                "--lr",
                args.lr,
                "--temperature",
                args.temperature,
                "--weight",
                args.weight,
                *flags,
                *clean,
            ],
            args.gpus,
        )
        key = "all" if not args.limit else f"first{args.limit}"
        selected = "best.pt" if loss == "combiner" else "last.pt"
        return (
            ROOT
            / "checkpoints/diagnosis"
            / dataset
            / f"{backbone}-layer{args.layer}"
            / key
            / name
            / selected
        )

    if args.stage == "release":
        cache("release", args.splits)
        for split in args.splits:
            evaluate("release", split)
    elif args.stage == "baseline":
        cache("vit", ["train", "val"])
        for split in ("train", "val"):
            evaluate("vit", split)
        fit("vit", "combiner", "combiner", "global")
        cache("vit_cir", ["train", "val"])
        for split in ("train", "val"):
            evaluate("vit_cir", split)
    elif args.stage == "probe":
        for feature in ("global", "cls", "patch"):
            name = f"probe_{feature}"
            head = fit("vit_cir", name, "probe", feature)
            evaluate("vit_cir", "val", name, head)
    elif args.stage == "review":
        for split in args.splits:
            invoke(
                "analysis.review",
                [
                    *common,
                    "--backbone",
                    "vit_cir",
                    "--split",
                    split,
                    "--label",
                    "base",
                    "--per-group",
                    args.per_group,
                ],
            )
    elif args.stage == "intervene":
        for split in args.splits:
            invoke(
                "analysis.intervene",
                [
                    *common,
                    "--backbone",
                    args.backbone,
                    "--review-backbone",
                    "vit_cir",
                    "--split",
                    split,
                    "--label",
                    "base",
                    "--temperature",
                    args.temperature,
                    *clean,
                ],
                args.gpus,
            )
    elif args.stage == "compare":
        controls = [
            ("linear", "retrieval", False),
            ("gate", "retrieval", True),
            ("hard", "hard", True),
            ("shuffled", "shuffled", True),
            ("necessity", "necessity", True),
        ]
        for name, loss, gate in controls:
            flags = ["--review-label", "base"]
            if gate:
                flags.append("--gate")
            head = fit("vit_cir", name, loss, args.feature, flags)
            evaluate("vit_cir", "val", name, head)
        invoke(
            "analysis.report", [*common, "--backbone", "vit_cir", "--models", *[c[0] for c in controls]]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["release", "baseline", "probe", "review", "intervene", "compare"])
    p.add_argument("--dataset", choices=["both", *CATEGORIES], default="both")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--per-group", type=int, default=100)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--feature", choices=["global", "patch", "cls"], default="patch")
    p.add_argument(
        "--backbone",
        choices=["vit_cir", "release"],
        default="vit_cir",
        help="Encoder for the intervention stage on the same reviewed pairs",
    )
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()
    if min(args.gpus, args.encode_batch, args.global_batch, args.epochs) < 1 or args.limit < 0:
        p.error("Counts must be positive; --limit 0 means all queries")
    for dataset in CATEGORIES if args.dataset == "both" else (args.dataset,):
        run(args, dataset)
