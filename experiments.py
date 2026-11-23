"""Run single-GPU experiments on the two native datasets, one at a time."""

import argparse
import hashlib
import subprocess
import sys
from urllib.request import urlopen

from baseline.data import CATEGORIES, DATA

REVISION = "65dd92fa2f44f1818e4d0f2d44add342f9ce644c"
HASHES = {
    "cirr": [
        "0da622c55f2f03e20707a3478c1aff5c5e5550907b7c9db76abaef6ebf7d3308",
        "8be0378f29b2a04bd24514609f4e2ac68dd91435ffa8e6357d8e08e5d2716ddd",
    ],
    "fashioniq": [
        "18f829da4d3b1ee8439f43998bae43dbbe8924c2eaafba84473ae849f564f5c7",
        "ba019a30d3b45f4b29c34d2d48638dbeeed2e2a42409f625922c65de5b2a7efa",
    ],
}


def slices(dataset, download):
    folder = DATA / "circus_slices"
    folder.mkdir(parents=True, exist_ok=True)
    for stratum, checksum in zip(("solved", "unsolved"), HASHES[dataset]):
        filename = f"mbeir_{dataset}_task7_test_validated_{stratum}.jsonl"
        path = folder / filename
        if path.exists():
            raw = path.read_bytes()
        elif download:
            url = f"https://raw.githubusercontent.com/EdinburghNLP/CIRCUS/{REVISION}/final_dataset/query_jsonl/{dataset}/validated_{stratum}/{filename}"
            with urlopen(url, timeout=60) as response:
                raw = response.read()
        else:
            raise FileNotFoundError(f"{path}. Prepare with --prepare-slices first.")
        if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != checksum:
            raise ValueError(f"CIRCUS annotation checksum mismatch: {path}")
        if not path.exists():
            path.write_bytes(raw)


def execute(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def run(args, ds):
    slices(ds, args.prepare_slices)
    if args.prepare_slices:
        return
    common = ["--dataset", ds, "--layer", 6, "--limit", args.limit]
    base = [*common, "--backbone", "vit_cir"]
    for phase in ("baseline", "probe"):
        execute("run.py", phase, *common, "--gpus", 1)
    execute("-m", "analysis.color_views", *common, "--backbone", "vit", "--splits", "train", "val")
    execute("-m", "analysis.paired_edits", *base, "--split", "train")
    execute("-m", "analysis.color_operator", "--dataset", ds)
    heads = ["probe_global", "probe_cls", "probe_patch"]
    for feat in ("global", "cls", "patch"):
        name = "retrieval_" + feat
        execute(
            "-m", "analysis.train", *base, "--name", name, "--feature", feat, "--loss", "retrieval"
        )
        heads.append(name)
    for seed in args.seeds:
        for loss in (
            ("retrieval", "hard", "shuffled", "necessity")
            if seed == 42
            else ("hard", "shuffled", "necessity")
        ):
            name = "color_" + loss + (f"_s{seed}" if seed != 42 else "")
            execute(
                "-m",
                "analysis.train",
                *base,
                "--name",
                name,
                "--paired-color",
                "--feature",
                "patch",
                "--gate",
                "--loss",
                loss,
                "--seed",
                seed,
            )
            if seed == 42:
                heads.append(name)
    execute(
        "-m",
        "analysis.paired_edits",
        *base,
        "--split",
        "val",
        "--operator",
        "--center-reference",
        "--heads",
        *heads,
    )
    if args.limit == 0:
        execute("-m", "analysis.validation_slices", *base)
        if ds == "fashioniq":
            execute(
                "-m",
                "analysis.controlled_color",
                *base,
                "--per-category",
                args.products,
                "--heads",
                "probe_global",
                "probe_patch",
                "color_hard",
                "color_shuffled",
                "color_necessity",
            )
            for script in ("analysis.target_state", "analysis.position"):
                execute("-m", script, *base, "--per-category", args.products)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["both", *CATEGORIES], default="both")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--products", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prepare-slices", action="store_true")
    args = p.parse_args()
    if 42 not in args.seeds or args.limit < 0 or args.products < 1:
        p.error("Include primary seed 42 and use valid sample counts")
    for ds in CATEGORIES if args.dataset == "both" else [args.dataset]:
        run(args, ds)
