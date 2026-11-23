"""Download OpenAI ViT-B/16 and the two official RN50x4_fullft task checkpoints per dataset."""

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from baseline.model import VIT, VIT_SHA, WEIGHTS, checkpoint_paths, digest

RELEASE = "https://github.com/ABaldrati/CLIP4Cir/releases/download/weights"


def download(url, path):
    if path.is_file():
        return
    part = path.with_suffix(path.suffix + ".partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(part),
            url,
        ],
        check=True,
    )
    part.replace(path)


def release_weights():
    parts_dir = WEIGHTS / "release_parts"
    parts_dir.mkdir(exist_ok=True)
    parts = []
    for i in range(5):
        name = f"pretrained_models.tar.gz.part-{i:03}"
        path = parts_dir / name
        download(f"{RELEASE}/{name}", path)
        parts.append(path)
    # Stream the split archive through tar, without writing a second multi-GB archive.
    with subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE) as cat:
        with tarfile.open(fileobj=cat.stdout, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or len(name.parts) < 3:
                    continue
                dataset, variant, filename = name.parts[-3:]
                if dataset not in ("FashionIQ", "CIRR") or variant != "RN50x4_fullft":
                    continue
                if not any(k in filename for k in ("_clip_", "_comb_")):
                    continue
                dest = WEIGHTS / dataset / variant / filename
                if dest.exists() and dest.stat().st_size == member.size:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".extracting")
                with archive.extractfile(member) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
                if tmp.stat().st_size != member.size:
                    raise OSError(f"Incomplete checkpoint: {dest}")
                tmp.replace(dest)
                print(dest, flush=True)
        # tar may reach its end marker before cat has drained all gzip padding.
        while cat.stdout.read(1024 * 1024):
            pass
        if cat.wait():
            raise OSError("Could not read all five release parts")
    for dataset in ("cirr", "fashioniq"):
        print(*checkpoint_paths(dataset, "release"), sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["both", "release", "vit"], default="both")
    args = parser.parse_args()
    WEIGHTS.mkdir(exist_ok=True, parents=True)
    if args.models in ("both", "vit"):
        download(f"https://openaipublic.azureedge.net/clip/models/{VIT_SHA}/ViT-B-16.pt", VIT)
        if digest(VIT) != VIT_SHA:
            raise ValueError(f"Checksum mismatch: {VIT}")
        print(VIT)
    if args.models in ("both", "release"):
        release_weights()
