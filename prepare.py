"""Unpack the two SPN4CIR image archives to the paths used by the split files."""

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from baseline.data import CATEGORIES, DATA, layout
from tqdm import tqdm


def copy_image(stream, dest, size, clean):
    if dest.is_file() and dest.stat().st_size == size and not clean:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".extracting")
    with tmp.open("wb") as out:
        shutil.copyfileobj(stream, out)
    if tmp.stat().st_size != size:
        raise IOError(f"Incomplete archive member for {dest}")
    os.replace(tmp, dest)


def unpack(dataset, clean):
    expected = {}
    for category in CATEGORIES[dataset]:
        for split in ("train", "val"):
            paths, _ = layout(dataset, category, split)
            for path in paths.values():
                if expected.setdefault(path.name, path) != path:
                    raise ValueError(f"Ambiguous image filename: {path.name}")
    seen = set()

    def accept(name):
        path = Path(name)
        if path.name not in expected or "__MACOSX" in path.parts:
            return None
        if path.name in seen:
            raise ValueError(f"Duplicate archive member for {path.name}")
        seen.add(path.name)
        return expected[path.name]

    if dataset == "cirr":
        with tarfile.open(DATA / "cirr_dataset/images.tar.gz", "r|gz") as archive:
            for member in tqdm(archive, desc="CIRR images", unit="entries"):
                if member.isfile() and (dest := accept(member.name)):
                    if not clean and dest.is_file() and dest.stat().st_size == member.size:
                        continue
                    with archive.extractfile(member) as src:
                        copy_image(src, dest, member.size, clean)
    else:
        with zipfile.ZipFile(DATA / "fashionIQ_dataset/images.zip") as archive:
            for member in tqdm(archive.infolist(), desc="FashionIQ images"):
                if not member.is_dir() and (dest := accept(member.filename)):
                    with archive.open(member) as src:
                        copy_image(src, dest, member.file_size, clean)
    missing = expected.keys() - seen
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required images absent from archive: {sorted(missing)[:10]}"
        )
    print(f"{dataset}: {len(seen)} train/val images ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*CATEGORIES, "both"], default="both")
    parser.add_argument("--clean", action="store_true", help="Re-extract the selected images")
    args = parser.parse_args()
    for name in CATEGORIES if args.dataset == "both" else [args.dataset]:
        unpack(name, args.clean)
