"""Download OpenAI ViT-B/16 and the two official RN50x4_fullft task checkpoints per dataset."""

import argparse
import io
import shutil
import subprocess
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path

from baseline.model import VIT, VIT_SHA, WEIGHTS, checkpoint_paths, digest

RELEASE = "https://github.com/ABaldrati/CLIP4Cir/releases/download/weights"
# GitHub release asset digests, in the author's split-file order.
PART_SHA256 = (
    "7a70c0e9f52edbfb2c479b747430391f25bba2f34a01d292fbd672a955964106",
    "8a976c834d36f206f8b3c5782b6abbb33331589313de5fe602d530def643fcb3",
    "849e141225607e096f74748527cc3acbb936f71109779eca0e09f7ab2b16378e",
    "45098bbf6d01c09856cdcce745a301458a1243f2b33904318f5c38cd5f5df89c",
    "8f999d94ed58edb7b3fe20bf01beb534a9c6f95d98a40b546793fd9655d10a1c",
)


def download(url, path, expected):
    if path.is_file():
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        return
    part = path.with_suffix(path.suffix + ".partial")
    command = [
        "curl",
        "--fail",
        "--location",
        "--connect-timeout",
        "20",
        "--speed-limit",
        "1024",
        "--speed-time",
        "90",
        "--retry",
        "2",
        "--continue-at",
        "-",
        "--output",
        str(part),
        url,
    ]
    for attempt in range(4):
        # A fresh curl process resumes at the current on-disk length after a reset.
        result = subprocess.run(command)
        if result.returncode == 0:
            break
        if result.returncode not in (7, 18, 28, 35, 52, 55, 56, 92) or attempt == 3:
            result.check_returncode()
        print(f"{path.name}: curl error {result.returncode}, retrying download", flush=True)
        time.sleep(2**attempt)
    if digest(part) != expected:
        raise ValueError(f"Checksum mismatch: {part}; replace this file before retrying")
    part.replace(path)


class JoinedParts(io.RawIOBase):
    """Expose the release volumes as one sequential binary stream."""

    def __init__(self, streams):
        super().__init__()
        self.streams = iter(streams)
        self.current = next(self.streams, None)

    def readable(self):
        return True

    def readinto(self, buffer):
        while self.current is not None:
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current = next(self.streams, None)
        return 0


def release_weights():
    parts_dir = WEIGHTS / "release_parts"
    parts_dir.mkdir(exist_ok=True)
    parts = []
    for i, expected in enumerate(PART_SHA256):
        name = f"pretrained_models.tar.gz.part-{i:03}"
        path = parts_dir / name
        download(f"{RELEASE}/{name}", path, expected)
        parts.append(path)
    # Stream the split archive through tar, without writing a second multi-GB archive.
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open("rb")) for p in parts]
        with (
            io.BufferedReader(JoinedParts(streams)) as reader,
            tarfile.open(fileobj=reader, mode="r|gz") as archive,
        ):
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
    for dataset in ("cirr", "fashioniq"):
        print(*checkpoint_paths(dataset, "release"), sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["both", "release", "vit"], default="both")
    args = parser.parse_args()
    WEIGHTS.mkdir(exist_ok=True, parents=True)
    if args.models in ("both", "vit"):
        download(f"https://openaipublic.azureedge.net/clip/models/{VIT_SHA}/ViT-B-16.pt", VIT, VIT_SHA)
        print(VIT)
    if args.models in ("both", "release"):
        release_weights()
