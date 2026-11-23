"""Read the CIRR rc2 and FashionIQ splits used by CLIP4Cir."""

import json
from pathlib import Path

ROOT = Path("../CIR_data")
DATA = ROOT / "datasets"
CATEGORIES = {"cirr": ("cirr",), "fashioniq": ("dress", "shirt", "toptee")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def layout(dataset, category, split):
    if dataset == "cirr":
        root = DATA / "cirr_dataset"
        meta, tag = root / "cirr", "rc2"
    else:
        root = meta = DATA / "fashionIQ_dataset"
        tag = category
    images = read(meta / "image_splits" / f"split.{tag}.{split}.json")
    if dataset == "cirr":
        paths = {name: root / path for name, path in images.items()}
    else:
        if len(set(images)) != len(images):
            raise ValueError(f"Duplicate image ID in {category}/{split}")
        paths = {name: root / "images" / f"{name}.png" for name in images}
    if any(not path.resolve().is_relative_to(root.resolve()) for path in paths.values()):
        raise ValueError(f"Image path outside {root}")
    records = read(meta / "captions" / f"cap.{tag}.{split}.json")
    queries = []
    for i, row in enumerate(records):
        if dataset == "cirr":
            reference, target = row["reference"], row["target_hard"]
            group = row["img_set"]["members"]
            if (
                reference not in group
                or target not in group
                or len(group) != len(set(group))
                or any(name not in paths for name in group)
            ):
                raise ValueError(f"Invalid CIRR group at {split}/{i}")
            caption, key = row["caption"], str(row["pairid"])
        else:
            reference, target = row["candidate"], row["target"]
            a, b = row["captions"]
            caption = f"{a.strip('.?, ').capitalize()} and {b.strip('.?, ')}"
            key, group = f"{category}-{i}", []
        if reference not in paths or target not in paths or reference == target or not caption.strip():
            raise ValueError(f"Invalid query {category}/{split}/{key}")
        queries.append(dict(id=key, reference=reference, target=target, text=caption, group=group))
    if not queries or len(set(q["id"] for q in queries)) != len(queries):
        raise ValueError(f"Empty or duplicate queries in {category}/{split}")
    return paths, queries
