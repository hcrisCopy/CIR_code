"""Read source and candidate states with train-only probes on frozen CLIP features."""

from collections import defaultdict
import torch
from torch.nn import functional as F
from tqdm import tqdm
from baseline.data import CATEGORIES, ROOT
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, parser, run_key, write_json
from .controlled_color import PALETTES, colorize, examples
from .position import position_views
from .paired_edits import cluster_stat


def encode(enc, rt, paths, rows, field, factor, palette):
    groups = defaultdict(list)
    for i in range(0, len(rows), 16):
        images = torch.stack(
            [load_image(paths[q[field]], enc.preprocess) for q in rows[i : i + 16]]
        ).to(rt.device)
        variants = position_views(images)[:2] if factor == "position" else colorize(images, palette)
        feat = enc.images(torch.cat(variants))
        for name, values in feat.items():
            groups[name].append(
                F.normalize(values).cpu().reshape(2, len(images), -1).transpose(0, 1)
            )
    return {k: torch.cat(v) for k, v in groups.items()}


def fit(x):
    x, labels = (
        x.flatten(0, 1).double(),
        torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(len(x)),
    )
    center = x.mean(0)
    x = x - center
    return center.float(), torch.linalg.solve(
        x.T @ x + 0.1 * torch.eye(x.shape[1], dtype=x.dtype), x.T @ labels
    ).float()


@torch.inference_mode()
def evaluate(args, rt, enc):
    out = {}
    for factor, palette in [("position", None), *PALETTES.items()]:
        train = defaultdict(list)
        for cat in CATEGORIES["fashioniq"]:
            paths, qs = examples(cat, "train", args.train_per_category)
            for name, t in encode(enc, rt, paths, qs, "target", factor, palette).items():
                train[name].append(t)
        weights = {k: fit(torch.cat(v)) for k, v in train.items()}
        rows = defaultdict(list)
        refs = []
        for cat in tqdm(CATEGORIES["fashioniq"], desc=factor):
            paths, qs = examples(cat, "val", args.per_category)
            src = encode(enc, rt, paths, qs, "reference", factor, palette)
            dst = encode(enc, rt, paths, qs, "target", factor, palette)
            refs.extend([q["reference"] for q in qs])
            nouns = ("left", "right") if factor == "position" else palette[:2]
            prompts = [
                f"a photo of a garment on the {s} side of the frame"
                if factor == "position"
                else f"a photo of a {s} garment"
                for s in nouns
            ]
            text = F.normalize(enc.texts(prompts)).cpu()
            for name in [*weights, "zero_shot"]:
                if name == "zero_shot":
                    sr, dt = (
                        src["global"] @ (text[0] - text[1]),
                        dst["global"] @ (text[0] - text[1]),
                    )
                else:
                    center, w = weights[name]
                    sr, dt = (src[name] - center) @ w, (dst[name] - center) @ w
                truth = torch.tensor([1.0, -1.0])
                rows[name + "/source_decode"].extend(((sr * truth) > 0).float().mean(1).tolist())
                rows[name + "/target_decode"].extend(((dt * truth) > 0).float().mean(1).tolist())
                delta = (dt[:, 0] - dt[:, 1])[:, None]
                for mode in ("keep", "set", "switch"):
                    pred = (sr.sign() if mode == "keep" else -sr.sign()) if mode != "set" else truth
                    desired = truth if mode != "switch" else -truth
                    rows[name + "/" + mode].extend(
                        ((pred * desired * delta) > 0).float().mean(1).tolist()
                    )
        out[factor] = {k: cluster_stat(v, refs) for k, v in rows.items()}
    folder = ROOT / "results/factor_readout" / run_key(args)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(
        folder / "summary.json",
        dict(
            train_products_per_category=args.train_per_category,
            val_products_per_category=args.per_category,
            methods=out,
        ),
    )
    print(folder)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--per-category", type=int, default=128)
    p.add_argument("--train-per-category", type=int, default=64)
    args = p.parse_args()
    if args.dataset != "fashioniq" or args.backbone != "vit_cir":
        p.error("Use FashionIQ ViT-CIR")
    rt = Runtime()
    enc = Encoder(
        checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit), rt.device, args.layer
    )
    evaluate(args, rt, enc)
    rt.close()
