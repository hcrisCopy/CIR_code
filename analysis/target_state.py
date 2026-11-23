"""Resolve a relative color instruction into an absolute state on controlled diagnostic pairs."""

from collections import defaultdict
import torch
from torch.nn import functional as F
from tqdm import tqdm

from baseline.data import CATEGORIES, ROOT
from baseline.model import Encoder, checkpoint_paths, load_image
from baseline.runtime import Runtime, parser, run_key, write_json
from .controlled_color import PALETTES, examples, colorize
from .paired_edits import cluster_stat


@torch.inference_mode()
def evaluate(args, rt, enc, cat):
    paths, qs = examples(cat, "val", args.per_category)
    measurements = defaultdict(list)
    accuracy = []
    refs = []
    for palette in PALETTES.values():
        proto = F.normalize(enc.texts([f"a photo of a {color} garment" for color in palette[:2]]))
        for i in tqdm(range(0, len(qs), 16), desc=cat):
            rows = qs[i : i + 16]
            n = len(rows)

            def encode(field):
                pix = torch.stack([load_image(paths[q[field]], enc.preprocess) for q in rows]).to(
                    rt.device
                )
                return enc.clip.encode_image(torch.cat(colorize(pix, palette)))

            src = encode("reference")
            left, right = F.normalize(encode("target")).chunk(2)
            direction = (left - right).repeat(2, 1)
            observed = (F.normalize(src) @ proto.T).argmax(1)
            truth = torch.arange(2, device=rt.device).repeat_interleave(n)
            accuracy.extend((observed == truth).view(2, n).float().mean(0).tolist())
            names = [q["reference"] for q in rows]
            refs.extend(names)
            outputs = []
            for color in palette[:2]:
                text = enc.texts(
                    [q["text"] + ", and the target garment is " + color for q in rows]
                ).repeat(2, 1)
                outputs.append(
                    torch.stack(
                        (
                            (enc.compose(src, text) * direction).sum(1),
                            (F.normalize(text) * direction).sum(1),
                        ),
                        1,
                    )
                )
            margins = torch.stack(outputs, 1)
            row_idx = torch.arange(2 * n, device=rt.device)
            for info, current in [("known_source", truth), ("clip_source", observed)]:
                for mode in ["keep", "set0", "set1", "switch"]:
                    state = {
                        "keep": current,
                        "switch": 1 - current,
                        "set0": torch.zeros_like(current),
                        "set1": torch.ones_like(current),
                    }[mode]
                    target = {
                        "keep": truth,
                        "switch": 1 - truth,
                        "set0": torch.zeros_like(truth),
                        "set1": torch.ones_like(truth),
                    }[mode]
                    picked = margins[row_idx, state]
                    success = (
                        (picked * (1 - 2 * target[:, None]) > 0).reshape(2, n, 2).float().mean(0)
                    )
                    for col, encoder in enumerate(["composed", "text"]):
                        measurements[info, encoder, mode].extend(success[:, col].tolist())
    folder = ROOT / "results/target_state" / run_key(args) / cat
    folder.mkdir(parents=True, exist_ok=True)
    stats = {"/".join(k): cluster_stat(v, refs) for k, v in measurements.items()}
    write_json(
        folder / "summary.json",
        dict(products=len(qs), source_color_accuracy=cluster_stat(accuracy, refs), methods=stats),
    )
    print(cat, folder)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--per-category", type=int, default=128)
    args = p.parse_args()
    if args.dataset != "fashioniq" or args.backbone != "vit_cir":
        p.error("Use the FashionIQ Combiner")
    rt = Runtime()
    enc = Encoder(
        checkpoint_paths(args.dataset, args.backbone, args.layer, args.limit), rt.device, args.layer
    )
    for cat in CATEGORIES["fashioniq"]:
        evaluate(args, rt, enc, cat)
    rt.close()
