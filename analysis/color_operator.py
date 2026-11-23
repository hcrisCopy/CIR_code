"""Estimate signed color dependence from train captions with explicit operator labels."""

import random
import torch
from torch.nn import functional as F

from baseline.data import CATEGORIES, ROOT, layout
from baseline.model import Encoder, checkpoint_paths, digest
from baseline.runtime import Runtime, parser, save
from .paired_edits import COLOR


def checkpoint(args):
    return ROOT / "checkpoints/operators" / args.dataset / "color_text_ridge.pt"


def coefficient(text, state):
    x = F.normalize(text.float()) - state["center"].to(text.device)
    return (state["mean"] + x @ state["weight"].to(text.device)).clamp(-1, 1)


def compose_signed(text, clean, gray, state):
    a = coefficient(text, state).unsqueeze(1)
    return F.normalize(gray + a * (clean - gray))


def train(args):
    rows = []
    for cat in CATEGORIES[args.dataset]:
        _, qs = layout(args.dataset, cat, "train")
        rows.extend(
            q["text"] for q in qs if not COLOR.search(q["text"]) and len(q["text"].split()) <= 19
        )
    rng = random.Random(42)
    prefixes = rng.sample(sorted(set(rows)), 192)
    templates = [
        (
            "preserve its hue whether {a} or {b}",
            "use {a} regardless of whether the original was {a} or {b}",
            "replace {a} with {b} and vice versa",
        ),
        (
            "match the source hue",
            "give it a {a} hue",
            "use the opposite of its present hue among {a} and {b}",
        ),
        ("leave its coloring untouched", "paint it {a}", "swap the two possible hues {a} and {b}"),
        (
            "whichever of {a} and {b} it is, retain it",
            "choose {a} over {b} every time",
            "if currently {a}, select {b}; otherwise select {a}",
        ),
    ]
    colors = ["yellow", "orange", "brown", "pink"]
    captions, labels = [], []
    for i, prefix in enumerate(prefixes):
        a, b = rng.sample(colors, 2)
        for template, label in zip(templates[i % len(templates)], [1.0, 0.0, -1.0]):
            captions.append(prefix + ", " + template.format(a=a, b=b))
            labels.append(label)
        # Conservative prior for training backgrounds whose color intent is unspecified.
        captions.append(prefix)
        labels.append(1.0)
    rt = Runtime()
    files = checkpoint_paths(args.dataset, "vit")
    encoder = Encoder(files, rt.device, args.layer)
    with torch.inference_mode():
        x = (
            F.normalize(
                torch.cat(
                    [encoder.texts(captions[a : a + 64]) for a in range(0, len(captions), 64)]
                ).float()
            )
            .cpu()
            .double()
        )
    y = torch.tensor(labels, dtype=x.dtype)
    center = x.mean(0)
    x -= center
    weight = torch.linalg.solve(
        x.T @ x + 0.1 * torch.eye(x.shape[1], dtype=x.dtype), x.T @ (y - y.mean())
    )
    path = checkpoint(args)
    path.parent.mkdir(exist_ok=True, parents=True)
    save(
        path,
        dict(
            center=center.float(),
            mean=float(y.mean()),
            weight=weight.float(),
            encoder=digest(files[0]),
            dataset=args.dataset,
            count=len(y),
        ),
    )
    rt.close()
    print(path, len(y))


if __name__ == "__main__":
    train(parser(__doc__).parse_args())
