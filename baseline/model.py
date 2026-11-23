"""OpenAI CLIP features with CLIP4Cir TargetPad and checkpoint conventions."""

import hashlib
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from clip.model import build_model
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .combiner import Combiner

WEIGHTS = Path("../CIR_data/checkpoints")
VIT = WEIGHTS / "ViT-B-16.pt"
VIT_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_paths(dataset, backbone, layer=6, limit=0):
    if backbone == "vit":
        return [VIT]
    if backbone == "vit_cir":
        folder = (
            WEIGHTS / "diagnosis" / dataset / f"vit-layer{layer}" / (f"first{limit}" if limit else "all")
        )
        return [VIT, folder / "combiner/best.pt"]
    folder = WEIGHTS / ("CIRR" if dataset == "cirr" else "FashionIQ") / "RN50x4_fullft"
    files = []
    for tag in ("clip", "comb"):
        matches = sorted(p for p in folder.iterdir() if f"_{tag}_" in p.name and p.is_file())
        if len(matches) != 1:
            raise ValueError(f"Expected one _{tag}_ checkpoint in {folder}, found {matches}")
        files.append(matches[0])
    return files


def pad(image):
    w, h = image.size
    side = max(w, h) / 1.25
    x, y = max(int((side - w) / 2), 0), max(int((side - h) / 2), 0)
    return TF.pad(image, [x, y, x, y], fill=0)


def rgb(image):
    return image.convert("RGB")


class Encoder:
    def __init__(self, files, device, layer):
        self.device = device
        self.head = None
        self.layer = layer
        if files[0] == VIT:
            self.clip, _ = clip.load(str(files[0]), device="cpu", jit=False)
            if len(files) == 2:
                state = torch.load(files[1], map_location="cpu", weights_only=True)
                cfg = state["config"]
                if cfg["loss"] != "combiner" or cfg["weights"] != [VIT_SHA]:
                    raise ValueError("Expected a trained ViT-B/16 Combiner")
                dim = self.clip.visual.output_dim
                self.head = Combiner(dim, 4 * dim, 8 * dim)
                self.head.load_state_dict(state["head"])
        else:
            weights = torch.load(files[0], map_location="cpu", weights_only=True)["CLIP"]
            self.clip = build_model(weights).float()
            self.clip.load_state_dict(weights, strict=True)
            state = torch.load(files[1], map_location="cpu", weights_only=True)["Combiner"]
            dim = self.clip.visual.output_dim
            self.head = Combiner(
                dim,
                state["image_projection_layer.weight"].shape[0],
                state["output_layer.weight"].shape[1],
            )
            self.head.load_state_dict(state, strict=True)
        if self.head is not None:
            self.head.to(device).eval()
        self.clip.float().to(device).eval()
        size = self.clip.visual.input_resolution
        self.preprocess = transforms.Compose(
            [
                pad,
                transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                rgb,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

    def texts(self, captions):
        return self.clip.encode_text(clip.tokenize(captions).to(self.device)).float()

    def images(self, images):
        visual = self.clip.visual
        if not hasattr(visual, "transformer"):
            return {"global": self.clip.encode_image(images)}
        if not 1 <= self.layer <= len(visual.transformer.resblocks):
            raise ValueError(f"Invalid 1-based ViT block {self.layer}")
        features = {}

        def capture(module, inputs, tokens):
            # OpenAI CLIP uses [tokens, batch, width]. Apply the same frozen readout.
            normalized = visual.ln_post(tokens)
            features["cls"] = normalized[0] @ visual.proj
            features["patch"] = normalized[1:].mean(0) @ visual.proj

        hook = visual.transformer.resblocks[self.layer - 1].register_forward_hook(capture)
        try:
            features["global"] = self.clip.encode_image(images)
        finally:
            hook.remove()
        return features

    def compose(self, reference, text):
        if self.head is None:
            return F.normalize(reference + text)
        return self.head.combine_features(reference, text)


def load_image(path, transform):
    with Image.open(path) as image:
        return transform(image)
