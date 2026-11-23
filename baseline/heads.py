import torch
import torch.nn.functional as F
from torch import nn

from .combiner import Combiner
from .model import digest
from .runtime import load


class ResidualHead(nn.Module):
    """A matched-width diagnostic head on a frozen CIR query."""

    def __init__(self, dimension, width, gate):
        super().__init__()
        self.down = nn.Linear(dimension, width, bias=False)
        self.up = nn.Linear(width, dimension, bias=False)
        nn.init.zeros_(self.up.weight)
        self.gate = gate

    def forward(self, base, feature, text):
        x = self.down(F.normalize(feature))
        if self.gate:
            gate = self.down(F.layer_norm(text, (text.shape[-1],)))
            x = x * (2 * torch.sigmoid(gate))
        return F.normalize(base + self.up(x))


class CachedCombiner(Combiner):
    def forward(self, base, feature, text):
        return self.combine_features(feature, text)


def make_head(config):
    dim = config["dim"]
    if config["loss"] == "combiner":
        return CachedCombiner(dim, 4 * dim, 8 * dim)
    return ResidualHead(dim, config["width"], config["gate"])


def read_head(path, args, weights, device):
    if path is None:
        return None, "global", "base"
    state = load(path)
    config = state["config"]
    if (
        config["weights"] != weights
        or config["dataset"] != args.dataset
        or config["backbone"] != args.backbone
        or config["layer"] != args.layer
    ):
        raise ValueError(f"Head and encoder configuration differ: {path}")
    head = make_head(config)
    head.load_state_dict(state["head"])
    return head.eval().to(device), config["feature"], digest(path)


def compose(head, base, features, text):
    return base if head is None else head(base, features, text)
