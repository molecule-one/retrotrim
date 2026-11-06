"""Load RP and RGP weights and scoring configuration."""

import json
from functools import partial
from pathlib import Path

import torch

from .graph.featurizer import ReactionGraphFeaturizer
from .graph.gat import GAT
from .graph.gat_layer import MultiHeadGraphConvLayer
from .rp import ReactionPrior
from .tokenizer import ReactionTokenizer
from .transformer import ReactionTransformer


def load_weights(model, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_rp(path, device="cpu"):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    network = load_weights(
        ReactionTransformer(config["architecture"]), path / "weights.pt", device
    )
    return ReactionPrior(network, ReactionTokenizer(), config["scorer"], device)


class GraphScorer:
    """Apply the graph network and return sigmoid scores for each reaction."""

    def __init__(self, model, featurizer, device):
        self.model, self.featurizer, self.device = model, featurizer, device

    @torch.no_grad()
    def score(self, reactions):
        if not reactions:
            return []
        batch = self.featurizer.batch(
            [reaction["substrates"] for reaction in reactions],
            [reaction["product"] for reaction in reactions],
        )
        inputs = {key: batch[key].to(self.device) for key in ("atom", "bond")}
        return torch.sigmoid(self.model(inputs)).cpu().tolist()


def build_rgp(config):
    return GAT(
        **config["architecture"],
        atom_features=config["input_shape"][0][0],
        bond_features=config["input_shape"][1][0],
        conv_layer_cls=partial(MultiHeadGraphConvLayer, **config["attention"]),
    )


def load_rgp(path, device="cpu"):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    features = ReactionGraphFeaturizer(
        path / "feature_values.json", **config["featurizer"]
    )
    if features.input_shape != config["input_shape"]:
        raise ValueError("Saved graph metadata does not match checkpoint dimensions.")
    network = load_weights(build_rgp(config), path / "weights.pt", device)
    return GraphScorer(network, features, device)
