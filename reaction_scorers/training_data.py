"""RGP examples, negative-type weights, and graph batching."""

import torch
from torch.utils.data import Dataset


def graph_sample_weights(frame):
    """Give generated negatives half the relative weight.

    Positives and forward negatives share the other half in proportion to their
    counts. An overall factor of 2/3 scales the weighted mean loss.
    """
    counts = frame["datum_type"].value_counts()
    types = {"positive", "negative-forward", "negative-generated"}
    if set(counts.index) != types:
        raise ValueError(f"Expected datum types {sorted(types)}.")
    remaining = counts["positive"] + counts["negative-forward"]
    proportions = {
        "negative-generated": 0.5,
        "positive": 0.5 * counts["positive"] / remaining,
        "negative-forward": 0.5 * counts["negative-forward"] / remaining,
    }
    return {
        kind: len(frame) / (counts[kind] * len(types)) * proportion * 2
        for kind, proportion in proportions.items()
    }


class RGPTrainingDataset(Dataset):
    """Each example is a graph, a binary label, and its datum-type loss weight."""

    def __init__(self, frame, featurizer, weights):
        self.frame = frame.reset_index(drop=True)
        self.featurizer, self.weights = featurizer, weights

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        graph = self.featurizer.batch([row.substrates], [row["product"]])
        return {
            "x": {key: graph[key][0] for key in ("atom", "bond")},
            "y": float(row["class"]),
            "weight": self.weights[row.datum_type],
        }


def collate_rgp(batch):
    """Pad graphs, limiting batch size to a 16,384 padded-node budget."""
    if max(item["x"]["atom"].shape[0] for item in batch) * len(batch) > 16384:
        ordered = sorted(batch, key=lambda item: item["x"]["atom"].shape[0])
        batch = [ordered[0]]
        for item in ordered[1:]:
            if item["x"]["atom"].shape[0] * (len(batch) + 1) > 16384:
                break
            batch.append(item)
    nodes = max(item["x"]["atom"].shape[0] for item in batch)
    first = batch[0]["x"]
    atoms = torch.zeros(len(batch), nodes, first["atom"].shape[-1])
    bonds = torch.zeros(len(batch), nodes, nodes, first["bond"].shape[-1])
    for i, item in enumerate(batch):
        size = item["x"]["atom"].shape[0]
        atoms[i, :size] = item["x"]["atom"]
        bonds[i, :size, :size] = item["x"]["bond"]
    return {
        "x": {"atom": atoms, "bond": bonds},
        "y": torch.tensor([item["y"] for item in batch]),
        "weight": torch.tensor([item["weight"] for item in batch]),
    }
