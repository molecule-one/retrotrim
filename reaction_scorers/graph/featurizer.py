"""Convert mapped reactions into the checkpoint's frozen one-hot graph features."""

import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .multi_graph import ReactionMultiGraph


def _one_hot(values, categories):
    # Index zero encodes missing/unknown values as an all-zero vector.
    result = np.zeros((values.size, categories + 1), dtype=int)
    result[np.arange(values.size), values.reshape(-1)] = 1
    return result[:, 1:].reshape(*values.shape, categories)


def graph_to_oh(nodes, edges, atom_dimensions, bond_dimensions):
    atoms = np.concatenate(
        [_one_hot(nodes[..., i], size) for i, size in enumerate(atom_dimensions)],
        axis=-1,
    )
    bonds = np.concatenate(
        [_one_hot(edges[..., i], size) for i, size in enumerate(bond_dimensions)],
        axis=-1,
    )
    return atoms, bonds


class ReactionGraphFeaturizer:
    """Encode reactions using the feature dictionaries in the model assets."""

    def __init__(self, metadata_path, features_variant="bond-simple", supernode=True):
        if features_variant != "bond-simple":
            raise ValueError("The released RGP uses bond-simple features.")
        metadata = json.loads(Path(metadata_path).read_text())
        self.dimensions = metadata["unraveled_shape"]
        self.categories = metadata["value2num"]
        self.graph_builder = ReactionMultiGraph(supernode=supernode)

    @property
    def input_shape(self):
        return [[sum(self.dimensions["atom"])], [sum(self.dimensions["bond"])]]

    @staticmethod
    def _encode(values, dictionaries):
        return np.asarray(
            [
                [
                    dictionary.get(str(value), 0)
                    for value, dictionary in zip(row, dictionaries)
                ]
                for row in values
            ],
            dtype=int,
        )

    def batch(self, substrates, products):
        if len(substrates) != len(products) or len(substrates) == 0:
            raise ValueError("Provide equally sized, nonempty substrates and products.")
        graphs = []
        for substrate, product in zip(substrates, products):
            molecules = []
            for smiles in (substrate, product):
                try:
                    molecule = Chem.MolFromSmiles(smiles)
                except RuntimeError:
                    molecule = None
                molecules.append(
                    molecule if molecule is not None else Chem.MolFromSmiles("")
                )
            nodes, (rows, columns, edges) = self.graph_builder.generate(molecules)
            graphs.append(
                (
                    self._encode(nodes, self.categories["atom"]),
                    rows,
                    columns,
                    self._encode(edges, self.categories["bond"]),
                )
            )
        sizes = np.asarray([len(graph[0]) for graph in graphs])
        max_nodes = max(sizes)
        atoms = np.zeros(
            (len(graphs), max_nodes, len(self.dimensions["atom"])), dtype=int
        )
        bonds = np.zeros(
            (len(graphs), max_nodes, max_nodes, len(self.dimensions["bond"])), dtype=int
        )
        for i, (nodes, rows, columns, edges) in enumerate(graphs):
            atoms[i, : len(nodes)] = nodes
            for row, column, values in zip(rows, columns, edges):
                bonds[i, row, column] = values
        atoms, bonds = graph_to_oh(
            atoms, bonds, self.dimensions["atom"], self.dimensions["bond"]
        )
        return {
            "n_nodes": sizes,
            "atom": torch.tensor(atoms, dtype=torch.float),
            "bond": torch.tensor(bonds, dtype=torch.float),
        }
