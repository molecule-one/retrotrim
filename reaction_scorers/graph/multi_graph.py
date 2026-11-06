"""Mapped substrate/product multigraph, with separate features for each side."""

import numpy as np

from .features import atom_features, bond_features


class ReactionMultiGraph:
    """Align atoms by map number, add self edges and an optional global node."""

    def __init__(self, supernode=True):
        self.supernode = supernode

    def generate(self, molecules):
        atom_indices, edge_indices = {}, {}
        if self.supernode:
            atom_indices[-1] = 0
            edge_indices[-1, -1] = 0

        # Preserve insertion order when assigning node and edge indices.
        for molecule in molecules:
            seen = set()
            for atom in molecule.GetAtoms():
                number = atom.GetAtomMapNum()
                if number < 0 or number in seen:
                    raise ValueError("Reaction has negative or duplicate atom maps.")
                seen.add(number)
                if number not in atom_indices:
                    atom_indices[number] = len(atom_indices)
                    edge_indices[number, number] = len(edge_indices)
                    if self.supernode:
                        edge_indices[-1, number] = len(edge_indices)
                        edge_indices[number, -1] = len(edge_indices)
            for bond in molecule.GetBonds():
                a, b = sorted(
                    (
                        bond.GetBeginAtom().GetAtomMapNum(),
                        bond.GetEndAtom().GetAtomMapNum(),
                    )
                )
                if (a, b) not in edge_indices:
                    edge_indices[a, b] = len(edge_indices)
                    edge_indices[b, a] = len(edge_indices)

        nodes = [[atom_features(None) for _ in molecules] for _ in atom_indices]
        edges = [[bond_features(None) for _ in molecules] for _ in edge_indices]
        rows = [atom_indices[a] for a, _ in edge_indices]
        columns = [atom_indices[b] for _, b in edge_indices]

        for side, molecule in enumerate(molecules):
            if self.supernode:
                nodes[atom_indices[-1]][side] = atom_features("supernode")
                edges[edge_indices[-1, -1]][side] = bond_features("supernode")
            for atom in molecule.GetAtoms():
                number = atom.GetAtomMapNum()
                nodes[atom_indices[number]][side] = atom_features(atom)
                edges[edge_indices[number, number]][side] = bond_features("self")
                if self.supernode:
                    edges[edge_indices[-1, number]][side] = bond_features("supernode")
                    edges[edge_indices[number, -1]][side] = bond_features("supernode")
            for bond in molecule.GetBonds():
                a, b = sorted(
                    (
                        bond.GetBeginAtom().GetAtomMapNum(),
                        bond.GetEndAtom().GetAtomMapNum(),
                    )
                )
                features = bond_features(bond)
                edges[edge_indices[a, b]][side] = features
                edges[edge_indices[b, a]][side] = features

        return (
            np.stack([np.concatenate(sides) for sides in nodes]),
            (rows, columns, [np.concatenate(sides) for sides in edges]),
        )
