"""Categorical atom and bond features used by the USPTO RGP checkpoint.

The order is part of the frozen vocabulary: supernode flag, atomic number,
formal charge, degree, ring membership, and total hydrogens. Bond features
distinguish ordinary bonds, self edges, and edges to the supernode.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def atom_features(atom):
    if atom is None or isinstance(atom, str):
        return np.asarray([atom] * 6)
    getters = (
        lambda: "no",
        atom.GetAtomicNum,
        atom.GetFormalCharge,
        lambda: atom.GetDegree() if atom.HasOwningMol() else 0,
        lambda: int(atom.IsInRing()) if atom.HasOwningMol() else 0,
        lambda: atom.GetTotalNumHs() if atom.HasOwningMol() else 0,
    )
    values = []
    for getter in getters:
        try:
            values.append(getter())
        except Exception as exc:
            logger.debug("Could not read atom feature: %s", exc)
            values.append(None)
    return np.asarray(values)


def bond_features(bond):
    return np.asarray([bond if bond is None or isinstance(bond, str) else "bond"])
