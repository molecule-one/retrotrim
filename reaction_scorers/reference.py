"""Reference scores and mapped-reaction encoding."""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def reference_score(reference_count):
    if not isinstance(reference_count, (int, np.integer)) or reference_count < 0:
        raise ValueError("reference_count must be a nonnegative integer.")
    return math.log1p(reference_count)


def encode_molecule(mapped_smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    mappings = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if any(value <= 0 for value in mappings):
        raise ValueError("Reference lookup requires completely mapped reactions.")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(molecule)
    # RDKit 2022.9 emits a Python-style list with a trailing comma.
    order = ast.literal_eval(molecule.GetProp("_smilesAtomOutputOrder"))
    return smiles, [mappings[index] for index in order]


def encode_reaction(substrates, product):
    """Assign unique atom maps and encode both reaction sides for index lookup."""
    from rdkit import Chem

    substrate_molecules = [
        Chem.MolFromSmiles(smiles) for smiles in substrates.split(".")
    ]
    product_molecule = Chem.MolFromSmiles(product)
    molecules = [*substrate_molecules, product_molecule]
    if any(molecule is None or molecule.GetNumAtoms() == 0 for molecule in molecules):
        raise ValueError("Invalid or empty reaction SMILES")
    next_map = (
        max(
            atom.GetAtomMapNum()
            for molecule in molecules
            for atom in molecule.GetAtoms()
        )
        + 1
    )
    repeated = {}
    for side in (substrate_molecules, [product_molecule]):
        occurrences = {}
        for molecule in side:
            for atom in molecule.GetAtoms():
                original = atom.GetAtomMapNum()
                occurrence = occurrences.get(original, 0)
                occurrences[original] = occurrence + 1
                if original == 0:
                    replacement = next_map
                    next_map += 1
                elif occurrence > 0:
                    key = (original, occurrence)
                    if key not in repeated:
                        repeated[key] = next_map
                        next_map += 1
                    replacement = repeated[key]
                else:
                    continue
                if replacement > 65535:
                    raise ValueError("Synthetic atom map exceeds u16 range")
                atom.SetAtomMapNum(replacement)

    def encode(molecule):
        maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(molecule)
        order = ast.literal_eval(molecule.GetProp("_smilesAtomOutputOrder"))
        return smiles, [maps[index] for index in order]

    encoded_substrates = [encode(molecule) for molecule in substrate_molecules]
    encoded_product = encode(product_molecule)
    return {
        "substrates": [smiles for smiles, _ in encoded_substrates],
        "substrates_to_map": [maps for _, maps in encoded_substrates],
        "product": encoded_product[0],
        "product_to_map": encoded_product[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.counts)
    data["score"] = [reference_score(value) for value in data["reference_count"]]
    if args.output:
        data.to_csv(args.output, index=False)
    from .evaluate import evaluate_frame

    print(json.dumps(evaluate_frame(data), indent=2))


if __name__ == "__main__":
    main()
