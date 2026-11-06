"""RP training examples: empty source, augmented substrates$product target.

Augmentation chooses a multicanonical representation, optionally snaps the
reaction, then optionally drops a substrate. It applies to both training and
validation examples.
"""

import random

import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset

from .chemistry.rdkit import is_mapped_reaction, smiles_to_unmapped
from .chemistry.smiles import smiles_to_canonical, smiles_to_multicanonical
from .chemistry.snapping import make_snapped_reaction
from .tokenizer import ReactionTokenizer


def reaction_size_prefix(radius):
    """The source prefix identifies the radius of a snapped reaction."""
    return "F" * (radius + 1)


class RPTrainingDataset(Dataset):
    """Augment reactions and tokenize them for next-token prediction."""

    def __init__(
        self,
        substrates,
        products,
        tokenizer=None,
        multicanonical_count=10,
        snap_probability=0.5,
        drop_probability=0.25,
    ):
        if len(substrates) != len(products):
            raise ValueError("Substrates and products must have the same length.")
        self.reactions = np.asarray(
            [s + "$" + p for s, p in zip(substrates, products)], dtype=object
        )
        self.tokenizer = tokenizer or ReactionTokenizer()
        self.multicanonical_count = multicanonical_count
        self.snap_probability = snap_probability
        self.drop_probability = drop_probability

    def __len__(self):
        return len(self.reactions)

    @staticmethod
    def _canonical_target(reaction):
        # Canonicalize each side separately so '$' never changes their order.
        return "$".join(smiles_to_unmapped(part) for part in reaction.split("$"))

    def _snap(self, reaction):
        radius = np.random.choice([0, 1, 2, 3, 4, 5])
        substrates, product = reaction.split("$")
        if (
            Chem.MolFromSmiles(substrates) is None
            or Chem.MolFromSmiles(product) is None
        ):
            return None
        substrates_mol, product_mol = make_snapped_reaction(
            substrates, product, radius, remove_radicals=True
        )
        if substrates_mol is None or product_mol is None:
            return None
        try:
            substrates = Chem.MolToSmiles(substrates_mol)
            product = Chem.MolToSmiles(product_mol)
        except RuntimeError:
            return None
        if not is_mapped_reaction(substrates, product):
            return None
        substrates = smiles_to_canonical(smiles_to_unmapped(substrates))
        product = smiles_to_canonical(smiles_to_unmapped(product))
        if not substrates or not product:
            return None
        return reaction_size_prefix(radius), substrates + "$" + product

    def __getitem__(self, index):
        reaction = self.reactions[index]
        source, target = "", self._canonical_target(reaction)

        if self.multicanonical_count > 1:
            root = random.randint(0, self.multicanonical_count - 1)
            target = "$".join(
                smiles_to_multicanonical(part, root) for part in target.split("$")
            )
            if root > 0:
                source = (
                    "#" * root + "."
                )  # The dot terminates the multicanonical source prefix.

        if self.snap_probability > 0 and random.random() < self.snap_probability:
            snapped = self._snap(reaction)
            if snapped is not None:
                source, target = snapped

        if self.drop_probability > 0 and random.random() < self.drop_probability:
            substrates, product = target.split("$")
            compounds = substrates.split(".")
            if len(compounds) > 1:
                compounds.pop(random.randint(0, len(compounds) - 1))
                target = ".".join(compounds) + "$" + product

        tokens = self.tokenizer.featurize_batch(
            {"source": [source], "target": [target]}
        )
        return {"x": tokens, "y": tokens["target"]}
