"""Reaction Prior: likelihood, competing reaction sites, and reaction center.

The final score is the product of three normalized components, each raised to
its saved weight. Also score the radius-2 snapped reaction and take the lower
score. Normalization uses the reference arrays in the model configuration.
"""

import logging

import numpy as np
import torch
from rdkit import Chem

from .chemistry.rdkit import (
    get_changed_atoms_indices_in_reaction_center,
    is_mapped_reaction,
    mapped_smiles_to_complete_mapped,
    smiles_to_unmapped,
)
from .chemistry.smiles import smiles_to_canonical
from .chemistry.snapping import make_snapped_reaction
from .chemistry.templates import generate_alternative_template_reactions
from .rp_dataset import reaction_size_prefix
from .rp_params import ReactionLikelihood, score_normalizer

logger = logging.getLogger(__name__)


class ReactionPrior:
    """Score mapped reactions with one RP network and its saved scoring settings."""

    def __init__(self, model, tokenizer, config, device="cpu"):
        if config["source_keys"] != [] or config["target_keys"] != [
            "substrates",
            "product",
        ]:
            raise ValueError(
                "RP expects an empty source and substrates$product target."
            )
        if config["score_normalization"] != "sqrt_output_length":
            raise ValueError(
                "The released RP uses square-root output-length normalization."
            )
        if config["n_test_time_augmentations"] != 1:
            raise ValueError("The released RP uses one deterministic scoring pass.")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.normalizers = {
            name: score_normalizer(config[key])
            for name, key in (
                ("prior", "reaction_prior_normalization_scores"),
                ("regio", "regioselectivity_normalization_scores"),
                ("center", "reaction_center_normalization_scores"),
            )
        }
        self.weights = {
            "prior": config["reaction_prior_weight"],
            "regio": config["regioselectivity_weight"],
            "center": config["reaction_center_weight"],
        }

    def _likelihood(self, reaction, source=""):
        substrates = smiles_to_unmapped(reaction["substrates"], sanitize=True)
        product = smiles_to_unmapped(reaction["product"], sanitize=True)
        # Canonicalize each side after removing atom maps.
        target = "$".join(map(smiles_to_canonical, (substrates, product)))
        tokens = self.tokenizer.featurize_batch(
            {"source": [source], "target": [target]}
        )
        tokens = {
            key: torch.tensor(value, device=self.device)
            for key, value in tokens.items()
        }
        values, padding = self.model.token_log_probabilities(tokens)
        values[padding] = 0
        log_probability = values.sum(dim=-1)[0].detach().cpu().numpy()
        return ReactionLikelihood(
            score=log_probability / np.sqrt(len(target)),
            log_probability=log_probability,
            token_log_probabilities=values.detach().cpu().numpy(),
            target=target,
            substrates=substrates,
            product=product,
            mapped_substrates=reaction["substrates"],
            mapped_product=reaction["product"],
        )

    def _regio_score(self, reaction, log_probability, source):
        alternatives = []
        # Stop at the first radius that produces alternative reactions.
        for radius in (2, 1, 0):
            try:
                alternatives = [
                    candidate
                    for candidate in generate_alternative_template_reactions(
                        reaction, rc_radius=radius, whole_rings=False
                    )
                    if candidate["product"] and candidate["substrates"]
                ]
            except Exception:
                continue
            if alternatives:
                break

        competing = []
        for candidate in alternatives:
            try:
                competing.append(self._likelihood(candidate, source).log_probability)
            except Exception as exc:
                logger.debug("Could not score alternative reaction: %s", exc)

        if not competing:
            return self.config["regioselectivity_default_regio_score"]
        count = 0
        for value in competing:
            ratio = value / log_probability
            if ratio < 3:
                count += min(2, 1 / ratio)
        return self.config["regioselectivity_default_regio_score"] / (count + 1)

    @staticmethod
    def _atom_token_mask(tokens, atom_ids):
        """Associate punctuation with the next atom in the token mask."""
        molecule = Chem.MolFromSmiles("".join(tokens))
        selected = set(atom_ids)
        mask, atom_index, prefix = [], 0, ""
        for token in tokens:
            if atom_index == molecule.GetNumAtoms():
                mask.append(0)
                continue
            mask.append(int(atom_index in selected))
            prefix += token.lower()
            symbol = molecule.GetAtomWithIdx(atom_index).GetSymbol().lower()
            if symbol in prefix:
                atom_index += 1
                prefix = ""
        assert atom_index == molecule.GetNumAtoms()
        return mask

    def _center_score(self, likelihood):
        if not likelihood.substrates or not likelihood.product:
            return None
        reaction, substrate_atoms, product_atoms = (
            get_changed_atoms_indices_in_reaction_center(
                likelihood.mapped_substrates,
                likelihood.mapped_product,
                center_size=self.config["reaction_center_size"],
                remove_mapping=True,
            )
        )
        substrates, product = reaction.split(">>")
        if not substrates or not product:
            return None
        substrate_tokens = self.tokenizer.tokenize_smiles(substrates)
        product_tokens = self.tokenizer.tokenize_smiles(product)
        assert substrates + "$" + product == likelihood.target
        assert likelihood.token_log_probabilities.shape == (
            1,
            len(substrate_tokens) + len(product_tokens) + 2,
        )
        mask = np.array(
            [
                self._atom_token_mask(substrate_tokens, substrate_atoms)
                + [1]  # Include '$', exclude the final EOS token.
                + self._atom_token_mask(product_tokens, product_atoms)
                + [0]
            ]
        )
        # In-place masking preserves float32. Out-of-place multiplication by
        # an integer mask promotes to float64 and changes near-tied scores.
        values = likelihood.token_log_probabilities.copy()
        values *= mask
        return np.sum(values[0])

    def _score_variant(self, reaction, source=""):
        try:
            likelihood = self._likelihood(reaction, source)
        except Exception as exc:
            logger.debug("Could not compute reaction likelihood: %s", exc)
            return 0.0, {"prior": 0.0, "regio": 0.0, "center": 0.0}
        raw = {
            "prior": likelihood.score,
            "regio": self._regio_score(reaction, likelihood.log_probability, source),
            "center": self._center_score(likelihood),
        }
        components = {
            name: self.normalizers[name](value) ** self.weights[name]
            if value is not None
            else 1.0
            for name, value in raw.items()
        }
        assert not any(np.isnan(value) for value in components.values())
        return components["prior"] * components["regio"] * components[
            "center"
        ], components

    def _snapped_variant(self, reaction):
        radius = self.config["evaluate_snapped_variant"]
        if radius < 0 or not is_mapped_reaction(
            reaction["substrates"], reaction["product"]
        ):
            return None
        substrates, product = make_snapped_reaction(
            reaction["substrates"], reaction["product"], radius, remove_radicals=True
        )
        if substrates is None or product is None:
            return None
        snapped = {
            "substrates": Chem.MolToSmiles(substrates),
            "product": Chem.MolToSmiles(product),
        }
        if not snapped["substrates"] or not snapped["product"]:
            return None
        return self._score_variant(snapped, reaction_size_prefix(radius))

    @torch.no_grad()
    def score_with_subscores(self, reactions):
        scores, components = [], []
        for reaction in reactions:
            substrates, product = mapped_smiles_to_complete_mapped(
                reaction["substrates"], reaction["product"]
            )
            mapped = {"substrates": substrates, "product": product}
            snapped = self._snapped_variant(mapped)
            score, subscores = self._score_variant(mapped)
            if snapped is not None and score > snapped[0]:
                score, subscores = snapped
            scores.append(float(score))
            components.append(subscores)
        return scores, components

    def score(self, reactions):
        return self.score_with_subscores(reactions)[0]
