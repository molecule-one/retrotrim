"""Frozen normalization and diagnostics for RP scoring."""

from dataclasses import dataclass

import numpy as np


def score_normalizer(reference_scores):
    """Map the frozen 15th/60th percentiles to sigmoid(-4)/sigmoid(4)."""
    low = np.quantile(reference_scores, 0.15)
    high = np.quantile(reference_scores, 0.60)
    center = (high + low) / 2
    scale = (high - center) / 4
    return lambda score: 1 / (1 + np.exp(-(score - center) / scale))


@dataclass
class ReactionLikelihood:
    """Token likelihoods and canonical inputs needed for center scoring."""

    score: float
    log_probability: float
    token_log_probabilities: np.ndarray
    target: str
    substrates: str
    product: str
    mapped_substrates: str
    mapped_product: str
