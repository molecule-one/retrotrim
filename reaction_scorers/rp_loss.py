"""RP token cross-entropy loss."""

from torch.nn import functional as F


def token_loss(logits, targets):
    """Average cross-entropy over non-padding tokens."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0
    )
