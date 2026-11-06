"""Pad RP examples; zero is the checkpoint's padding token."""

import torch
from torch.nn.utils.rnn import pad_sequence


def _pad(sequences):
    tensors = [torch.as_tensor(sequence, dtype=torch.long) for sequence in sequences]
    padded = pad_sequence(tensors, batch_first=True, padding_value=0)
    # The trim length counts nonzero tokens, even when padding occurs
    # inside a sequence.
    length = torch.max(torch.sum(padded != 0, dim=-1))
    return padded[:, :length]


def collate_rp(batch):
    """Flatten per-example arrays and pad source, decoder target, and loss target."""
    return {
        "x": {
            key: _pad([row for item in batch for row in item["x"][key]])
            for key in ("source", "target")
        },
        "y": _pad([row for item in batch for row in item["y"]]),
    }
