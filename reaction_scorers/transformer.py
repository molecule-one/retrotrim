"""Autoregressive BART network used by RP, for training and token likelihoods."""

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right


class ReactionTransformer(nn.Module):
    """Predict each reaction token given the preceding tokens and source prefix.

    Keep the nested `model` attribute: its name is part of the saved state dict.
    """

    def __init__(self, architecture):
        super().__init__()
        self.config = BartConfig(**architecture)
        self.config.use_cache = False
        self.model = BartForConditionalGeneration(self.config)

    def forward(self, batch):
        source = batch["source"]
        decoder_input = shift_tokens_right(
            batch["target"],
            pad_token_id=self.config.pad_token_id,
            decoder_start_token_id=self.config.decoder_start_token_id,
        )
        return self.model(
            input_ids=source.long(),
            attention_mask=(source != self.config.pad_token_id).float(),
            decoder_input_ids=decoder_input,
            return_dict=True,
        ).logits

    def token_log_probabilities(self, batch):
        """Return target-token log probabilities and the padding mask."""
        log_probabilities = torch.log_softmax(self(batch), dim=-1)
        target = batch["target"]
        values = log_probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return values, target == self.config.pad_token_id
