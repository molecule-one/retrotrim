"""Fixed Molecular Transformer tokenization used by the RP checkpoint."""

import json
import re
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizer

# Bracketed atoms, element symbols, bonds, rings, and the '$' reaction separator.
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


class ReactionTokenizer(PreTrainedTokenizer):
    """Tokenize reactions with the model vocabulary and special tokens."""

    def __init__(self, max_seq_length=1024):
        self.max_seq_length = max_seq_length
        self.vocabulary = json.loads(
            Path(__file__).with_name("vocabulary.json").read_text()
        )
        self.token_ids = {token: i for i, token in enumerate(self.vocabulary)}
        super().__init__(
            model_max_length=max_seq_length,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token=">>",
            pad_token="<pad>",
            mask_token="*",
        )

    def get_vocab(self):
        return self.token_ids

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def _convert_id_to_token(self, index):
        return self.vocabulary[index]

    def _convert_token_to_id(self, token):
        return self.token_ids.get(token, self.token_ids["<unk>"])

    @staticmethod
    def tokenize_smiles(smiles):
        tokens = SMILES_PATTERN.findall(smiles)
        if "".join(tokens) != smiles:
            raise ValueError(f"Unrecognized SMILES fragment: {smiles}")
        return tokens

    def _tokenize(self, text, **kwargs):
        parts = text.split(">>")
        tokens = []
        for part in parts[:-1]:
            tokens.extend(self.tokenize_smiles(part))
            tokens.append(">>")
        tokens.extend(self.tokenize_smiles(parts[-1]))
        return tokens[: self.max_seq_length]

    def _encode_sequence(self, text, source):
        tokens = np.asarray(self(text)["input_ids"])
        if source:
            tokens = np.insert(tokens, 0, self.bos_token_id)
        # Targets omit BOS: the BART network shifts them right itself.
        return np.append(tokens[: self.max_seq_length - 1], self.eos_token_id)

    def featurize_batch(self, batch):
        """Return padded source and target token arrays."""
        result = {}
        for key in ("source", "target"):
            sequences = [
                self._encode_sequence(text, source=key == "source")
                for text in batch[key]
            ]
            padded = np.full(
                (len(sequences), max(map(len, sequences))), self.pad_token_id, dtype=int
            )
            for i, sequence in enumerate(sequences):
                padded[i, : len(sequence)] = sequence
            result[key] = padded
        return result
