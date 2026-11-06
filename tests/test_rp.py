import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rdkit")

from reaction_scorers.rp import ReactionPrior  # noqa: E402
from reaction_scorers.rp_params import ReactionLikelihood  # noqa: E402
from reaction_scorers.tokenizer import ReactionTokenizer  # noqa: E402


def test_center_sum_preserves_checkpoint_float32_reduction():
    scorer = ReactionPrior.__new__(ReactionPrior)
    scorer.config = {"reaction_center_size": 1}
    scorer.tokenizer = ReactionTokenizer()
    values = np.array([[1e8, 1, -1e8, 0, 0, 0, 0, 0, 99]], dtype=np.float32)
    likelihood = ReactionLikelihood(
        score=0,
        log_probability=0,
        token_log_probabilities=values,
        target="CCO$CC=O",
        substrates="CCO",
        product="CC=O",
        mapped_substrates="[CH3:1][CH2:2][OH:3]",
        mapped_product="[CH3:1][CH:2]=[O:3]",
    )
    score = scorer._center_score(likelihood)
    expected = values.copy()
    expected[0, -1] = 0  # EOS is excluded from center scoring.
    assert score.dtype == np.float32
    assert score == expected.sum()
    assert score != expected.astype(np.float64).sum()


def test_tokenizer_preserves_source_bos_and_target_eos():
    tokens = ReactionTokenizer().featurize_batch({"source": [""], "target": ["C$O"]})
    assert tokens["source"].tolist() == [[1, 2]]
    # '$' is encoded with the checkpoint's unknown-token ID.
    assert tokens["target"].tolist() == [[6, 3, 10, 2]]
