import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from reaction_scorers.rp_loss import token_loss  # noqa: E402
from reaction_scorers.schedules import warmup_and_cosine_decay  # noqa: E402
from reaction_scorers.training_data import graph_sample_weights  # noqa: E402


def test_token_loss_ignores_padding_and_has_gradients():
    logits = torch.tensor([[[0.0, 2.0, -1.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    loss = token_loss(logits, torch.tensor([[1, 0]]))
    assert loss.item() == pytest.approx(-torch.log_softmax(logits[0, 0], -1)[1].item())
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() == 0


def test_archived_graph_weighting_balances_generated_negatives():
    data = pd.DataFrame(
        {
            "datum_type": ["positive"] * 2
            + ["negative-forward"]
            + ["negative-generated"] * 9
        }
    )
    weights = graph_sample_weights(data)
    assert weights["positive"] == pytest.approx(weights["negative-forward"])
    generated_weight = 9 * weights["negative-generated"]
    remaining_weight = 2 * weights["positive"] + weights["negative-forward"]
    assert generated_weight == pytest.approx(remaining_weight)
    assert (generated_weight + remaining_weight) / len(data) == pytest.approx(2 / 3)


def test_rp_schedule_uses_the_archived_scaled_step_count():
    def rate(step):
        return warmup_and_cosine_decay(
            step, warmup_steps=3000, total_steps=495944, decay_to=0.01
        )

    assert rate(0) == 0
    assert rate(3000) == 1
    assert rate(495944) == pytest.approx(0.01)
