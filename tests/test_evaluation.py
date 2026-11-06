import numpy as np
import pandas as pd
import pytest

from reaction_scorers.evaluate import evaluate_frame



def test_missing_score_does_not_change_positive_label_definition():
    data = pd.DataFrame({"class": [1, 0, 1], "score": [0.9, 0.1, np.nan]})
    result = evaluate_frame(data)
    assert result["full"]["positives"] == result["full"]["negatives"] == 1
    assert result["nonfinite_scores"] == 1
    assert result["full"]["roc_auc"] == result["full"]["pr_auc_ap"] == 1.0


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        evaluate_frame(pd.DataFrame({"class": [0, 2], "score": [0.1, 0.2]}))
