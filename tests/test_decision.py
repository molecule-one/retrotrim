import numpy as np
import pandas as pd
import pytest

from reaction_scorers.decision import DecisionRule, evaluate_decisions, select_threshold, tune


def test_paper_gate_requires_both_models_and_precedent():
    rule = DecisionRule(rp_threshold=.75, rgp_threshold=.85)
    assert not rule.accepts({"rp_score": 0.8, "rgp_score": 0.1, "reference_count": 1})
    assert not rule.accepts({"rp_score": 0.1, "rgp_score": 0.9, "reference_count": 1})
    assert rule.accepts({"rp_score": 0.8, "rgp_score": 0.9, "reference_count": 1})
    assert not rule.accepts({"rp_score": 0.8, "rgp_score": 0.9, "reference_count": 0})
    assert not rule.accepts({"rp_score": 0.75, "rgp_score": 0.85, "reference_count": 1})


def test_disabled_scorers_need_no_values_and_single_scorers_work():
    rule = DecisionRule()
    assert rule.accepts({}, ())
    assert rule.accepts({"rp_score": 0.8}, ("rp",))
    assert rule.accepts({"reference_count": 1}, ("rrs",))
    assert not rule.accepts({"rp_score": np.nan}, ("rp",))
    assert not rule.accepts({"reference_count": np.nan}, ("rrs",))


def test_tuning_uses_all_rows_and_exports_only_aggregates():
    frame = pd.DataFrame({
        "class": [0, 1] * 25,
        "rp_score": [0.2, 0.8] * 25,
        "rgp_score": [0.3, 0.9] * 25,
        "reference_count": [1] * 50,
        "reaction_id": ['example-row'] * 50,
        "substrates": ['example-structure'] * 50,
    })
    result = tune(frame)
    assert result == tune(frame)
    assert result["calibration"]["rows"] == 50
    assert result["calibration"]["precision"] >= .8
    assert "example-row" not in str(result) and "example-structure" not in str(result)
    assert "validation" not in result
    assert evaluate_decisions(frame, DecisionRule(**result["thresholds"]))["balanced_accuracy"] == 1


def test_calibration_excludes_nonfinite_scores_per_model_and_applies_reference_gate():
    frame = pd.DataFrame({
        'class': [0, 0, 1, 1], 'rp_score': [np.nan, .2, .8, .9],
        'rgp_score': [.1, .2, .8, .9], 'reference_count': [1, 1, 1, 1],
    })
    result = tune(frame)
    assert result['input_rows'] == 4
    assert result['individual_calibration']['rp']['excluded_nonfinite_scores'] == 1
    assert result['individual_calibration']['rp']['rows'] == 3
    assert result['calibration']['rows'] == 4
    assert tune(frame.assign(reference_count=0))['calibration']['accepted'] == 0


def test_recall_floor_is_strict_and_score_ties_cannot_be_split():
    result = select_threshold(
        [1, 1, 1, 1, 0, 1, 0, 0],
        [.99, .98, .97, .96, .95, .9, .9, .1],
        min_recall=.8,
    )
    # The seemingly perfect first four positives give exactly .8 recall and
    # are ineligible. Including the fifth also includes its tied negative.
    assert result['recall'] == 1
    assert result['precision'] == pytest.approx(5/7)
    assert result['threshold'] == pytest.approx(.5)


def test_precision_ties_prefer_the_higher_cutoff():
    result = select_threshold([1, 0, 1, 0], [.9, .8, .7, .6], min_recall=0)
    assert result['threshold'] == pytest.approx(.85)
    with pytest.raises(ValueError, match='No threshold achieves recall'):
        select_threshold([0, 1], [.2, 0.0])
