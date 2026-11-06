"""Reaction filtering and precision–recall threshold calibration."""

import argparse
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class DecisionRule:
    # Defaults fitted independently on the published 500 reactions: maximize
    # precision subject to recall > 0.8; highest cutoff breaks precision ties.
    # Search requires both enabled model gates and, when enabled, RRS count >= 1.
    rp_threshold: float = 4.66879215893153e-08
    rgp_threshold: float = 0.8639821410179138
    min_reference_count: int = 1

    def __post_init__(self):
        if not 0 <= self.rp_threshold <= 1 or not 0 <= self.rgp_threshold <= 1:
            raise ValueError("RP and RGP thresholds must lie in [0, 1].")
        if self.min_reference_count < 0:
            raise ValueError("The reference count threshold must be nonnegative.")

    def accepts(self, scores, scorers=("rp", "rgp", "rrs")):
        enabled = set(scorers)
        if not enabled <= {"rp", "rgp", "rrs"}:
            raise ValueError("Unknown scorer.")
        if "rrs" in enabled:
            count = scores.get("reference_count")
            if count is None or not np.isfinite(count) or count < self.min_reference_count:
                return False
        decisions = []
        for name, threshold in (("rp", self.rp_threshold), ("rgp", self.rgp_threshold)):
            if name in enabled:
                value = scores.get(name + "_score")
                decisions.append(value is not None and np.isfinite(value) and value > threshold)
        return all(decisions) if decisions else True

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(**data.get("thresholds", data))


def evaluate_decisions(frame, rule, scorers=("rp", "rgp", "rrs")):
    predictions = [rule.accepts(row, scorers) for row in frame.to_dict("records")]
    return {
        "rows": len(frame),
        "accepted": int(sum(predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["class"], predictions)),
        "precision": float(precision_score(frame["class"], predictions, zero_division=0)),
        "recall": float(recall_score(frame["class"], predictions, zero_division=0)),
    }


def select_threshold(labels, scores, min_recall=0.8):
    """Maximize standalone precision with recall strictly above the floor.

    Consider every attainable score cutoff, retaining tied scores together.
    Prefer the highest cutoff on precision ties. Place the strict `score > t`
    threshold halfway between the accepted and rejected score groups.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not 0 <= min_recall < 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Supply binary labels and a recall floor in [0, 1).")
    finite = np.isfinite(scores)
    labels, values = labels[finite], scores[finite]
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires finite scores from both classes.")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Model scores must lie in [0, 1].")
    order = np.argsort(-values, kind='stable')
    values, labels = values[order], labels[order]
    true_positives = np.cumsum(labels)
    positives = int(labels.sum())
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
    best = None
    for end in ends:
        tp, accepted = int(true_positives[end]), int(end+1)
        if tp / positives <= min_recall or values[end] <= 0:
            continue
        lower = float(values[end+1]) if end+1 < len(values) else 0.0
        upper = float(values[end])
        threshold = lower + (upper-lower)/2
        if threshold >= upper:
            threshold = float(np.nextafter(upper, -np.inf))
        rank = (Fraction(tp, accepted), threshold)
        if best is None or rank > best[0]:
            best = rank, {
                'threshold': threshold, 'rows': len(values),
                'excluded_nonfinite_scores': int((~finite).sum()),
                'accepted': accepted, 'true_positives': tp,
                'false_positives': accepted-tp,
                'precision': tp/accepted, 'recall': tp/positives,
            }
    if best is None:
        raise ValueError("No threshold achieves recall strictly above the requested floor.")
    return best[1]


def tune(frame, min_recall=0.8):
    """Calibrate model thresholds and report individual and combined performance."""
    required = {"class", "rp_score", "rgp_score", "reference_count"}
    if not required <= set(frame) or not frame["class"].isin([0, 1]).all():
        raise ValueError("Supply binary class labels and all three scorer outputs.")
    individual = {
        kind: select_threshold(frame['class'], frame[kind+'_score'], min_recall)
        for kind in ['rp', 'rgp']
    }
    rule = DecisionRule(individual['rp']['threshold'], individual['rgp']['threshold'], 1)
    return {
        "combination": "reference_count >= 1 AND RP > threshold AND RGP > threshold",
        "thresholds": asdict(rule),
        "minimum_individual_recall_strict": min_recall,
        "objective": "For each model independently, maximize precision subject to recall strictly above the floor. Highest cutoff breaks precision ties; tied scores are retained together.",
        "threshold_convention": "score > threshold; midpoint between adjacent accepted/rejected score groups",
        "input_rows": len(frame),
        "individual_calibration": individual,
        "combined_without_reference_gate": evaluate_decisions(frame, rule, ('rp', 'rgp')),
        "calibration": evaluate_decisions(frame, rule),
        "scope": "Each model is calibrated separately. Combining the model thresholds and reference gate can yield lower recall.",
        "note": "Calibration statistics are measured on the calibration data.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()
    report = tune(pd.read_csv(args.input), args.min_recall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
