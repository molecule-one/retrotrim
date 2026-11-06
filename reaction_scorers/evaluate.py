"""Reproduce ROC-AUC and the paper's PR-AUC (average precision) from scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_frame(frame):
    required = {"class", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if not frame["class"].isin([0, 1]).all():
        raise ValueError("Labels must be binary, with class=1 the positive class.")
    scores = pd.to_numeric(frame["score"], errors="raise")
    valid = frame.loc[np.isfinite(scores)].copy()
    valid["score"] = scores.loc[valid.index]

    def metrics(rows):
        if rows["class"].nunique() != 2:
            raise ValueError("ROC-AUC requires both positive and negative labels.")
        return {
            "rows": len(rows),
            "positives": int((rows["class"] == 1).sum()),
            "negatives": int((rows["class"] == 0).sum()),
            "roc_auc": float(roc_auc_score(rows["class"], rows["score"])),
            "pr_auc_ap": float(average_precision_score(rows["class"], rows["score"])),
        }

    result = {
        "total_rows": len(frame),
        "nonfinite_scores": len(frame) - len(valid),
        "pr_auc_definition": "sklearn.metrics.average_precision_score",
        "full": metrics(valid),
        "subsets": {},
    }
    if "robustness_test_name" in valid:
        positives = valid.loc[valid["class"] == 1]
        negatives = valid.loc[valid["class"] == 0]
        for name, group in negatives.groupby("robustness_test_name", sort=True):
            result["subsets"][str(name)] = metrics(pd.concat([positives, group]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Scored reaction CSV.")
    parser.add_argument("--json", type=Path, help="Write metrics JSON.")
    parser.add_argument("--score-column", default="score", help="Name of the score column.")
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    if args.score_column not in frame:
        parser.error(f"Missing score column: {args.score_column}")
    frame["score"] = frame[args.score_column]
    result = evaluate_frame(frame)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
