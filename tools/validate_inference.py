"""Compare re-scored reactions with saved USPTO experiment outputs."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reaction_scorers.evaluate import evaluate_frame
from reaction_scorers.loading import load_rgp, load_rp
from reaction_scorers.score import get_reaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, required=True, help="User-supplied reactions with reference scores.")
    parser.add_argument("--model", choices=["rp", "rgp", "both"], default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute score tolerance for cross-platform float32 inference.",
    )
    parser.add_argument(
        "--reuse-rescored",
        action="store_true",
        help="Validate previously saved inference outputs without re-running models.",
    )
    parser.add_argument(
        "--metric-atol", type=float, default=1e-4,
        help="Absolute metric tolerance; float32 ties can change rank metrics even when scores match.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    logging.disable(logging.WARNING)
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    reports = {}
    for name in ["rp", "rgp"] if args.model == "both" else [args.model]:
        start = time.monotonic()
        scorer = (
            None
            if args.reuse_rescored
            else {"rp": load_rp, "rgp": load_rgp}[name](
                args.root / "models" / name, args.device
            )
        )
        frame = pd.read_csv(args.input)
        if args.limit:
            frame = frame.iloc[: args.limit].copy()
        scores = []
        errors = []
        if args.reuse_rescored:
            cached = pd.read_csv(args.root / "validation" / f"{name}_rescored.csv")
            if len(cached) != len(frame) or not cached.reaction_id.equals(
                frame.reaction_id
            ):
                raise ValueError(
                    "Previously rescored rows do not match the evaluation population."
                )
            scores = cached.score.tolist()
        for index, row in [] if args.reuse_rescored else frame.iterrows():
            try:
                value = scorer.score([get_reaction(row.substrates, row["product"])])[0]
            except Exception as exc:
                value = np.nan
                errors.append(
                    {
                        "row": int(index),
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
            scores.append(value)
            if len(scores) % 100 == 0:
                print(
                    f"{name}: {len(scores)}/{len(frame)} rows, "
                    f"{time.monotonic() - start:.1f}s",
                    flush=True,
                )
        expected, actual = frame["score"].to_numpy(), np.asarray(scores)
        finite = np.isfinite(expected) & np.isfinite(actual)
        mask_equal = bool(np.array_equal(np.isfinite(expected), np.isfinite(actual)))
        max_error = (
            float(np.max(np.abs(expected[finite] - actual[finite])))
            if finite.any()
            else None
        )
        report = {
            "rows": len(frame),
            "device": args.device,
            "torch_version": torch.__version__,
            "absolute_score_tolerance": args.atol,
            "absolute_metric_tolerance": args.metric_atol,
            "reused_saved_inference": args.reuse_rescored,
            "nonfinite_mask_matches": mask_equal,
            "maximum_absolute_score_error": max_error,
            "scores_match": bool(
                mask_equal
                and np.allclose(
                    expected[finite], actual[finite], rtol=0, atol=args.atol
                )
            ),
            "errors": errors,
            "elapsed_seconds": time.monotonic() - start,
        }
        frame["score"] = actual
        destination = args.root / "validation"
        destination.mkdir(exist_ok=True)
        frame.to_csv(destination / f"{name}_rescored.csv", index=False)
        if frame["class"].nunique() == 2:
            report["metrics"] = evaluate_frame(frame)
            original_metrics = evaluate_frame(
                pd.read_csv(args.input).iloc[
                    : len(frame)
                ]
            )
            report["full_metrics_match_at_six_decimals"] = all(
                round(report["metrics"]["full"][key], 6)
                == round(original_metrics["full"][key], 6)
                for key in ["roc_auc", "pr_auc_ap"]
            )
            report["full_metric_absolute_errors"] = {
                key: abs(report["metrics"]["full"][key] - original_metrics["full"][key])
                for key in ["roc_auc", "pr_auc_ap"]
            }
            report["full_metrics_match_within_tolerance"] = all(
                error <= args.metric_atol
                for error in report["full_metric_absolute_errors"].values()
            )
        (destination / f"{name}_inference.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        reports[name] = report
        print(
            json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2),
            flush=True,
        )
    if not all(
        report["scores_match"]
        and report.get("full_metrics_match_within_tolerance", True)
        for report in reports.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
