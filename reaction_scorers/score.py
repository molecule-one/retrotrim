"""Score atom-mapped reactions from a CSV file."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .chemistry.smiles import complete_mappings
from .loading import load_rgp, load_rp


def get_reaction(substrates, product):
    """Complete partial maps and select the shortest product component."""
    sub, prod = complete_mappings(
        Chem.MolFromSmiles(substrates), Chem.MolFromSmiles(product)
    )
    return {
        "substrates": Chem.MolToSmiles(sub),
        "product": sorted(Chem.MolToSmiles(prod).split("."), key=len)[0],
    }


def score_frame(frame, scorer):
    result = frame.copy()
    scores = []
    for index, row in result.iterrows():
        try:
            score = scorer.score([get_reaction(row["substrates"], row["product"])])[0]
        except Exception as exc:
            print(f"Row {index}: {type(exc).__name__}: {exc}", flush=True)
            score = np.nan
        scores.append(score)
    result["score"] = scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rp", "rgp"], required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    scorer = {"rp": load_rp, "rgp": load_rgp}[args.model](
        args.models_dir / args.model, args.device
    )
    result = score_frame(pd.read_csv(args.input), scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} rows; {result.score.isna().sum()} missing scores.")


if __name__ == "__main__":
    main()
