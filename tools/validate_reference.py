"""Compare reference-index counts with the evaluation results."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from reaction_scorers.chemistry.smiles import complete_mappings
from reaction_scorers.reference_sqlite import PythonReferenceIndex


def validate(index_path, frame):
    failures = []
    with PythonReferenceIndex(index_path) as index:
        metadata = dict(index.connection.execute("SELECT name,value FROM metadata"))
        for row_number, row in frame.iterrows():
            try:
                sub, prod = complete_mappings(
                    Chem.MolFromSmiles(row.substrates),
                    Chem.MolFromSmiles(row["product"]),
                )
                result = index.count(Chem.MolToSmiles(sub), Chem.MolToSmiles(prod))
                expected = {
                    "reference_count": int(row.reference_count),
                    "rtype_reference_count": int(row.rtype_reference_count),
                    "minimal_reference_count": int(row.minimal_reference_count),
                    "has_exact_match": bool(row.has_exact_match),
                }
                actual = {
                    "reference_count": result["reference_count"],
                    "has_exact_match": result["has_exact_match"],
                    **result["details"],
                }
                if expected != actual:
                    failures.append(
                        {"row": row_number, "expected": expected, "actual": actual}
                    )
            except Exception as exc:
                failures.append({"row": row_number, "error": type(exc).__name__})
    return {
        "implementation": "rdkit-kernel-v1",
        "index_source_rows": int(metadata["source_rows"]),
        "index_indexed_rows": int(metadata["indexed_rows"]),
        "rows_checked": len(frame),
        "mismatches": len(failures),
        "mismatch_examples": failures[:10],
        "passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("models/rrs/reference.sqlite")
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("validation/rrs_port.json"))
    args = parser.parse_args()
    report = validate(args.index, pd.read_csv(args.input))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(not report["passed"])


if __name__ == "__main__":
    main()
