"""Check the benchmark schema, annotations, and checksum."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_published_benchmark_boundary():
    path = ROOT / "benchmark/reactions.csv"
    frame = pd.read_csv(path)
    report = json.loads((ROOT / "benchmark/selection.json").read_text())
    assert len(frame) == 500
    assert list(frame.columns) == ["substrates", "product", "category", "issue"]
    assert not frame.isna().any().any()
    assert frame["category"].value_counts().to_dict() == {
        "Safe bet": 397, "Rather not": 59, "Nonsense": 44,
    }
    assert set(frame["issue"]) == {
        "No problem", "Magic", "Selectivity", "Functional group incompatibility",
        "One pot", "Reactivity", "Reactants mismatch", "Unstable",
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["csv_sha256"]
