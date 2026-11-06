"""Check reaction-table joins and positional split assignments."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "restore_training", Path(__file__).parents[1] / "tools/restore_training.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_ids_are_split_by_position(tmp_path):
    with ZipFile(tmp_path / "data.zip", "w") as archive:
        archive.writestr(
            "reactions.tsv",
            "reaction_id\tsubstrates\tproduct\nA\tS1\tP1\nA\tS2\tP2\nB\tS3\tP3\n",
        )
    with ZipFile(tmp_path / "splits.zip", "w") as archive:
        archive.writestr(
            "split.tsv", "id\ttrain\tval\ttest\nA\t1\t0\t0\nA\t0\t1\t0\nB\t1\t0\t0\n"
        )
    counts = module.restore(
        tmp_path / "data.zip", tmp_path / "splits.zip", tmp_path / "restored"
    )
    assert counts == {"train": 2, "val": 1, "test": 0}
    assert pd.read_csv(tmp_path / "restored/val.csv").substrates.tolist() == ["S2"]
    with pytest.raises(FileExistsError):
        module.restore(
            tmp_path / "data.zip", tmp_path / "splits.zip", tmp_path / "restored"
        )
