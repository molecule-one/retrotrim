"""Restore reaction tables and split assignments to train, validation, and test CSVs."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def read_table(path, suffix):
    with ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.split("/")[-1] == suffix]
        if len(matches) != 1:
            raise ValueError(f"Expected one {suffix} in {path.name}")
        with archive.open(matches[0]) as stream:
            return pd.read_csv(stream, sep="\t")


def restore(data_path, splits_path, output):
    data = read_table(data_path, "reactions.tsv")
    split = read_table(splits_path, "split.tsv")
    if len(data) != len(split) or not data.reaction_id.equals(split["id"]):
        raise ValueError("Reaction and split IDs differ in row order")
    flags = split[["train", "val", "test"]]
    if not flags.isin([0, 1]).all().all() or not flags.sum(axis=1).eq(1).all():
        raise ValueError("Split membership must be binary and disjoint")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name in flags:
        frame = data.loc[flags[name].eq(1)]
        frame.to_csv(output / f"{name}.csv", index=False)
        counts[name] = len(frame)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(restore(args.data, args.splits, args.output))


if __name__ == "__main__":
    main()
