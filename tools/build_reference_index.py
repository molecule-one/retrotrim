"""Build a SQLite reference-count index from the saved USPTO reaction table."""

import argparse
import csv
import io
import json
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from reaction_scorers.reference_sqlite import build_index


@contextmanager
def reaction_table(path):
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.split("/")[-1] == "reactions.tsv"
            ]
            if len(names) != 1:
                raise ValueError("Expected one reactions.tsv in the archive")
            with archive.open(names[0]) as raw, io.TextIOWrapper(raw) as text:
                yield text
    else:
        with path.open() as text:
            yield text


def build(data, output, workers=4, limit=None, expected_rows=969320, exclusions=None):
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    expected = min(expected_rows, limit) if limit is not None else expected_rows
    if exclusions is None:
        default = Path(__file__).resolve().parents[1] / "configs/rrs_excluded_rows.json"
        if default.exists():
            exclusions = default
    excluded_rows = []
    if exclusions is not None:
        definition = json.loads(Path(exclusions).read_text())
        if definition["source_rows"] != expected_rows:
            raise ValueError("Exclusions refer to a different source corpus size")
        excluded_rows = [i for i in definition["excluded_rows"] if i < expected]
    with reaction_table(Path(data)) as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if not {"substrates", "product"}.issubset(rows.fieldnames or []):
            raise ValueError("Missing substrates/product columns")
        return build_index(
            islice(rows, limit) if limit is not None else rows,
            output,
            expected_rows=expected,
            excluded_rows=excluded_rows,
            workers=workers,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-rows", type=int, default=969320)
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="JSON file listing excluded source row positions",
    )
    args = parser.parse_args()
    report = build(
        args.data,
        args.output,
        args.workers,
        args.limit,
        args.expected_rows,
        args.exclusions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
