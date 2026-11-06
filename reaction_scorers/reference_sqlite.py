"""Build and query a SQLite reference-count index."""

import json
import hashlib
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from itertools import islice
from pathlib import Path

from rdkit import __version__ as rdkit_version

from .reference import reference_score
from .reference_keys import reaction_keys

SCHEMA_VERSION = "sqlite-reference-v2"


def key_hash(key):
    return hashlib.sha256(key.encode("utf-8")).digest()


def _key_batch(rows):
    return [
        reaction_keys(row["substrates"], row["product"]) if row is not None else None
        for row in rows
    ]


def _prepared_keys(rows, excluded_rows, workers, key_function):
    selected = (None if i in excluded_rows else row for i, row in enumerate(rows))
    if workers == 1:
        for row in selected:
            yield (
                key_function(row["substrates"], row["product"])
                if row is not None
                else None
            )
        return
    if key_function is not reaction_keys:
        raise ValueError("Custom key functions require workers=1")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            chunks = []
            # Process rows in bounded, ordered chunks to limit memory use.
            for _ in range(workers * 2):
                chunk = list(islice(selected, 128))
                if not chunk:
                    break
                chunks.append(chunk)
            if not chunks:
                break
            for keys in pool.map(_key_batch, chunks):
                yield from keys


class PythonReferenceIndex(AbstractContextManager):
    def __init__(self, path):
        self.connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True
        )
        version = self.connection.execute(
            "SELECT value FROM metadata WHERE name='schema'"
        ).fetchone()
        if version != (SCHEMA_VERSION,):
            self.connection.close()
            raise ValueError("Unsupported reference index schema")

    def count_keys(self, keys):
        fine = {
            row[0]
            for row in self.connection.execute(
                "SELECT row_id FROM members JOIN buckets ON buckets.id=members.bucket_id "
                "WHERE tier='fine' AND key=?",
                (key_hash(keys["fine"]),),
            )
        }
        coarse = {
            row[0]
            for row in self.connection.execute(
                "SELECT row_id FROM members JOIN buckets ON buckets.id=members.bucket_id "
                "WHERE tier='coarse' AND key=?",
                (key_hash(keys["coarse"]),),
            )
        }
        exact = self.connection.execute(
            "SELECT row_id FROM exact WHERE key=?", (key_hash(keys["exact"]),)
        ).fetchone()
        union = fine | coarse
        if exact is not None:
            union.add(exact[0])
        return {
            "reference_count": len(union),
            "has_exact_match": exact is not None,
            "details": {
                "rtype_reference_count": len(fine),
                "minimal_reference_count": len(coarse),
            },
        }

    def count(self, substrates, product):
        return self.count_keys(reaction_keys(substrates, product))

    def score(self, reactions):
        return [
            reference_score(self.count(**reaction)["reference_count"])
            for reaction in reactions
        ]

    def close(self):
        self.connection.close()

    def __exit__(self, *args):
        self.close()


def build_index(
    rows,
    output,
    expected_rows=None,
    key_function=reaction_keys,
    excluded_rows=(),
    workers=1,
):
    """Stream source rows to disk, preserving duplicates and last exact matches.

    A failed build removes its temporary database. Existing outputs are never
    replaced. A single transaction avoids leaving a partially usable index.
    """
    excluded_rows = set(excluded_rows)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if any(type(i) is not int or i < 0 for i in excluded_rows):
        raise ValueError("Excluded source rows must be nonnegative integer positions")
    output = Path(output)
    temporary = output.with_name(output.name + ".building")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects an existing interrupted build.
    with temporary.open("xb"):
        pass
    connection = None
    try:
        if output.exists():
            raise FileExistsError(output)
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE buckets(id INTEGER PRIMARY KEY, tier TEXT NOT NULL, key BLOB NOT NULL,
                                 UNIQUE(tier,key));
            CREATE TABLE members(bucket_id INTEGER NOT NULL, row_id INTEGER NOT NULL);
            CREATE TABLE exact(key BLOB PRIMARY KEY, row_id INTEGER NOT NULL) WITHOUT ROWID;
        """
        )
        source_rows = indexed_rows = 0
        for keys in _prepared_keys(rows, excluded_rows, workers, key_function):
            source_rows += 1
            if keys is None:
                continue
            # Abort on parsing errors to keep the reference corpus complete.
            for tier in ("fine", "coarse"):
                if tier == "fine" and not keys[tier]:
                    continue
                digest = key_hash(keys[tier])
                connection.execute(
                    "INSERT OR IGNORE INTO buckets(tier,key) VALUES(?,?)",
                    (tier, digest),
                )
                bucket_id = connection.execute(
                    "SELECT id FROM buckets WHERE tier=? AND key=?", (tier, digest)
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO members VALUES(?,?)", (bucket_id, indexed_rows)
                )
            connection.execute(
                "INSERT OR REPLACE INTO exact VALUES(?,?)",
                (key_hash(keys["exact"]), indexed_rows),
            )
            indexed_rows += 1
            if source_rows % 65536 == 0:
                print(
                    f"Processed {source_rows} source rows; indexed {indexed_rows}",
                    flush=True,
                )
        if expected_rows is not None and source_rows != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} source rows, received {source_rows}"
            )
        if any(i >= source_rows for i in excluded_rows):
            raise ValueError("An excluded source row is outside the input corpus")
        connection.execute("CREATE INDEX membership_lookup ON members(bucket_id)")
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            [
                ("schema", SCHEMA_VERSION),
                ("source_rows", str(source_rows)),
                ("indexed_rows", str(indexed_rows)),
                ("rdkit_version", rdkit_version),
                ("excluded_source_rows", json.dumps(sorted(excluded_rows))),
            ],
        )
        connection.commit()
        connection.close()
        connection = None
        # Hard-link publication is atomic and refuses an output created concurrently.
        output.hardlink_to(temporary)
        temporary.unlink()
        return {
            "source_rows": source_rows,
            "indexed_rows": indexed_rows,
            "skipped_rows": source_rows - indexed_rows,
            "schema": SCHEMA_VERSION,
        }
    except BaseException:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise


def main():
    import argparse

    import numpy as np
    import pandas as pd
    from rdkit import Chem

    from .chemistry.smiles import complete_mappings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    results = []
    with PythonReferenceIndex(args.index) as index:
        for number, row in frame.iterrows():
            try:
                sub, prod = complete_mappings(
                    Chem.MolFromSmiles(row.substrates),
                    Chem.MolFromSmiles(row["product"]),
                )
                # RRS uses the whole product, including salts and other components.
                result = index.count(Chem.MolToSmiles(sub), Chem.MolToSmiles(prod))
                results.append(
                    {
                        "reference_count": result["reference_count"],
                        "has_exact_match": result["has_exact_match"],
                        **result["details"],
                        "score": reference_score(result["reference_count"]),
                    }
                )
            except Exception as exc:
                print(f"Row {number}: {type(exc).__name__}: {exc}", flush=True)
                results.append(
                    {
                        "reference_count": np.nan,
                        "has_exact_match": np.nan,
                        "rtype_reference_count": np.nan,
                        "minimal_reference_count": np.nan,
                        "score": np.nan,
                    }
                )
    for column, values in pd.DataFrame(results, index=frame.index).items():
        frame[column] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Scored {len(frame)} rows; {frame.score.isna().sum()} missing scores.")


if __name__ == "__main__":
    main()
