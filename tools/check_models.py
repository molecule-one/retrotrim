"""Verify model file sizes and checksums against models/manifest.json."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--models", nargs="+", choices=["rp", "rgp", "rrs"])
    args = parser.parse_args()
    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    count, total = 0, 0
    for name, model in manifest.items():
        if args.models and name not in args.models:
            continue
        for relative, expected in model["files"].items():
            path = args.models_dir / name / relative
            if not path.is_file():
                raise SystemExit(f"Missing artifact: {path}")
            if (
                path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]
            ):
                raise SystemExit(f"Artifact checksum mismatch: {path}")
            count += 1
            total += path.stat().st_size
    print(f"Validated {count} files ({total / 2**20:.1f} MiB).")


if __name__ == "__main__":
    main()
