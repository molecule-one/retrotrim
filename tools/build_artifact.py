"""Build a reproducible archive of Git-tracked files and local model assets.

Use --models-dir and --training-dir to include weights and training data.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def build(root, output, models_dir=None, training_dir=None):
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True
    ).split("\0")
    files = {name: root / name for name in names if name}
    if models_dir:
        manifest = json.loads((root / "models/manifest.json").read_text())
        for kind, model in manifest.items():
            for relative, expected in model["files"].items():
                path = models_dir / kind / relative
                if path.stat().st_size != expected["bytes"]:
                    raise ValueError(f"Incorrect size: {kind}/{relative}")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(1024**2):
                        digest.update(block)
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Incorrect checksum: {kind}/{relative}")
                files[f"models/{kind}/{relative}"] = path
    if training_dir:
        manifest = json.loads((root / "provenance/training_artifacts.json").read_text())
        for relative, expected in manifest["datasets_and_splits"].items():
            path = training_dir / relative
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024**2):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Incorrect checksum: {relative}")
            files[f"training/{relative}"] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=4
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name, path in sorted(files.items()):
            if not path.is_file():
                raise ValueError(f"Missing release file: {name}")
            entry = tarfile.TarInfo("uspto-reaction-scorers/" + name)
            entry.size = path.stat().st_size
            entry.mode = 0o644
            # Fixed ownership and timestamps make archive builds reproducible.
            with path.open("rb") as stream:
                archive.addfile(entry, stream)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return {
        "artifact": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "files": len(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.root, args.output, args.models_dir, args.training_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
