"""Copy model assets from a local directory or an HTTPS URL and verify checksums."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from check_models import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing rp/, rgp/, and optionally rrs/.",
    )
    group.add_argument("--base-url", help="HTTPS URL for a released models directory.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--models", nargs="+", choices=["rp", "rgp", "rrs"], default=["rp", "rgp"]
    )
    args = parser.parse_args()
    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    required = sum(
        file["bytes"]
        for name in args.models
        for file in manifest[name]["files"].values()
    )
    if shutil.disk_usage(args.models_dir).free < required:
        raise SystemExit(
            "Insufficient disk space; selected artifacts require "
            f"{required / 2**20:.1f} MiB."
        )
    print(f"Selected artifact size: {required / 2**20:.1f} MiB.")
    for name in args.models:
        for relative, expected in manifest[name]["files"].items():
            destination = args.models_dir / name / relative
            if destination.is_file() and sha256(destination) == expected["sha256"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            try:
                if args.source_dir:
                    shutil.copyfile(args.source_dir / name / relative, temporary)
                else:
                    if not args.base_url.startswith("https://"):
                        raise ValueError("--base-url must use HTTPS.")
                    with (
                        urlopen(
                            args.base_url.rstrip("/")
                            + "/"
                            + quote(name + "/" + relative, safe="/"),
                            timeout=120,
                        ) as source,
                        temporary.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
                if (
                    temporary.stat().st_size != expected["bytes"]
                    or sha256(temporary) != expected["sha256"]
                ):
                    raise ValueError(f"Checksum mismatch for {name}/{relative}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Installed {name}/{relative}")


if __name__ == "__main__":
    main()
