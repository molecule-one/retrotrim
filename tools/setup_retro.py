"""Install Syntheseus and fetch its public pretrained RootAligned checkpoint."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use an already prepared environment.")
    parser.add_argument("--download-generator", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/syntheseus"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root) + "[models,reference,retro]"], check=True)
    if args.download_generator:
        sys.path.insert(0, str(root))
        from reaction_scorers.retro import load_public_generator

        model = load_public_generator("cpu", args.cache_dir)
        print(f"Public RootAligned checkpoint ready: {model.model_dir}")


if __name__ == "__main__":
    main()
