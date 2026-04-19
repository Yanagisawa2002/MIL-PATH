"""Build stage-0 artifacts from mapped PrimeKG + DrugMechDB files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_dataset import build_artifacts
from src.utils.config import load_experiment_config, prepare_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/tiny_sanity.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    summary = build_artifacts(config)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written under: {Path(config['paths']['processed_dir']).resolve()}")


if __name__ == "__main__":
    main()
