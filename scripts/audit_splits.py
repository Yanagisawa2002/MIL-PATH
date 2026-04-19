"""Audit split artifacts for overlap, leakage, and label balance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    audit_path = processed_dir / "split_audit.json"
    pair_tables_path = processed_dir / "pair_tables.pt"
    if audit_path.exists():
        print(audit_path.read_text(encoding="utf-8"))
    if pair_tables_path.exists():
        tables = torch.load(pair_tables_path, weights_only=False)
        unresolved = {
            split: int(payload["num_unresolved_pairs"])
            for split, payload in tables.items()
        }
        print({"unresolved_pair_indices": unresolved})


if __name__ == "__main__":
    main()
