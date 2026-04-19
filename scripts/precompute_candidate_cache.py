"""Precompute lazy candidate path cache for a chosen split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import load_artifact_bundle, split_pair_id
from src.training.pipeline import build_candidate_store
from src.utils.config import load_experiment_config, prepare_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/full_fast_random.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "positive_no_path", "gold_only_pairs"],
    )
    parser.add_argument("--profile", type=str, default="default", choices=["default", "stage3_cached", "stage4"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    bundle = load_artifact_bundle(config["paths"]["processed_dir"])
    candidate_store = build_candidate_store(config, bundle, profile=args.profile)
    pair_table = bundle.pair_tables[args.split]

    indices = list(range(int(pair_table["labels"].numel())))
    if args.mode == "positive_no_path":
        indices = [
            idx
            for idx in indices
            if int(pair_table["labels"][idx].item()) == 1
            and not bool(pair_table["has_gold_rationale"][idx].item())
        ]
    elif args.mode == "gold_only_pairs":
        indices = [
            idx
            for idx in indices
            if bool(pair_table["has_gold_rationale"][idx].item())
        ]
    if args.limit is not None:
        indices = indices[: args.limit]

    nonempty = 0
    for offset, idx in enumerate(indices, start=1):
        pair_id = pair_table["pair_ids"][idx]
        drug_id, disease_id = split_pair_id(pair_id)
        bag = candidate_store.get_pair_paths(
            pair_id=pair_id,
            drug_id=drug_id,
            disease_id=disease_id,
            include_gold=False,
        )
        if bag:
            nonempty += 1
        if offset % 100 == 0:
            print({"processed": offset, "nonempty": nonempty})

    print({"processed": len(indices), "nonempty": nonempty})


if __name__ == "__main__":
    main()
