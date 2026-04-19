"""Analyze Stage-4 retrieval coverage and candidate-bag statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import load_artifact_bundle, split_pair_id
from src.training.pipeline import build_candidate_store
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/full_fast_random.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--missing-only", action="store_true")
    return parser.parse_args()


def _quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(float(v) for v in values)
    return {
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p90": float(sorted_values[min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))]),
        "p95": float(sorted_values[min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))]),
        "max": float(sorted_values[-1]),
    }


def _top_items(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": int(count)}
        for key, count in counter.most_common(limit)
    ]


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 4 Retrieval Summary",
        "",
        f"- Split: `{summary['split']}`",
        f"- Positive-no-path pairs: `{summary['num_pairs']}`",
        f"- Non-empty bags: `{summary['num_nonempty']}`",
        f"- Coverage: `{summary['coverage']:.4f}`",
        f"- Singleton non-empty bags: `{summary['bag_size_histogram'].get('1', 0)}`",
        "",
        "## Bag Size",
        "",
        f"- All pairs mean/median/p90: `{summary['bag_size_all']['mean']:.2f}` / `{summary['bag_size_all']['median']:.2f}` / `{summary['bag_size_all']['p90']:.2f}`",
        f"- Non-empty mean/median/p90: `{summary['bag_size_nonempty']['mean']:.2f}` / `{summary['bag_size_nonempty']['median']:.2f}` / `{summary['bag_size_nonempty']['p90']:.2f}`",
        "",
        "## Top Schemas",
        "",
    ]
    for item in summary["top_schemas"][:10]:
        lines.append(f"- `{item['key']}`: `{item['count']}`")
    lines.extend(["", "## Top Strategies", ""])
    for item in summary["top_strategies"]:
        lines.append(f"- `{item['key']}`: `{item['count']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    bundle = load_artifact_bundle(config["paths"]["processed_dir"])
    candidate_store = build_candidate_store(config, bundle, profile="stage4")
    pair_table = bundle.pair_tables[args.split]

    indices = [
        idx
        for idx in range(int(pair_table["labels"].numel()))
        if int(pair_table["labels"][idx].item()) == 1
        and not bool(pair_table["has_gold_rationale"][idx].item())
    ]
    if args.missing_only:
        indices = [
            idx
            for idx in indices
            if not candidate_store._cache_path(pair_table["pair_ids"][idx]).exists()
        ]
    if args.limit is not None:
        indices = indices[: args.limit]

    bag_sizes_all: list[int] = []
    bag_sizes_nonempty: list[int] = []
    bag_size_counter: Counter[str] = Counter()
    top_schema_counter: Counter[str] = Counter()
    schema_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    hop_counter: Counter[str] = Counter()
    nonempty = 0

    for offset, pair_idx in enumerate(indices, start=1):
        pair_id = pair_table["pair_ids"][pair_idx]
        drug_id, disease_id = split_pair_id(pair_id)
        bag = candidate_store.get_pair_paths(
            pair_id=pair_id,
            drug_id=drug_id,
            disease_id=disease_id,
            include_gold=False,
        )
        bag_size = len(bag)
        bag_sizes_all.append(bag_size)
        bag_size_counter[str(bag_size)] += 1
        if bag:
            nonempty += 1
            bag_sizes_nonempty.append(bag_size)
            top_schema_counter[bag[0]["schema_id"]] += 1
            for record in bag:
                schema_counter[record["schema_id"]] += 1
                strategy_counter[str(record.get("retrieval_strategy", record.get("path_source", "unknown")))] += 1
                hop_counter[str(int(record["hop_count"]))] += 1
        if offset % args.progress_every == 0:
            print({"processed": offset, "nonempty": nonempty})

    summary = {
        "config": args.config,
        "split": args.split,
        "num_pairs": len(indices),
        "num_nonempty": nonempty,
        "coverage": float(nonempty / len(indices)) if indices else 0.0,
        "bag_size_all": _quantiles(bag_sizes_all),
        "bag_size_nonempty": _quantiles(bag_sizes_nonempty),
        "bag_size_histogram": {key: int(value) for key, value in sorted(bag_size_counter.items(), key=lambda item: int(item[0]))},
        "top_schemas": _top_items(schema_counter),
        "top_top1_schemas": _top_items(top_schema_counter),
        "top_strategies": _top_items(strategy_counter),
        "hop_histogram": _top_items(hop_counter),
    }

    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"])
    json_path = output_dir / f"stage4_retrieval_summary_{args.split}.json"
    md_path = output_dir / f"stage4_retrieval_summary_{args.split}.md"
    save_json(summary, json_path)
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
