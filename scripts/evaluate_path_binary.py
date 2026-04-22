"""Evaluate path-level binary classification on controlled hard candidate sets.

This script builds a 1:1 positive/negative dataset from a `per_path_hard_ranking_*.csv`
file by selecting, for each pair:

- one positive path: the highest-scoring gold path
- one negative path: the highest-scoring non-gold path

It then reports binary AUROC/AUPRC and a few thresholded metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import binary_auprc, binary_auroc
from src.utils.io import ensure_dir, save_json, write_csv


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _stable_index(key: str, size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % size


def _stable_pick(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["path_id"]))
    return ordered[_stable_index(key, len(ordered))]


def _selected_row_payload(
    row: dict[str, Any],
    *,
    pair_id: str,
    label: int,
    is_gold: int,
    selection_role: str,
) -> dict[str, Any]:
    payload = {
        "pair_id": pair_id,
        "path_id": row["path_id"],
        "schema_id": row["schema_id"],
        "path_source": row["path_source"],
        "score": float(row["score"]),
        "label": label,
        "is_gold": is_gold,
        "selection_role": selection_role,
    }
    if "binary_score" in row and str(row["binary_score"]).strip() not in {"", "nan", "NaN"}:
        payload["binary_score"] = float(row["binary_score"])
    for key, value in row.items():
        if key in payload:
            continue
        if not (key.endswith("_score") or key.endswith("_prob")):
            continue
        if str(value).strip() in {"", "nan", "NaN"}:
            continue
        payload[key] = float(value)
    return payload


def _select_pair_matched_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)

    selected_rows: list[dict[str, Any]] = []
    skipped_pairs = 0
    for pair_id, pair_rows in grouped.items():
        gold_rows = [row for row in pair_rows if int(row["is_gold"]) == 1]
        negative_rows = [row for row in pair_rows if int(row["is_gold"]) == 0]
        if not gold_rows or not negative_rows:
            skipped_pairs += 1
            continue

        best_gold = max(gold_rows, key=lambda row: float(row["score"]))
        hardest_negative = max(negative_rows, key=lambda row: float(row["score"]))

        selected_rows.append(_selected_row_payload(best_gold, pair_id=pair_id, label=1, is_gold=1, selection_role="positive_best_gold"))
        selected_rows.append(
            _selected_row_payload(
                hardest_negative,
                pair_id=pair_id,
                label=0,
                is_gold=0,
                selection_role="negative_hardest_non_gold",
            )
        )

    diagnostics = {
        "num_pairs_total": len(grouped),
        "num_pairs_used": len(selected_rows) // 2,
        "num_pairs_skipped": skipped_pairs,
        "num_positive": len(selected_rows) // 2,
        "num_negative": len(selected_rows) // 2,
    }
    return selected_rows, diagnostics


def _select_pair_matched_fixed_mixed_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)

    negative_sources = ("corrupt_internal", "cross_pair_same_schema")
    selected_rows: list[dict[str, Any]] = []
    skipped_pairs = 0
    source_usage = {source: 0 for source in negative_sources}
    fallback_usage = 0

    for pair_id, pair_rows in grouped.items():
        gold_rows = [row for row in pair_rows if int(row["is_gold"]) == 1]
        if not gold_rows:
            skipped_pairs += 1
            continue

        positive = _stable_pick(gold_rows, f"{pair_id}::gold")
        preferred_idx = _stable_index(f"{pair_id}::neg_source", len(negative_sources))
        source_order = [
            negative_sources[preferred_idx],
            negative_sources[1 - preferred_idx],
        ]

        chosen_negative = None
        chosen_source = None
        for source in source_order:
            candidates = [row for row in pair_rows if row["path_source"] == source]
            if candidates:
                chosen_negative = _stable_pick(candidates, f"{pair_id}::{source}")
                chosen_source = source
                break

        if chosen_negative is None or chosen_source is None:
            skipped_pairs += 1
            continue

        if chosen_source != source_order[0]:
            fallback_usage += 1
        source_usage[chosen_source] += 1

        selected_rows.append(_selected_row_payload(positive, pair_id=pair_id, label=1, is_gold=1, selection_role="positive_stable_gold"))
        selected_rows.append(
            _selected_row_payload(
                chosen_negative,
                pair_id=pair_id,
                label=0,
                is_gold=0,
                selection_role=f"negative_fixed_{chosen_source}",
            )
        )

    diagnostics = {
        "num_pairs_total": len(grouped),
        "num_pairs_used": len(selected_rows) // 2,
        "num_pairs_skipped": skipped_pairs,
        "num_positive": len(selected_rows) // 2,
        "num_negative": len(selected_rows) // 2,
        "negative_source_usage": source_usage,
        "negative_source_fallbacks": fallback_usage,
    }
    return selected_rows, diagnostics


def _select_pair_matched_fixed_hard_1to4_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)

    negative_sources = ("corrupt_internal", "cross_pair_same_schema", "cross_pair_same_hop")
    selected_rows: list[dict[str, Any]] = []
    skipped_pairs = 0
    source_usage = {source: 0 for source in negative_sources}
    fallback_usage = 0

    for pair_id, pair_rows in grouped.items():
        gold_rows = [row for row in pair_rows if int(row["is_gold"]) == 1]
        if not gold_rows:
            skipped_pairs += 1
            continue

        positive = _stable_pick(gold_rows, f"{pair_id}::gold")

        source_to_rows = {
            source: sorted(
                [row for row in pair_rows if row["path_source"] == source],
                key=lambda row: str(row["path_id"]),
            )
            for source in negative_sources
        }
        total_hard = sum(len(rows_for_source) for rows_for_source in source_to_rows.values())
        if total_hard < 4:
            skipped_pairs += 1
            continue

        rotation = _stable_index(f"{pair_id}::source_rotation", len(negative_sources))
        rotated_sources = [
            negative_sources[(rotation + offset) % len(negative_sources)]
            for offset in range(len(negative_sources))
        ]

        selected_negative_ids: set[str] = set()
        selected_negatives: list[dict[str, Any]] = []

        # First pass: maximize source diversity (at most one per source).
        for source in rotated_sources:
            candidates = [row for row in source_to_rows[source] if str(row["path_id"]) not in selected_negative_ids]
            if not candidates:
                continue
            chosen = _stable_pick(candidates, f"{pair_id}::{source}::primary")
            selected_negatives.append(chosen)
            selected_negative_ids.add(str(chosen["path_id"]))
            source_usage[source] += 1
            if len(selected_negatives) == 4:
                break

        # Second pass: fill remaining slots from all hard negatives, still without score-based selection.
        if len(selected_negatives) < 4:
            fallback_usage += 1
            remaining_pool: list[dict[str, Any]] = []
            for source in rotated_sources:
                remaining_pool.extend(
                    [row for row in source_to_rows[source] if str(row["path_id"]) not in selected_negative_ids]
                )
            remaining_pool = sorted(remaining_pool, key=lambda row: (str(row["path_source"]), str(row["path_id"])))
            remaining_slots = 4 - len(selected_negatives)
            for idx in range(remaining_slots):
                candidates = [row for row in remaining_pool if str(row["path_id"]) not in selected_negative_ids]
                if not candidates:
                    break
                chosen = _stable_pick(candidates, f"{pair_id}::fallback::{idx}")
                selected_negatives.append(chosen)
                selected_negative_ids.add(str(chosen["path_id"]))
                source_usage[str(chosen["path_source"])] += 1

        if len(selected_negatives) != 4:
            skipped_pairs += 1
            continue

        selected_rows.append(_selected_row_payload(positive, pair_id=pair_id, label=1, is_gold=1, selection_role="positive_stable_gold"))

        for neg_idx, negative in enumerate(selected_negatives):
            selected_rows.append(
                _selected_row_payload(
                    negative,
                    pair_id=pair_id,
                    label=0,
                    is_gold=0,
                    selection_role=f"negative_hard_fixed_{neg_idx}",
                )
            )

    diagnostics = {
        "num_pairs_total": len(grouped),
        "num_pairs_used": len(selected_rows) // 5,
        "num_pairs_skipped": skipped_pairs,
        "num_positive": len(selected_rows) // 5,
        "num_negative": (len(selected_rows) // 5) * 4,
        "negative_source_usage": source_usage,
        "negative_source_fallback_pairs": fallback_usage,
    }
    return selected_rows, diagnostics


def _compute_metrics(rows: list[dict[str, Any]], score_column: str = "score") -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row[score_column]) for row in rows], dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-scores))
    pred = (scores >= 0.0).astype(np.int64)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, len(labels))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    return {
        "auroc": binary_auroc(labels, scores),
        "auprc": binary_auprc(labels, scores),
        "accuracy_at_logit_0": accuracy,
        "precision_at_logit_0": precision,
        "recall_at_logit_0": recall,
        "f1_at_logit_0": f1,
        "mean_positive_score": float(scores[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_negative_score": float(scores[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "mean_positive_prob": float(probs[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_negative_prob": float(probs[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "confusion_at_logit_0": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to per_path_hard_ranking_*.csv")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="path_binary_eval")
    parser.add_argument("--score-column", type=str, default="score")
    parser.add_argument(
        "--selection",
        type=str,
        default="best_gold_vs_hardest_non_gold",
        choices=["best_gold_vs_hardest_non_gold", "fixed_mixed", "fixed_hard_1to4"],
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = ensure_dir(Path(args.output_dir))

    rows = _load_rows(input_path)
    if args.selection == "fixed_mixed":
        selected_rows, diagnostics = _select_pair_matched_fixed_mixed_rows(rows)
        selection_name = "pair_matched_stable_gold_vs_fixed_mixed_negatives"
    elif args.selection == "fixed_hard_1to4":
        selected_rows, diagnostics = _select_pair_matched_fixed_hard_1to4_rows(rows)
        selection_name = "pair_matched_stable_gold_vs_fixed_hard_negatives_1to4"
    else:
        selected_rows, diagnostics = _select_pair_matched_rows(rows)
        selection_name = "pair_matched_best_gold_vs_hardest_non_gold"
    metrics = _compute_metrics(selected_rows, score_column=args.score_column)
    summary = {
        "name": args.name,
        "input_csv": str(input_path),
        "score_column": args.score_column,
        "selection": selection_name,
        "diagnostics": diagnostics,
        "metrics": metrics,
    }
    save_json(summary, output_dir / f"{args.name}.json")
    write_csv(selected_rows, output_dir / f"{args.name}_rows.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
