"""Late-fuse strong pairwise baseline scores with Mech branch scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import Evaluator
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/full_fast_cold_drug.yaml")
    parser.add_argument("--pairwise-dir", type=str, required=True)
    parser.add_argument("--mech-dir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="pairwise_mech_fusion")
    parser.add_argument("--pairwise-column", type=str, default="score")
    parser.add_argument("--mech-column", type=str, default="score")
    parser.add_argument("--objective", choices=["auprc", "auroc"], default="auprc")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def _read_prediction_csv(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            pair_id = row["pair_id"]
            rows[pair_id] = row
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def _prob_to_logit(values: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        std = 1.0
    return (values - mean) / std, mean, std


def _load_split_rows(
    pairwise_dir: Path,
    mech_dir: Path,
    split: str,
    pairwise_column: str,
    mech_column: str,
    eps: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    pairwise_rows = _read_prediction_csv(pairwise_dir / f"per_pair_predictions_{split}.csv")
    mech_rows = _read_prediction_csv(mech_dir / f"per_pair_predictions_{split}.csv")
    shared_pair_ids = sorted(set(pairwise_rows) & set(mech_rows))
    if not shared_pair_ids:
        raise ValueError(f"no shared pairs found for split={split}")
    if len(shared_pair_ids) != len(pairwise_rows) or len(shared_pair_ids) != len(mech_rows):
        missing_from_pairwise = sorted(set(mech_rows) - set(pairwise_rows))
        missing_from_mech = sorted(set(pairwise_rows) - set(mech_rows))
        raise ValueError(
            "pair prediction mismatch for split="
            f"{split}: pairwise_only={len(missing_from_mech)} mech_only={len(missing_from_pairwise)}"
        )

    labels = []
    pairwise_scores = []
    mech_scores = []
    joined_rows: list[dict[str, Any]] = []
    for pair_id in shared_pair_ids:
        pairwise_row = pairwise_rows[pair_id]
        mech_row = mech_rows[pair_id]
        pairwise_label = int(pairwise_row["label"])
        mech_label = int(mech_row["label"])
        if pairwise_label != mech_label:
            raise ValueError(f"label mismatch for pair_id={pair_id}: {pairwise_label} vs {mech_label}")
        pairwise_score = float(pairwise_row[pairwise_column])
        mech_score = float(mech_row[mech_column])
        labels.append(pairwise_label)
        pairwise_scores.append(pairwise_score)
        mech_scores.append(mech_score)
        joined_rows.append(
            {
                "pair_id": pair_id,
                "label": pairwise_label,
                "pairwise_raw_score": pairwise_score,
                "mech_raw_score": mech_score,
            }
        )

    pairwise_scores_array = _prob_to_logit(np.asarray(pairwise_scores, dtype=np.float64), eps=eps)
    mech_scores_array = np.asarray(mech_scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    return joined_rows, labels_array, pairwise_scores_array, mech_scores_array


def _attach_fused_scores(
    base_rows: list[dict[str, Any]],
    fused_scores: np.ndarray,
    pairwise_scores: np.ndarray,
    mech_scores: np.ndarray,
    weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, fused_score, pairwise_score, mech_score in zip(
        base_rows,
        fused_scores.tolist(),
        pairwise_scores.tolist(),
        mech_scores.tolist(),
        strict=True,
    ):
        rows.append(
            {
                **row,
                "score": float(fused_score),
                "pairwise_score": float(pairwise_score),
                "mech_score": float(mech_score),
                "w_mech": float(weight),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"] / args.output_name)
    evaluator = Evaluator(output_dir=output_dir, ks=config["evaluation"]["ks"])
    pairwise_dir = Path(args.pairwise_dir)
    mech_dir = Path(args.mech_dir)

    valid_base_rows, valid_labels, valid_pairwise_logits, valid_mech_scores = _load_split_rows(
        pairwise_dir=pairwise_dir,
        mech_dir=mech_dir,
        split="valid",
        pairwise_column=args.pairwise_column,
        mech_column=args.mech_column,
        eps=args.eps,
    )
    test_base_rows, test_labels, test_pairwise_logits, test_mech_scores = _load_split_rows(
        pairwise_dir=pairwise_dir,
        mech_dir=mech_dir,
        split="test",
        pairwise_column=args.pairwise_column,
        mech_column=args.mech_column,
        eps=args.eps,
    )

    valid_pairwise_norm, pairwise_mean, pairwise_std = _normalize(valid_pairwise_logits)
    valid_mech_norm, mech_mean, mech_std = _normalize(valid_mech_scores)
    test_pairwise_norm = (test_pairwise_logits - pairwise_mean) / pairwise_std
    test_mech_norm = (test_mech_scores - mech_mean) / mech_std

    objective_key = args.objective
    best_weight = None
    best_valid_rows = None
    best_test_rows = None
    best_valid_metrics = None
    best_test_metrics = None
    best_objective = float("-inf")
    search_rows: list[dict[str, Any]] = []
    for weight in np.linspace(0.0, 1.0, num=args.grid_size):
        valid_fused = (1.0 - weight) * valid_pairwise_norm + weight * valid_mech_norm
        test_fused = (1.0 - weight) * test_pairwise_norm + weight * test_mech_norm
        valid_rows = _attach_fused_scores(
            base_rows=valid_base_rows,
            fused_scores=valid_fused,
            pairwise_scores=valid_pairwise_norm,
            mech_scores=valid_mech_norm,
            weight=float(weight),
        )
        test_rows = _attach_fused_scores(
            base_rows=test_base_rows,
            fused_scores=test_fused,
            pairwise_scores=test_pairwise_norm,
            mech_scores=test_mech_norm,
            weight=float(weight),
        )
        valid_metrics = evaluator.evaluate_pairs(valid_rows)
        test_metrics = evaluator.evaluate_pairs(test_rows)
        search_row = {
            "w_mech": float(weight),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        search_rows.append(search_row)
        if valid_metrics[objective_key] > best_objective:
            best_objective = valid_metrics[objective_key]
            best_weight = float(weight)
            best_valid_rows = valid_rows
            best_test_rows = test_rows
            best_valid_metrics = valid_metrics
            best_test_metrics = test_metrics

    assert best_weight is not None and best_valid_rows is not None and best_test_rows is not None
    save_json(
        {
            "pairwise_dir": str(pairwise_dir),
            "mech_dir": str(mech_dir),
            "pairwise_column": args.pairwise_column,
            "mech_column": args.mech_column,
            "objective": objective_key,
            "grid_size": args.grid_size,
            "selected_w_mech": best_weight,
            "valid": best_valid_metrics,
            "test": best_test_metrics,
            "normalization": {
                "pairwise_logit_valid_mean": pairwise_mean,
                "pairwise_logit_valid_std": pairwise_std,
                "mech_valid_mean": mech_mean,
                "mech_valid_std": mech_std,
            },
        },
        output_dir / "metrics.json",
    )
    write_csv(search_rows, output_dir / "fusion_search.csv")
    write_csv(best_valid_rows, output_dir / "per_pair_predictions_valid.csv")
    write_csv(best_test_rows, output_dir / "per_pair_predictions_test.csv")


if __name__ == "__main__":
    main()
