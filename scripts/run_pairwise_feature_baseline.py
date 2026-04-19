"""Run an explicit pair-feature baseline without Mech supervision."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.pairwise_features import PairwiseFeatureBuilder
from src.evaluation.evaluator import Evaluator
from src.training.pipeline import load_artifact_bundle
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/full_fast_cold_drug.yaml")
    parser.add_argument("--output-name", type=str, default="pairwise_feature_baseline")
    return parser.parse_args()


def _candidate_models(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "logreg_c0.1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, max_iter=4000, random_state=seed)),
                ]
            ),
        ),
        (
            "logreg_c1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=4000, random_state=seed)),
                ]
            ),
        ),
        (
            "logreg_c10",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=10.0, max_iter=4000, random_state=seed)),
                ]
            ),
        ),
        (
            "rf_d8",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "rf_d12",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "rf_none",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "extra_trees_d8",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "extra_trees_d12",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "extra_trees_none",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            ),
        ),
    ]


def _predict_rows(pair_ids: list[str], labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair_id,
            "label": int(label),
            "score": float(score),
        }
        for pair_id, label, score in zip(pair_ids, labels.tolist(), scores.tolist(), strict=True)
    ]


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    processed_dir = Path(config["paths"]["processed_dir"])
    bundle = load_artifact_bundle(processed_dir)
    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"] / args.output_name)
    evaluator = Evaluator(output_dir=output_dir, ks=config["evaluation"]["ks"])
    seed = int(config["project"].get("seed", 42))

    builder = PairwiseFeatureBuilder(graph_data=bundle.graph_data, pair_tables=bundle.pair_tables)
    x_train, y_train, train_ids = builder.transform_split("train")
    x_valid, y_valid, valid_ids = builder.transform_split("valid")
    x_test, y_test, test_ids = builder.transform_split("test")

    search_rows: list[dict[str, Any]] = []
    best_name = None
    best_model = None
    best_valid_auprc = float("-inf")
    best_valid_metrics: dict[str, float] | None = None

    for name, model in _candidate_models(seed):
        try:
            model.fit(x_train, y_train)
            valid_scores = model.predict_proba(x_valid)[:, 1]
            valid_rows = _predict_rows(valid_ids, y_valid, valid_scores)
            valid_metrics = evaluator.evaluate_pairs(valid_rows)
            search_rows.append({"model": name, "status": "ok", **valid_metrics})
            print({"model": name, "valid_auprc": valid_metrics["auprc"], "valid_auroc": valid_metrics["auroc"]}, flush=True)
            if valid_metrics["auprc"] > best_valid_auprc:
                best_name = name
                best_model = deepcopy(model)
                best_valid_auprc = valid_metrics["auprc"]
                best_valid_metrics = valid_metrics
        except Exception as exc:  # pragma: no cover - baseline search fallback
            search_rows.append({"model": name, "status": "failed", "error": str(exc)})
            print({"model": name, "status": "failed", "error": str(exc)}, flush=True)

    assert best_model is not None
    valid_scores = best_model.predict_proba(x_valid)[:, 1]
    test_scores = best_model.predict_proba(x_test)[:, 1]
    valid_rows = _predict_rows(valid_ids, y_valid, valid_scores)
    test_rows = _predict_rows(test_ids, y_test, test_scores)
    valid_metrics = evaluator.evaluate_pairs(valid_rows)
    test_metrics = evaluator.evaluate_pairs(test_rows)

    save_json(
        {
            "selected_model": best_name,
            "best_valid_metrics": best_valid_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
            "feature_dim": len(builder.feature_names),
            "feature_names": builder.feature_names,
            "search_results": search_rows,
            "train_shape": list(x_train.shape),
            "valid_shape": list(x_valid.shape),
            "test_shape": list(x_test.shape),
        },
        output_dir / "metrics.json",
    )
    write_csv(search_rows, output_dir / "model_search.csv")
    write_csv(valid_rows, output_dir / "per_pair_predictions_valid.csv")
    write_csv(test_rows, output_dir / "per_pair_predictions_test.csv")


if __name__ == "__main__":
    main()
