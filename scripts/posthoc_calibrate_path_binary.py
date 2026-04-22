"""Fit a post-hoc path-binary calibrator from hard-ranking CSVs.

This script trains a lightweight logistic calibrator on the validation
`per_path_hard_ranking_*.csv` rows and writes calibrated logits/probabilities for
both valid and test files. It is designed to improve controlled path-binary
evaluation without perturbing the pair/pipeline training objective.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import binary_auprc, binary_auroc
from src.utils.io import ensure_dir, save_json


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _score_group_features(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    work = df[["pair_id", column]].copy()
    grouped = work.groupby("pair_id")[column]
    work[f"{prefix}_row_max"] = grouped.transform("max")

    def _second_best(values: pd.Series) -> float:
        arr = np.sort(values.to_numpy(dtype=np.float64))
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            return float(arr[-1])
        return float(arr[-2])

    second_best = grouped.transform(_second_best)
    work[f"{prefix}_gap_to_max"] = work[column] - work[f"{prefix}_row_max"]
    work[f"{prefix}_margin_to_second"] = work[column] - second_best
    work[f"{prefix}_is_row_top"] = (work[column] >= work[f"{prefix}_row_max"] - 1e-8).astype(np.float64)
    return work.drop(columns=["pair_id", column])


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    data = df.copy()
    for score_column in ("evidence_score", "explanation_score", "binary_score"):
        data[score_column] = data[score_column].astype(np.float64)
    data["is_gold"] = data["is_gold"].astype(np.float64)

    data["evidence_prob"] = _sigmoid_np(data["evidence_score"].to_numpy())
    data["explanation_prob"] = _sigmoid_np(data["explanation_score"].to_numpy())
    data["binary_prob"] = _sigmoid_np(data["binary_score"].to_numpy())

    data["agreement_exp_bin"] = 1.0 - np.abs(data["explanation_prob"] - data["binary_prob"])
    data["agreement_evi_exp"] = 1.0 - np.abs(data["evidence_prob"] - data["explanation_prob"])
    data["agreement_evi_bin"] = 1.0 - np.abs(data["evidence_prob"] - data["binary_prob"])
    data["binary_minus_explanation"] = data["binary_score"] - data["explanation_score"]
    data["binary_minus_evidence"] = data["binary_score"] - data["evidence_score"]
    data["explanation_minus_evidence"] = data["explanation_score"] - data["evidence_score"]

    feature_parts = [
        data[
            [
                "evidence_score",
                "explanation_score",
                "binary_score",
                "evidence_prob",
                "explanation_prob",
                "binary_prob",
                "agreement_exp_bin",
                "agreement_evi_exp",
                "agreement_evi_bin",
                "binary_minus_explanation",
                "binary_minus_evidence",
                "explanation_minus_evidence",
            ]
        ]
    ]
    feature_names = list(feature_parts[0].columns)

    for score_column, prefix in (
        ("evidence_score", "evidence"),
        ("explanation_score", "explanation"),
        ("binary_score", "binary"),
    ):
        group_features = _score_group_features(data, score_column, prefix)
        feature_parts.append(group_features)
        feature_names.extend(list(group_features.columns))

    feature_df = pd.concat(feature_parts, axis=1)
    return feature_df.to_numpy(dtype=np.float32), feature_names


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def _fit_logistic_calibrator(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    lr: float,
    epochs: int,
    weight_decay: float,
) -> tuple[torch.nn.Linear, list[float]]:
    device = torch.device("cpu")
    x_tensor = torch.from_numpy(train_x).to(device)
    y_tensor = torch.from_numpy(train_y.astype(np.float32)).to(device)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_count = float((train_y == 1).sum())
    neg_count = float((train_y == 0).sum())
    pos_weight = torch.tensor(neg_count / max(1.0, pos_count), dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return model, losses


def _predict_logits(model: torch.nn.Linear, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float()).squeeze(-1)
    return logits.detach().cpu().numpy().astype(np.float64)


def _binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probs = _sigmoid_np(logits)
    return {
        "auroc": binary_auroc(labels, logits),
        "auprc": binary_auprc(labels, logits),
        "mean_positive_score": float(logits[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_negative_score": float(logits[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "mean_positive_prob": float(probs[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_negative_prob": float(probs[labels == 0].mean()) if np.any(labels == 0) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-input", type=str, required=True)
    parser.add_argument("--test-input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="posthoc_binary_calibration")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    valid_df = pd.read_csv(args.valid_input)
    test_df = pd.read_csv(args.test_input)

    valid_x_raw, feature_names = _build_features(valid_df)
    test_x_raw, _ = _build_features(test_df)
    valid_y = valid_df["is_gold"].astype(np.int64).to_numpy()
    test_y = test_df["is_gold"].astype(np.int64).to_numpy()

    valid_x, test_x, mean, std = _standardize(valid_x_raw, test_x_raw)
    model, losses = _fit_logistic_calibrator(
        valid_x,
        valid_y,
        lr=float(args.lr),
        epochs=int(args.epochs),
        weight_decay=float(args.weight_decay),
    )

    valid_logits = _predict_logits(model, valid_x)
    test_logits = _predict_logits(model, test_x)
    valid_probs = _sigmoid_np(valid_logits)
    test_probs = _sigmoid_np(test_logits)

    valid_df["posthoc_binary_score"] = valid_logits
    valid_df["posthoc_binary_prob"] = valid_probs
    test_df["posthoc_binary_score"] = test_logits
    test_df["posthoc_binary_prob"] = test_probs

    valid_out = output_dir / f"{args.name}_valid.csv"
    test_out = output_dir / f"{args.name}_test.csv"
    valid_df.to_csv(valid_out, index=False)
    test_df.to_csv(test_out, index=False)

    linear = model.state_dict()
    summary = {
        "name": args.name,
        "valid_input": str(args.valid_input),
        "test_input": str(args.test_input),
        "feature_names": feature_names,
        "train_shape": [int(valid_x.shape[0]), int(valid_x.shape[1])],
        "optimizer": {
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "weight_decay": float(args.weight_decay),
        },
        "training_loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
        },
        "valid_metrics": _binary_metrics(valid_y, valid_logits),
        "test_metrics": _binary_metrics(test_y, test_logits),
        "linear_weight": linear["weight"].detach().cpu().numpy().reshape(-1).tolist(),
        "linear_bias": linear["bias"].detach().cpu().numpy().reshape(-1).tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "valid_output_csv": str(valid_out),
        "test_output_csv": str(test_out),
    }
    save_json(summary, output_dir / f"{args.name}.json")
    print(summary)


if __name__ == "__main__":
    main()
