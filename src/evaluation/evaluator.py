"""Evaluation and artifact writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import (
    binary_auprc,
    binary_auroc,
    faithfulness_drop,
    hits_at_k,
    query_hits_at_k,
    query_recall_at_k,
    path_ranking_metrics,
    recall_at_k,
)
from src.utils.io import ensure_dir, save_json, write_csv


class Evaluator:
    """Compute metrics and write standard experiment outputs."""

    def __init__(self, output_dir: str | Path, ks: list[int]) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.ks = ks

    def evaluate_pairs(self, pair_rows: list[dict[str, Any]]) -> dict[str, float]:
        labels = np.asarray([row["label"] for row in pair_rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in pair_rows], dtype=np.float64)
        metrics = {
            "auroc": binary_auroc(labels, scores),
            "auprc": binary_auprc(labels, scores),
        }
        for k in self.ks:
            global_recall = recall_at_k(labels, scores, k)
            global_hits = hits_at_k(labels, scores, k)
            drug_query_recall = query_recall_at_k(pair_rows, k=k, query_side="drug")
            disease_query_recall = query_recall_at_k(pair_rows, k=k, query_side="disease")
            drug_query_hits = query_hits_at_k(pair_rows, k=k, query_side="drug")
            disease_query_hits = query_hits_at_k(pair_rows, k=k, query_side="disease")

            metrics[f"global_recall@{k}"] = global_recall
            metrics[f"global_hits@{k}"] = global_hits
            metrics[f"drug_recall@{k}"] = drug_query_recall
            metrics[f"disease_recall@{k}"] = disease_query_recall
            metrics[f"drug_hits@{k}"] = drug_query_hits
            metrics[f"disease_hits@{k}"] = disease_query_hits
            metrics[f"recall@{k}"] = float(np.nanmean([drug_query_recall, disease_query_recall]))
            metrics[f"hits@{k}"] = float(np.nanmean([drug_query_hits, disease_query_hits]))
        return metrics

    def evaluate_paths(self, path_rows: list[dict[str, Any]]) -> dict[str, float]:
        return path_ranking_metrics(path_rows, self.ks)

    def evaluate_explanations(
        self,
        full_scores: np.ndarray,
        ablated_scores: np.ndarray,
    ) -> dict[str, float]:
        return {"faithfulness_drop": faithfulness_drop(full_scores, ablated_scores)}

    def write_outputs(
        self,
        metrics: dict[str, Any],
        pair_rows: list[dict[str, Any]],
        path_rows: list[dict[str, Any]],
        pseudo_summary: dict[str, Any],
    ) -> None:
        save_json(metrics, self.output_dir / "metrics.json")
        write_csv(pair_rows, self.output_dir / "per_pair_predictions.csv")
        write_csv(path_rows, self.output_dir / "per_path_ranking.csv")
        save_json(pseudo_summary, self.output_dir / "pseudo_label_summary.json")
