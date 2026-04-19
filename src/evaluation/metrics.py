"""Pair-level, path-level, and explanation-level metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1
    num_pos = pos.sum()
    num_neg = len(labels) - num_pos
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - num_pos * (num_pos - 1) / 2) / (num_pos * num_neg))


def binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    cum_tp = np.cumsum(labels == 1)
    precision = cum_tp / np.arange(1, len(labels) + 1)
    recall = cum_tp / max(1, (labels == 1).sum())
    return float(np.trapz(precision, recall))


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(labels[order].sum() / max(1, labels.sum()))


def hits_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(labels[order].max())


def _pair_endpoints(row: dict[str, Any]) -> tuple[str, str]:
    drug_id = row.get("drug_id")
    disease_id = row.get("disease_id")
    if drug_id is not None and disease_id is not None:
        return str(drug_id), str(disease_id)
    pair_id = row["pair_id"]
    drug_id, disease_id = str(pair_id).split("::", 1)
    return drug_id, disease_id


def _grouped_pair_rows(
    rows: list[dict[str, Any]],
    query_side: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        drug_id, disease_id = _pair_endpoints(row)
        query_id = drug_id if query_side == "drug" else disease_id
        grouped[query_id].append(row)
    return grouped


def query_recall_at_k(
    rows: list[dict[str, Any]],
    *,
    k: int,
    query_side: str,
) -> float:
    grouped = _grouped_pair_rows(rows, query_side=query_side)
    per_query = []
    for query_rows in grouped.values():
        positives = [row for row in query_rows if int(row["label"]) == 1]
        if not positives:
            continue
        ranked = sorted(query_rows, key=lambda item: float(item["score"]), reverse=True)
        hits = sum(int(row["label"]) == 1 for row in ranked[:k])
        per_query.append(hits / len(positives))
    return float(np.mean(per_query)) if per_query else float("nan")


def query_hits_at_k(
    rows: list[dict[str, Any]],
    *,
    k: int,
    query_side: str,
) -> float:
    grouped = _grouped_pair_rows(rows, query_side=query_side)
    per_query = []
    for query_rows in grouped.values():
        positives = [row for row in query_rows if int(row["label"]) == 1]
        if not positives:
            continue
        ranked = sorted(query_rows, key=lambda item: float(item["score"]), reverse=True)
        hit = any(int(row["label"]) == 1 for row in ranked[:k])
        per_query.append(float(hit))
    return float(np.mean(per_query)) if per_query else float("nan")


def path_ranking_metrics(rows: list[dict[str, float]], ks: list[int]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)

    reciprocal_ranks = []
    hit_buckets = {k: [] for k in ks}
    recall_buckets = {k: [] for k in ks}
    for group_rows in grouped.values():
        ranked = sorted(group_rows, key=lambda item: item["score"], reverse=True)
        gold_positions = [idx + 1 for idx, row in enumerate(ranked) if row["is_gold"]]
        if not gold_positions:
            continue
        reciprocal_ranks.append(1.0 / min(gold_positions))
        for k in ks:
            hit_buckets[k].append(float(any(pos <= k for pos in gold_positions)))
            recall_buckets[k].append(sum(pos <= k for pos in gold_positions) / len(gold_positions))

    metrics = {"mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")}
    for k in ks:
        metrics[f"hits@{k}"] = float(np.mean(hit_buckets[k])) if hit_buckets[k] else float("nan")
        metrics[f"gold_recall@{k}"] = float(np.mean(recall_buckets[k])) if recall_buckets[k] else float("nan")
    return metrics


def faithfulness_drop(full_scores: np.ndarray, ablated_scores: np.ndarray) -> float:
    return float(np.mean(full_scores - ablated_scores))
