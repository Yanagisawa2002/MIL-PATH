"""Loss helpers for hierarchical semi-supervised training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def path_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float,
    reduction: str = "mean",
    top_k: int | None = None,
) -> torch.Tensor:
    violations = F.relu(margin - positive_scores.unsqueeze(-1) + negative_scores)
    if reduction == "mean":
        return violations.mean()
    if reduction == "max":
        return violations.max(dim=-1).values.mean()
    if reduction == "topk_mean":
        k = min(int(top_k or 1), int(violations.size(-1)))
        return violations.topk(k=k, dim=-1).values.mean()
    msg = f"Unsupported path ranking reduction: {reduction}"
    raise ValueError(msg)


def pair_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    return 0.5 * ((probs_a - probs_b) ** 2).mean()


def pseudo_path_loss(pseudo_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return -(weights * F.logsigmoid(pseudo_scores)).mean()
