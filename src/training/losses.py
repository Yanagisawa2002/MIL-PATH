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


def path_binary_aux_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    logits = torch.cat([positive_logits.reshape(-1), negative_logits.reshape(-1)], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(positive_logits.reshape(-1)),
            torch.zeros_like(negative_logits.reshape(-1)),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def head_alignment_loss(
    evidence_logits: torch.Tensor,
    explanation_logits: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(evidence_logits), torch.sigmoid(explanation_logits))


def high_confidence_distillation_loss(
    evidence_logits: torch.Tensor,
    explanation_logits: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = mask.bool()
    if not valid_mask.any():
        zero = evidence_logits.new_zeros(())
        return zero, evidence_logits.new_tensor(0.0)

    teacher_probs = torch.sigmoid(explanation_logits.detach())[valid_mask]
    student_logits = evidence_logits[valid_mask]
    losses = F.binary_cross_entropy_with_logits(student_logits, teacher_probs, reduction="none")
    if weights is not None:
        sample_weights = weights[valid_mask].clamp(min=0.0)
        denom = sample_weights.sum().clamp(min=1e-6)
        loss = (losses * sample_weights).sum() / denom
    else:
        loss = losses.mean()
    return loss, evidence_logits.new_tensor(float(valid_mask.sum().item()))


def teacher_guided_reranker_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    bag_mask: torch.Tensor,
    is_gold: torch.Tensor,
    *,
    is_retrieved: torch.Tensor | None = None,
    use_retrieved_only: bool = False,
    top_k: int = 0,
    teacher_temperature: float = 1.0,
    student_temperature: float = 1.0,
    gold_boost: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_losses: list[torch.Tensor] = []
    valid_rows = 0
    for row_idx in range(student_logits.size(0)):
        row_mask = bag_mask[row_idx].bool()
        if use_retrieved_only and is_retrieved is not None:
            row_mask = row_mask & (is_retrieved[row_idx].bool() | is_gold[row_idx].bool())
        if not row_mask.any():
            continue

        row_gold = is_gold[row_idx].bool() & row_mask
        row_non_gold = (~is_gold[row_idx].bool()) & row_mask
        if not row_gold.any() or not row_non_gold.any():
            continue

        row_teacher = teacher_logits[row_idx].detach()
        if top_k > 0:
            non_gold_teacher = row_teacher.masked_fill(~row_non_gold, float("-inf"))
            k = min(max(1, int(top_k)), int(row_non_gold.sum().item()))
            top_idx = non_gold_teacher.topk(k=k, dim=0).indices
            support_mask = row_gold.clone()
            support_mask[top_idx] = True
            support_mask = support_mask & row_mask
        else:
            support_mask = row_mask

        if support_mask.sum().item() < 2:
            continue

        student_row = student_logits[row_idx][support_mask] / max(float(student_temperature), 1e-6)
        teacher_row = row_teacher[support_mask]
        if gold_boost != 0.0:
            teacher_row = teacher_row + float(gold_boost) * is_gold[row_idx][support_mask].float()
        teacher_row = teacher_row / max(float(teacher_temperature), 1e-6)

        target_probs = torch.softmax(teacher_row, dim=0)
        student_log_probs = torch.log_softmax(student_row, dim=0)
        row_loss = F.kl_div(student_log_probs, target_probs, reduction="batchmean")
        valid_losses.append(row_loss)
        valid_rows += 1

    if not valid_losses:
        zero = student_logits.new_zeros(())
        return zero, student_logits.new_tensor(0.0)

    return torch.stack(valid_losses).mean(), student_logits.new_tensor(float(valid_rows))


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    return 0.5 * ((probs_a - probs_b) ** 2).mean()


def pseudo_path_loss(pseudo_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return -(weights * F.logsigmoid(pseudo_scores)).mean()


def pseudo_pair_loss(pseudo_logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp(min=1e-6)
    return -(weights * F.logsigmoid(pseudo_logits)).sum() / denom
