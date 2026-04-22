"""Confidence-gated pseudo-rationale selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class PseudoSelectionResult:
    accepted_mask: torch.Tensor
    confidence: torch.Tensor
    summary: dict[str, Any]


@dataclass(slots=True)
class PseudoPairSelectionResult:
    accepted_mask: torch.Tensor
    confidence: torch.Tensor
    agreement: torch.Tensor
    summary: dict[str, Any]


class PseudoRationaleSelector:
    """Select pseudo rationales for positive-no-path pairs."""

    def __init__(self, config: dict[str, Any], trusted_schema_ids: set[str] | None = None) -> None:
        self.config = config
        self.trusted_schema_ids = trusted_schema_ids or set()

    def select(
        self,
        pair_logits: torch.Tensor,
        path_logits: torch.Tensor,
        top_schema_ids: list[str] | list[list[str]],
        stability: torch.Tensor,
        bag_mask: torch.Tensor | None = None,
    ) -> PseudoSelectionResult:
        pair_prob = torch.sigmoid(pair_logits)
        if bag_mask is None:
            valid_counts = torch.full(
                (path_logits.size(0),),
                path_logits.size(1),
                dtype=torch.long,
                device=path_logits.device,
            )
            masked_logits = path_logits
        else:
            bag_mask = bag_mask.to(dtype=torch.bool, device=path_logits.device)
            valid_counts = bag_mask.sum(dim=1)
            masked_logits = path_logits.masked_fill(~bag_mask, float("-inf"))

        path_prob = torch.sigmoid(masked_logits)
        top1_prob, top1_idx = path_prob.max(dim=1)
        top2_values = path_prob.topk(k=min(2, path_prob.size(1)), dim=1).values
        top2_prob = top2_values[:, -1]
        margin = torch.where(
            valid_counts <= 1,
            torch.ones_like(top1_prob),
            top1_prob - top2_prob,
        )

        accepted = (
            (pair_prob >= self.config["pair_score_threshold"])
            & (top1_prob >= self.config["top1_path_threshold"])
            & (margin >= self.config["top12_margin_threshold"])
            & (stability >= self.config["min_stability"])
            & (valid_counts > 0)
        )
        if self.config["require_schema_whitelist"] and self.trusted_schema_ids:
            if top_schema_ids and isinstance(top_schema_ids[0], list):
                resolved_schema_ids = []
                for batch_idx, path_idx in enumerate(top1_idx.tolist()):
                    schema_list = top_schema_ids[batch_idx]
                    if path_idx < len(schema_list):
                        resolved_schema_ids.append(schema_list[path_idx])
                    else:
                        resolved_schema_ids.append("")
            else:
                resolved_schema_ids = [top_schema_ids[idx] for idx in top1_idx.tolist()]
            schema_gate = torch.tensor(
                [schema_id in self.trusted_schema_ids for schema_id in resolved_schema_ids],
                device=accepted.device,
            )
            accepted = accepted & schema_gate

        confidence = (pair_prob * top1_prob * stability).clamp(min=0.0, max=1.0)
        accepted = accepted & (confidence >= self.config["min_confidence"])
        return PseudoSelectionResult(
            accepted_mask=accepted,
            confidence=confidence,
            summary={
                "num_pairs": int(pair_logits.numel()),
                "num_accepted": int(accepted.sum().item()),
                "mean_confidence": float(confidence.mean().item()),
                "num_nonempty_bags": int((valid_counts > 0).sum().item()),
                "num_singleton_bags": int((valid_counts == 1).sum().item()),
            },
        )


class PseudoPositivePairSelector:
    """Select mechanism-consistent pseudo-positive pairs from unlabeled pools."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def select(
        self,
        pair_logits: torch.Tensor,
        path_logits: torch.Tensor,
        *,
        binary_logits: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
    ) -> PseudoPairSelectionResult:
        pair_prob = torch.sigmoid(pair_logits)
        if bag_mask is None:
            bag_mask = torch.ones_like(path_logits, dtype=torch.bool)
        else:
            bag_mask = bag_mask.to(dtype=torch.bool, device=path_logits.device)
        valid_counts = bag_mask.sum(dim=1)
        masked_path_logits = path_logits.masked_fill(~bag_mask, float("-inf"))
        path_prob = torch.sigmoid(masked_path_logits)
        top1_prob, top1_idx = path_prob.max(dim=1)
        top2_values = path_prob.topk(k=min(2, path_prob.size(1)), dim=1).values
        top2_prob = top2_values[:, -1]
        margin = torch.where(valid_counts <= 1, torch.ones_like(top1_prob), top1_prob - top2_prob)

        if binary_logits is None:
            binary_prob = path_prob
            top1_binary_prob = top1_prob
            agreement = torch.ones_like(top1_prob)
        else:
            masked_binary_logits = binary_logits.masked_fill(~bag_mask, float("-inf"))
            binary_prob = torch.sigmoid(masked_binary_logits)
            top1_binary_prob = binary_prob.gather(1, top1_idx.unsqueeze(1)).squeeze(1)
            agreement = 1.0 - (top1_prob - top1_binary_prob).abs().clamp(min=0.0, max=1.0)

        if reliability is None:
            reliability = torch.where(valid_counts > 0, torch.ones_like(pair_prob), torch.zeros_like(pair_prob))
        if uncertainty is None:
            uncertainty = 1.0 - reliability

        accepted = (
            (pair_prob >= self.config["pair_score_threshold"])
            & (top1_prob >= self.config["top1_path_threshold"])
            & (margin >= self.config["top12_margin_threshold"])
            & (top1_binary_prob >= self.config["top1_binary_threshold"])
            & (reliability >= self.config["min_reliability"])
            & (uncertainty <= self.config["max_uncertainty"])
            & (agreement >= self.config["min_agreement"])
            & (valid_counts >= int(self.config.get("min_bag_size", 1)))
        )

        confidence = (pair_prob * top1_prob * top1_binary_prob * reliability * agreement).clamp(min=0.0, max=1.0)
        accepted = accepted & (confidence >= self.config["min_confidence"])
        confidence = confidence * float(self.config.get("weight_scale", 1.0))
        return PseudoPairSelectionResult(
            accepted_mask=accepted,
            confidence=confidence,
            agreement=agreement,
            summary={
                "num_pairs": int(pair_logits.numel()),
                "num_accepted": int(accepted.sum().item()),
                "accept_rate": float(accepted.float().mean().item()),
                "mean_confidence": float(confidence.mean().item()),
                "mean_agreement": float(agreement.mean().item()),
                "num_nonempty_bags": int((valid_counts > 0).sum().item()),
                "num_minbag_eligible": int((valid_counts >= int(self.config.get("min_bag_size", 1))).sum().item()),
            },
        )
