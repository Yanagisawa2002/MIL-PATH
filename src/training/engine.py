"""Stage-wise training skeleton for the hierarchical framework."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.training.losses import (
    consistency_loss,
    pair_classification_loss,
    path_ranking_loss,
    pseudo_path_loss,
)


class TrainingEngine:
    """Run stage-specific training steps on pre-batched tensors."""

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.loss_weights = config["training"]["loss_weights"]
        self.margin = float(config["training"]["ranking_margin"])
        self.path_rank_reduction = str(config["training"].get("stage2_ranking_reduction", "mean"))
        self.path_rank_top_k = int(config["training"].get("stage2_ranking_top_k", 2))
        self.stage4_pair_cls_weight = float(
            config["training"].get("stage4_pair_cls_weight", self.loss_weights["pair_cls"])
        )

    def stage1_loss(self, encoder, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_embeddings = encoder(batch["graph_artifact"])
        relation_loss = encoder.relation_reconstruction_loss(
            node_embeddings=node_embeddings,
            head_idx=batch["head_idx"],
            relation_idx=batch["relation_idx"],
            tail_idx=batch["tail_idx"],
        )
        type_loss = encoder.masked_type_prediction_loss(
            node_embeddings=node_embeddings,
            node_type_ids=batch["node_type_ids"],
        )
        total = relation_loss + type_loss
        return {
            "total": total,
            "relation_reconstruction": relation_loss,
            "masked_type": type_loss,
        }

    def stage2_loss(
        self,
        path_scorer,
        pair_embedding: torch.Tensor,
        positive_batch: dict[str, torch.Tensor],
        negative_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pos_scores, _ = path_scorer(pair_embedding=pair_embedding, **positive_batch)
        bsz, num_neg, seq_len, hidden_dim = negative_batch["node_states"].shape
        flat_neg = {
            "node_states": negative_batch["node_states"].reshape(bsz * num_neg, seq_len, hidden_dim),
            "relation_ids": negative_batch["relation_ids"].reshape(bsz * num_neg, seq_len - 1),
            "node_type_ids": negative_batch["node_type_ids"].reshape(bsz * num_neg, seq_len),
            "mask": negative_batch["mask"].reshape(bsz * num_neg, seq_len),
        }
        repeated_pair = pair_embedding.unsqueeze(1).repeat(1, num_neg, 1).reshape(bsz * num_neg, -1)
        neg_scores, _ = path_scorer(pair_embedding=repeated_pair, **flat_neg)
        neg_scores = neg_scores.reshape(bsz, num_neg)
        rank_loss = path_ranking_loss(
            pos_scores,
            neg_scores,
            margin=self.margin,
            reduction=self.path_rank_reduction,
            top_k=self.path_rank_top_k,
        )
        return {
            "total": self.loss_weights["path_rank"] * rank_loss,
            "path_rank": rank_loss,
        }

    def stage3_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_loss = pair_classification_loss(model_outputs["pair_score"], labels)
        return {
            "total": self.loss_weights["pair_cls"] * pair_loss,
            "pair_cls": pair_loss,
        }

    def stage4_loss(
        self,
        model_outputs_a: dict[str, torch.Tensor],
        model_outputs_b: dict[str, torch.Tensor],
        pseudo_scores: torch.Tensor,
        pseudo_weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_loss = pair_classification_loss(model_outputs_a["pair_score"], labels)
        cons_loss = consistency_loss(
            model_outputs_a["path_attention"],
            model_outputs_b["path_attention"],
        )
        pseudo_loss = pseudo_path_loss(pseudo_scores, pseudo_weights)
        total = (
            self.stage4_pair_cls_weight * pair_loss
            + self.loss_weights["consistency"] * cons_loss
            + self.loss_weights["pseudo"] * pseudo_loss
        )
        return {
            "total": total,
            "pair_cls": pair_loss,
            "consistency": cons_loss,
            "pseudo": pseudo_loss,
        }

    def summarize_history(self, history: list[dict[str, float]]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for step in history:
            for name, value in step.items():
                buckets[name].append(float(value))
        return {name: sum(values) / len(values) for name, values in buckets.items() if values}
