"""Direct pair branch, cross-branch interaction, and final pair scoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DirectPairEncoder(nn.Module):
    """Encode endpoint embeddings into a direct pair score."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        pair_feature_dim: int = 0,
        pair_feature_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.pair_feature_dim = int(pair_feature_dim)
        self.pair_feature_proj = None
        pair_proj_input_dim = hidden_dim * 4
        if self.pair_feature_dim > 0:
            feature_hidden_dim = int(pair_feature_hidden_dim or hidden_dim)
            self.pair_feature_proj = nn.Sequential(
                nn.Linear(self.pair_feature_dim, feature_hidden_dim),
                nn.LayerNorm(feature_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            pair_proj_input_dim += feature_hidden_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_proj_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        drug_embedding: torch.Tensor,
        disease_embedding: torch.Tensor,
        pair_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_parts = [
            drug_embedding,
            disease_embedding,
            torch.abs(drug_embedding - disease_embedding),
            drug_embedding * disease_embedding,
        ]
        if self.pair_feature_proj is not None:
            if pair_features is None:
                msg = "pair_features are required when direct pair feature augmentation is enabled"
                raise ValueError(msg)
            pair_parts.append(self.pair_feature_proj(pair_features))
        pair_embedding = torch.cat(
            [
                *pair_parts,
            ],
            dim=-1,
        )
        pair_hidden = self.pair_proj(pair_embedding)
        direct_score = self.score_head(pair_hidden).squeeze(-1)
        return pair_hidden, direct_score


class PathBagAggregator(nn.Module):
    """Aggregate path evidence into a single pair-level path score."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.mode = config["type"]
        self.top_k = config["top_k"]
        self.temperature = config["attention_temperature"]
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        bag_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_scores = path_scores
        empty_rows = None
        if bag_mask is not None:
            empty_rows = ~bag_mask.any(dim=1)
            masked_scores = path_scores.masked_fill(~bag_mask, float("-inf"))

        def _aggregate_repr(weights: torch.Tensor) -> torch.Tensor:
            agg_repr = (weights.unsqueeze(-1) * path_reprs).sum(dim=1)
            if empty_rows is not None and empty_rows.any():
                agg_repr = agg_repr.masked_fill(empty_rows.unsqueeze(-1), 0.0)
            return agg_repr

        if self.mode == "max":
            agg_score = masked_scores.max(dim=1).values
            weights = torch.nn.functional.one_hot(
                masked_scores.argmax(dim=1),
                num_classes=masked_scores.size(1),
            ).float()
            if empty_rows is not None and empty_rows.any():
                agg_score = agg_score.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "topk_logsumexp":
            k = min(self.top_k, masked_scores.size(1))
            top_scores, top_idx = masked_scores.topk(k=k, dim=1)
            agg_score = torch.logsumexp(top_scores, dim=1) - torch.log(
                torch.tensor(float(k), device=masked_scores.device)
            )
            weights = torch.zeros_like(masked_scores)
            weights.scatter_(1, top_idx, 1.0 / k)
            if empty_rows is not None and empty_rows.any():
                agg_score = agg_score.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "attention":
            pair_expanded = pair_repr.unsqueeze(1).expand_as(path_reprs)
            attn_logits = self.attn(torch.cat([pair_expanded, path_reprs], dim=-1)).squeeze(-1)
            if bag_mask is not None:
                attn_logits = attn_logits.masked_fill(~bag_mask, float("-inf"))
                if empty_rows is not None and empty_rows.any():
                    attn_logits = attn_logits.masked_fill(empty_rows.unsqueeze(1), 0.0)
            weights = torch.softmax(attn_logits / self.temperature, dim=1)
            if empty_rows is not None and empty_rows.any():
                weights[empty_rows] = 0.0
            agg_score = (weights * path_scores).sum(dim=1)
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "noisy_or":
            probs = torch.sigmoid(masked_scores)
            if bag_mask is not None:
                probs = probs * bag_mask.float()
            agg_prob = 1.0 - torch.prod(1.0 - probs.clamp(max=0.999), dim=1)
            weights = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            logits = torch.logit(agg_prob.clamp(1e-5, 1 - 1e-5))
            if empty_rows is not None and empty_rows.any():
                logits = logits.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return logits, weights, _aggregate_repr(weights)

        msg = f"Unsupported aggregator mode: {self.mode}"
        raise ValueError(msg)


class CrossBranchInteractor(nn.Module):
    """Refine direct and mechanistic branches with explicit cross-branch gating."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", True))
        self.alpha = float(config.get("alpha", 0.5))
        dropout = float(config.get("dropout", 0.1))
        interaction_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        fusion_input_dim = hidden_dim * 4 + 3
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, interaction_hidden_dim),
            nn.LayerNorm(interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direct_gate = nn.Sequential(
            nn.Linear(interaction_hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.path_gate = nn.Sequential(
            nn.Linear(interaction_hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.direct_update = nn.Linear(hidden_dim, hidden_dim)
        self.path_update = nn.Linear(hidden_dim, hidden_dim)
        self.direct_norm = nn.LayerNorm(hidden_dim)
        self.path_norm = nn.LayerNorm(hidden_dim)
        self.direct_delta_head = nn.Linear(hidden_dim, 1)
        self.path_delta_head = nn.Linear(hidden_dim, 1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden_dim, 1),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        direct_score: torch.Tensor,
        path_repr: torch.Tensor,
        path_score: torch.Tensor,
        bag_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            final_score = self.alpha * direct_score + (1.0 - self.alpha) * path_score
            fusion_gate = torch.where(
                bag_available.squeeze(-1).bool(),
                torch.full_like(direct_score, self.alpha),
                torch.ones_like(direct_score),
            )
            return {
                "refined_pair_repr": pair_repr,
                "refined_path_repr": path_repr,
                "refined_direct_score": direct_score,
                "refined_path_score": path_score,
                "fusion_gate": fusion_gate,
                "interaction_delta": torch.zeros_like(direct_score),
                "pair_score": final_score,
            }

        pair_path_features = torch.cat(
            [
                pair_repr,
                path_repr,
                torch.abs(pair_repr - path_repr),
                pair_repr * path_repr,
                direct_score.unsqueeze(-1),
                path_score.unsqueeze(-1),
                bag_available,
            ],
            dim=-1,
        )
        context = self.context_mlp(pair_path_features)
        direct_gate = bag_available * self.direct_gate(context)
        path_gate = bag_available * self.path_gate(context)

        refined_pair_repr = self.direct_norm(pair_repr + direct_gate * self.direct_update(path_repr))
        refined_path_repr = self.path_norm(path_repr + path_gate * self.path_update(pair_repr))

        refined_direct_score = direct_score + bag_available.squeeze(-1) * self.direct_delta_head(refined_pair_repr).squeeze(-1)
        refined_path_score = path_score + bag_available.squeeze(-1) * self.path_delta_head(refined_path_repr).squeeze(-1)

        fusion_features = torch.cat(
            [
                refined_pair_repr,
                refined_path_repr,
                refined_direct_score.unsqueeze(-1),
                refined_path_score.unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.fusion_gate(fusion_features)).squeeze(-1)
        learned_gate = torch.where(bag_available.squeeze(-1).bool(), learned_gate, torch.ones_like(learned_gate))
        interaction_delta = (
            self.residual_head(fusion_features).squeeze(-1) * bag_available.squeeze(-1)
        )
        final_score = (
            learned_gate * refined_direct_score
            + (1.0 - learned_gate) * refined_path_score
            + interaction_delta
        )
        return {
            "refined_pair_repr": refined_pair_repr,
            "refined_path_repr": refined_path_repr,
            "refined_direct_score": refined_direct_score,
            "refined_path_score": refined_path_score,
            "fusion_gate": learned_gate,
            "interaction_delta": interaction_delta,
            "pair_score": final_score,
        }


class HierarchicalPairModel(nn.Module):
    """Combine direct pair and path bag branches into the final pair score."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        pair_feature_cfg = config.get("direct_pair_features", {})
        self.direct_pair = DirectPairEncoder(
            hidden_dim=hidden_dim,
            dropout=config["path_scorer"]["dropout"],
            pair_feature_dim=int(pair_feature_cfg.get("feature_dim", 0)) if pair_feature_cfg.get("enabled", False) else 0,
            pair_feature_hidden_dim=pair_feature_cfg.get("hidden_dim"),
        )
        self.aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=config["aggregator"])
        interaction_cfg = dict(config.get("interaction", {}))
        interaction_cfg.setdefault("alpha", config["aggregator"]["alpha"])
        interaction_cfg.setdefault("dropout", config["path_scorer"]["dropout"])
        self.interactor = CrossBranchInteractor(hidden_dim=hidden_dim, config=interaction_cfg)

    def forward(
        self,
        drug_embedding: torch.Tensor,
        disease_embedding: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        pair_features: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pair_repr, direct_score = self.direct_pair(drug_embedding, disease_embedding, pair_features=pair_features)
        path_score, attn, path_bag_repr = self.aggregator(pair_repr, path_scores, path_reprs, bag_mask=bag_mask)
        if bag_mask is None:
            bag_available = torch.ones(pair_repr.size(0), 1, dtype=pair_repr.dtype, device=pair_repr.device)
        else:
            bag_available = bag_mask.any(dim=1, keepdim=True).to(dtype=pair_repr.dtype)
        interaction_outputs = self.interactor(
            pair_repr=pair_repr,
            direct_score=direct_score,
            path_repr=path_bag_repr,
            path_score=path_score,
            bag_available=bag_available,
        )
        return {
            "pair_repr": pair_repr,
            "direct_pair_score": direct_score,
            "path_bag_score": path_score,
            "path_bag_repr": path_bag_repr,
            "refined_pair_repr": interaction_outputs["refined_pair_repr"],
            "refined_path_repr": interaction_outputs["refined_path_repr"],
            "refined_direct_score": interaction_outputs["refined_direct_score"],
            "refined_path_score": interaction_outputs["refined_path_score"],
            "fusion_gate": interaction_outputs["fusion_gate"],
            "interaction_delta": interaction_outputs["interaction_delta"],
            "pair_score": interaction_outputs["pair_score"],
            "path_attention": attn,
        }
