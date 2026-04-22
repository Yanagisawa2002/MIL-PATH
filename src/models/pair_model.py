"""Direct pair branch, cross-branch interaction, and final pair scoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


MASK_NEG_LARGE = -1e9


def _masked_topk_mask(scores: torch.Tensor, bag_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return bag_mask.bool()
    k = min(max(1, int(top_k)), int(scores.size(1)))
    masked_scores = scores.masked_fill(~bag_mask.bool(), MASK_NEG_LARGE)
    top_idx = masked_scores.topk(k=k, dim=1).indices
    top_mask = torch.zeros_like(bag_mask, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)
    return top_mask & bag_mask.bool()


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
        self.min_top_k = int(config.get("min_top_k", 1))
        self.temperature = config["attention_temperature"]
        self.selector_temperature = float(config.get("selector_temperature", self.temperature))
        self.adaptive_mass_threshold = float(config.get("adaptive_mass_threshold", 0.8))
        self.adaptive_use_confidence = bool(config.get("adaptive_use_confidence", True))
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def _adaptive_topk_selection(
        self,
        masked_selection_scores: torch.Tensor,
        path_confidence: torch.Tensor | None,
        bag_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_k = min(self.top_k, masked_selection_scores.size(1))
        top_selector, top_idx = masked_selection_scores.topk(k=max_k, dim=1)
        top_weight_logits = top_selector
        if path_confidence is not None and self.adaptive_use_confidence:
            top_confidence = path_confidence.gather(1, top_idx).clamp(min=1e-6)
            top_weight_logits = top_weight_logits + torch.log(top_confidence)

        normalized_weights = torch.softmax(top_weight_logits / self.selector_temperature, dim=1)
        cumulative_mass = normalized_weights.cumsum(dim=1)
        threshold_hits = cumulative_mass >= self.adaptive_mass_threshold
        first_hit = threshold_hits.float().argmax(dim=1) + 1

        if bag_mask is not None:
            valid_count = bag_mask.sum(dim=1).clamp(min=1, max=max_k)
        else:
            valid_count = torch.full(
                (masked_selection_scores.size(0),),
                fill_value=max_k,
                device=masked_selection_scores.device,
                dtype=torch.long,
            )

        selected_k = first_hit.clamp(min=self.min_top_k, max=max_k)
        selected_k = torch.minimum(selected_k, valid_count)
        selection_mask = (
            torch.arange(max_k, device=masked_selection_scores.device).unsqueeze(0)
            < selected_k.unsqueeze(1)
        )
        return top_selector, top_idx, selection_mask

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        bag_mask: torch.Tensor | None = None,
        selector_scores: torch.Tensor | None = None,
        path_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value_scores = path_scores
        selection_scores = path_scores if selector_scores is None else selector_scores
        masked_scores = value_scores
        masked_selection_scores = selection_scores
        empty_rows = None
        if bag_mask is not None:
            empty_rows = ~bag_mask.any(dim=1)
            masked_scores = value_scores.masked_fill(~bag_mask, MASK_NEG_LARGE)
            masked_selection_scores = selection_scores.masked_fill(~bag_mask, MASK_NEG_LARGE)

        def _aggregate_repr(weights: torch.Tensor) -> torch.Tensor:
            agg_repr = (weights.unsqueeze(-1) * path_reprs).sum(dim=1)
            if empty_rows is not None and empty_rows.any():
                agg_repr = agg_repr.masked_fill(empty_rows.unsqueeze(-1), 0.0)
            return agg_repr

        if self.mode == "max":
            selected_idx = masked_selection_scores.argmax(dim=1)
            agg_score = masked_scores.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
            weights = torch.nn.functional.one_hot(
                selected_idx,
                num_classes=masked_scores.size(1),
            ).float()
            if empty_rows is not None and empty_rows.any():
                agg_score = agg_score.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "topk_logsumexp":
            k = min(self.top_k, masked_selection_scores.size(1))
            top_selector, top_idx = masked_selection_scores.topk(k=k, dim=1)
            top_scores = value_scores.gather(1, top_idx)
            if selector_scores is None and path_confidence is None:
                agg_score = torch.logsumexp(top_scores, dim=1) - torch.log(
                    torch.tensor(float(k), device=masked_scores.device)
                )
                top_weights = torch.full_like(top_scores, 1.0 / k)
            else:
                top_weight_logits = top_selector
                if path_confidence is not None:
                    top_confidence = path_confidence.gather(1, top_idx).clamp(min=1e-6)
                    top_weight_logits = top_weight_logits + torch.log(top_confidence)
                top_weights = torch.softmax(top_weight_logits / self.selector_temperature, dim=1)
                agg_score = torch.logsumexp(
                    top_scores + torch.log(top_weights.clamp(min=1e-6)),
                    dim=1,
                )
            weights = torch.zeros_like(masked_scores)
            weights.scatter_(1, top_idx, top_weights)
            if empty_rows is not None and empty_rows.any():
                agg_score = agg_score.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "adaptive_topk_logsumexp":
            top_selector, top_idx, selection_mask = self._adaptive_topk_selection(
                masked_selection_scores=masked_selection_scores,
                path_confidence=path_confidence,
                bag_mask=bag_mask,
            )
            top_scores = value_scores.gather(1, top_idx)
            if selector_scores is None and path_confidence is None:
                masked_top_scores = top_scores.masked_fill(~selection_mask, MASK_NEG_LARGE)
                selected_k = selection_mask.sum(dim=1).clamp(min=1).to(dtype=top_scores.dtype)
                agg_score = torch.logsumexp(masked_top_scores, dim=1) - torch.log(selected_k)
                top_weights = selection_mask.to(dtype=top_scores.dtype) / selected_k.unsqueeze(1)
            else:
                top_weight_logits = top_selector
                if path_confidence is not None:
                    top_confidence = path_confidence.gather(1, top_idx).clamp(min=1e-6)
                    top_weight_logits = top_weight_logits + torch.log(top_confidence)
                top_weight_logits = top_weight_logits.masked_fill(~selection_mask, MASK_NEG_LARGE)
                top_weights = torch.softmax(top_weight_logits / self.selector_temperature, dim=1)
                top_weights = top_weights * selection_mask.to(dtype=top_scores.dtype)
                top_weights = top_weights / top_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                agg_score = torch.logsumexp(
                    top_scores + torch.log(top_weights.clamp(min=1e-6)),
                    dim=1,
                )
            weights = torch.zeros_like(masked_scores)
            weights.scatter_(1, top_idx, top_weights)
            if empty_rows is not None and empty_rows.any():
                agg_score = agg_score.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "attention":
            pair_expanded = pair_repr.unsqueeze(1).expand_as(path_reprs)
            attn_logits = self.attn(torch.cat([pair_expanded, path_reprs], dim=-1)).squeeze(-1)
            if selector_scores is not None:
                attn_logits = attn_logits + selection_scores
            if path_confidence is not None:
                attn_logits = attn_logits + torch.log(path_confidence.clamp(min=1e-6))
            if bag_mask is not None:
                attn_logits = attn_logits.masked_fill(~bag_mask, MASK_NEG_LARGE)
                if empty_rows is not None and empty_rows.any():
                    attn_logits = attn_logits.masked_fill(empty_rows.unsqueeze(1), 0.0)
            weights = torch.softmax(attn_logits / self.temperature, dim=1)
            if empty_rows is not None and empty_rows.any():
                weights[empty_rows] = 0.0
            agg_score = (weights * value_scores).sum(dim=1)
            return agg_score, weights, _aggregate_repr(weights)

        if self.mode == "noisy_or":
            probs = torch.sigmoid(masked_scores)
            if bag_mask is not None:
                probs = probs * bag_mask.float()
            if path_confidence is not None:
                probs = probs * path_confidence.clamp(min=0.0, max=1.0)
            agg_prob = 1.0 - torch.prod(1.0 - probs.clamp(max=0.999), dim=1)
            weights = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            logits = torch.logit(agg_prob.clamp(1e-5, 1 - 1e-5))
            if empty_rows is not None and empty_rows.any():
                logits = logits.masked_fill(empty_rows, 0.0)
                weights[empty_rows] = 0.0
            return logits, weights, _aggregate_repr(weights)

        msg = f"Unsupported aggregator mode: {self.mode}"
        raise ValueError(msg)


class HierarchicalPathBagAggregator(nn.Module):
    """Aggregate paths in two stages: within group, then across groups."""

    def __init__(self, hidden_dim: int, base_config: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        if not self.enabled:
            self.within_group_aggregator = None
            self.between_group_aggregator = None
            self.group_by = []
            self.max_groups = 0
            return
        self.group_by = [str(item) for item in config.get("group_by", ["schema", "hop", "source"])]
        self.max_groups = int(config.get("max_groups", 16))
        within_group_cfg = dict(base_config)
        within_group_cfg.update(dict(config.get("within_group_aggregator", {})))
        between_group_cfg = dict(base_config)
        between_group_cfg.update(dict(config.get("between_group_aggregator", {})))
        self.within_group_aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=within_group_cfg)
        self.between_group_aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=between_group_cfg)

    def _compose_group_ids(
        self,
        scores: torch.Tensor,
        path_metadata: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if path_metadata is None or not self.group_by:
            return torch.zeros_like(scores, dtype=torch.long)
        group_ids = torch.zeros_like(scores, dtype=torch.long)
        if "schema" in self.group_by:
            group_ids = group_ids * 4099 + path_metadata.get(
                "schema_bucket_ids",
                torch.zeros_like(scores, dtype=torch.long),
            ).clamp(min=0, max=4098)
        if "hop" in self.group_by:
            group_ids = group_ids * 17 + path_metadata.get(
                "hop_counts",
                torch.zeros_like(scores, dtype=torch.long),
            ).clamp(min=0, max=16)
        if "source" in self.group_by:
            group_ids = group_ids * 11 + path_metadata.get(
                "path_source_ids",
                torch.zeros_like(scores, dtype=torch.long),
            ).clamp(min=0, max=10)
        return group_ids

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        *,
        bag_mask: torch.Tensor | None = None,
        selector_scores: torch.Tensor | None = None,
        path_confidence: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return PathBagAggregator(path_reprs.size(-1), {"type": "topk_logsumexp", "top_k": 4, "attention_temperature": 1.0, "selector_temperature": 1.0})(
                pair_repr,
                path_scores,
                path_reprs,
                bag_mask=bag_mask,
                selector_scores=selector_scores,
                path_confidence=path_confidence,
            )

        if bag_mask is None:
            bag_mask = torch.ones_like(path_scores, dtype=torch.bool)
        batch_size, num_paths = path_scores.shape
        group_ids = self._compose_group_ids(path_scores, path_metadata)
        agg_scores: list[torch.Tensor] = []
        agg_reprs: list[torch.Tensor] = []
        agg_attn: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            row_mask = bag_mask[batch_idx].bool()
            if not row_mask.any():
                agg_scores.append(path_scores.new_zeros(()))
                agg_reprs.append(path_reprs.new_zeros(path_reprs.size(-1)))
                agg_attn.append(path_scores.new_zeros(num_paths))
                continue

            row_group_ids = torch.unique(group_ids[batch_idx][row_mask])
            if row_group_ids.numel() > self.max_groups:
                ranking_base = selector_scores if selector_scores is not None else path_scores
                group_rank_values: list[tuple[float, int]] = []
                for group_id in row_group_ids.tolist():
                    group_mask = row_mask & (group_ids[batch_idx] == group_id)
                    group_rank_values.append(
                        (float(ranking_base[batch_idx][group_mask].max().detach().cpu().item()), int(group_id))
                    )
                group_rank_values.sort(key=lambda item: item[0], reverse=True)
                kept_ids = [group_id for _, group_id in group_rank_values[: self.max_groups]]
                row_group_ids = torch.tensor(kept_ids, device=group_ids.device, dtype=group_ids.dtype)

            group_scores: list[torch.Tensor] = []
            group_reprs: list[torch.Tensor] = []
            group_selector: list[torch.Tensor] = []
            group_confidence: list[torch.Tensor] = []
            group_path_attn: list[torch.Tensor] = []
            for group_id in row_group_ids.tolist():
                group_mask = row_mask & (group_ids[batch_idx] == group_id)
                group_score, group_attn, group_repr = self.within_group_aggregator(
                    pair_repr[batch_idx : batch_idx + 1],
                    path_scores[batch_idx : batch_idx + 1],
                    path_reprs[batch_idx : batch_idx + 1],
                    bag_mask=group_mask.unsqueeze(0),
                    selector_scores=selector_scores[batch_idx : batch_idx + 1] if selector_scores is not None else None,
                    path_confidence=path_confidence[batch_idx : batch_idx + 1] if path_confidence is not None else None,
                )
                group_scores.append(group_score.squeeze(0))
                group_reprs.append(group_repr.squeeze(0))
                if selector_scores is not None:
                    group_selector.append(selector_scores[batch_idx][group_mask].max())
                else:
                    group_selector.append(group_score.squeeze(0))
                if path_confidence is not None:
                    group_confidence.append(path_confidence[batch_idx][group_mask].mean())
                else:
                    group_confidence.append(path_scores.new_tensor(1.0))
                group_path_attn.append(group_attn.squeeze(0))

            group_scores_tensor = torch.stack(group_scores, dim=0).unsqueeze(0)
            group_reprs_tensor = torch.stack(group_reprs, dim=0).unsqueeze(0)
            group_selector_tensor = torch.stack(group_selector, dim=0).unsqueeze(0)
            group_confidence_tensor = torch.stack(group_confidence, dim=0).unsqueeze(0)
            group_mask_tensor = torch.ones_like(group_scores_tensor, dtype=torch.bool)
            bag_score, group_weights, bag_repr = self.between_group_aggregator(
                pair_repr[batch_idx : batch_idx + 1],
                group_scores_tensor,
                group_reprs_tensor,
                bag_mask=group_mask_tensor,
                selector_scores=group_selector_tensor,
                path_confidence=group_confidence_tensor,
            )
            row_attn = path_scores.new_zeros(num_paths)
            for group_idx, group_attn in enumerate(group_path_attn):
                row_attn = row_attn + group_weights[0, group_idx] * group_attn
            agg_scores.append(bag_score.squeeze(0))
            agg_reprs.append(bag_repr.squeeze(0))
            agg_attn.append(row_attn)
        return torch.stack(agg_scores, dim=0), torch.stack(agg_attn, dim=0), torch.stack(agg_reprs, dim=0)


class CrossViewBagAttention(nn.Module):
    """Refine dual-aggregation bag states through controlled cross-view attention."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.use_in_reliability = bool(config.get("use_in_reliability", True))
        self.summary_dim = 6 if self.enabled else 0
        if not self.enabled:
            self.use_score_gating = False
            self.score_temperature = 1.0
            self.delta_scale = 0.0
            self.score_delta_scale = 0.0
            self.evidence_attn = None
            self.explanation_attn = None
            self.evidence_gate = None
            self.explanation_gate = None
            self.evidence_update = None
            self.explanation_update = None
            self.evidence_norm = None
            self.explanation_norm = None
            self.evidence_score_head = None
            self.explanation_score_head = None
            return

        dropout = float(config.get("dropout", 0.1))
        hidden = int(config.get("hidden_dim", hidden_dim))
        num_heads = int(config.get("num_heads", 4))
        self.use_score_gating = bool(config.get("use_score_gating", True))
        self.score_temperature = float(config.get("score_temperature", 1.0))
        self.delta_scale = float(config.get("delta_scale", 0.25))
        self.score_delta_scale = float(config.get("score_delta_scale", 0.15))

        self.evidence_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.explanation_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.evidence_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.explanation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.evidence_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.explanation_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.explanation_norm = nn.LayerNorm(hidden_dim)
        self.evidence_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.explanation_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _prepare_values(
        self,
        path_reprs: torch.Tensor,
        path_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_score_gating or path_scores is None:
            return path_reprs
        score_scale = torch.sigmoid(path_scores / max(self.score_temperature, 1e-6)).unsqueeze(-1)
        return path_reprs * score_scale

    def _safe_attention(
        self,
        attn: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
        bag_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bag_mask is None:
            key_padding_mask = None
            empty_rows = None
        else:
            safe_mask = bag_mask.bool().clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            key_padding_mask = ~safe_mask
        context, attn_weights = attn(
            query=query.unsqueeze(1),
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        context = context.squeeze(1)
        attn_weights = attn_weights.squeeze(1)
        if bag_mask is not None and empty_rows is not None and empty_rows.any():
            context = context.masked_fill(empty_rows.unsqueeze(-1), 0.0)
            attn_weights = attn_weights.masked_fill(empty_rows.unsqueeze(-1), 0.0)
        return context, attn_weights

    @staticmethod
    def _attention_entropy(attn_weights: torch.Tensor, bag_mask: torch.Tensor | None) -> torch.Tensor:
        if bag_mask is None:
            valid_counts = torch.full(
                (attn_weights.size(0),),
                float(attn_weights.size(1)),
                device=attn_weights.device,
            )
            safe_weights = attn_weights.clamp(min=1e-6)
        else:
            valid_counts = bag_mask.sum(dim=1).clamp(min=1).float()
            safe_weights = torch.where(bag_mask.bool(), attn_weights.clamp(min=1e-6), torch.ones_like(attn_weights))
        entropy = -(attn_weights.clamp(min=1e-6) * torch.log(safe_weights)).sum(dim=1)
        return entropy / torch.log(valid_counts + 1.0)

    def forward(
        self,
        *,
        evidence_bag_repr: torch.Tensor,
        evidence_bag_score: torch.Tensor,
        explanation_bag_repr: torch.Tensor,
        explanation_bag_score: torch.Tensor,
        evidence_path_reprs: torch.Tensor,
        evidence_path_scores: torch.Tensor | None,
        explanation_path_reprs: torch.Tensor,
        explanation_path_scores: torch.Tensor | None,
        bag_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        if not self.enabled:
            return {
                "evidence_bag_repr": evidence_bag_repr,
                "evidence_bag_score": evidence_bag_score,
                "explanation_bag_repr": explanation_bag_repr,
                "explanation_bag_score": explanation_bag_score,
                "summary": None,
                "evidence_attention": None,
                "explanation_attention": None,
            }

        explanation_values = self._prepare_values(explanation_path_reprs, explanation_path_scores)
        evidence_values = self._prepare_values(evidence_path_reprs, evidence_path_scores)

        explanation_context, explanation_attn = self._safe_attention(
            self.evidence_attn,
            evidence_bag_repr,
            explanation_values,
            bag_mask,
        )
        evidence_context, evidence_attn = self._safe_attention(
            self.explanation_attn,
            explanation_bag_repr,
            evidence_values,
            bag_mask,
        )

        evidence_features = torch.cat([evidence_bag_repr, explanation_context, explanation_bag_repr], dim=-1)
        explanation_features = torch.cat([explanation_bag_repr, evidence_context, evidence_bag_repr], dim=-1)
        evidence_gate = self.delta_scale * torch.sigmoid(self.evidence_gate(evidence_features)).squeeze(-1)
        explanation_gate = self.delta_scale * torch.sigmoid(self.explanation_gate(explanation_features)).squeeze(-1)

        refined_evidence_repr = self.evidence_norm(
            evidence_bag_repr + evidence_gate.unsqueeze(-1) * self.evidence_update(evidence_features)
        )
        refined_explanation_repr = self.explanation_norm(
            explanation_bag_repr + explanation_gate.unsqueeze(-1) * self.explanation_update(explanation_features)
        )
        evidence_score_delta = self.score_delta_scale * evidence_gate * self.evidence_score_head(
            torch.cat([refined_evidence_repr, explanation_context], dim=-1)
        ).squeeze(-1)
        explanation_score_delta = self.score_delta_scale * explanation_gate * self.explanation_score_head(
            torch.cat([refined_explanation_repr, evidence_context], dim=-1)
        ).squeeze(-1)

        evidence_entropy = self._attention_entropy(explanation_attn, bag_mask).clamp(min=0.0, max=1.0)
        explanation_entropy = self._attention_entropy(evidence_attn, bag_mask).clamp(min=0.0, max=1.0)
        repr_agreement = torch.nn.functional.cosine_similarity(
            refined_evidence_repr,
            refined_explanation_repr,
            dim=-1,
            eps=1e-6,
        )
        repr_agreement = ((repr_agreement + 1.0) * 0.5).clamp(min=0.0, max=1.0)
        score_agreement = 1.0 - (
            torch.sigmoid(evidence_bag_score + evidence_score_delta)
            - torch.sigmoid(explanation_bag_score + explanation_score_delta)
        ).abs().clamp(min=0.0, max=1.0)
        summary = torch.stack(
            [
                evidence_gate,
                explanation_gate,
                1.0 - evidence_entropy,
                1.0 - explanation_entropy,
                repr_agreement,
                score_agreement,
            ],
            dim=-1,
        )
        return {
            "evidence_bag_repr": refined_evidence_repr,
            "evidence_bag_score": evidence_bag_score + evidence_score_delta,
            "explanation_bag_repr": refined_explanation_repr,
            "explanation_bag_score": explanation_bag_score + explanation_score_delta,
            "summary": summary,
            "evidence_attention": explanation_attn,
            "explanation_attention": evidence_attn,
        }


class BagPathInteraction(nn.Module):
    """Refine path representations inside a bag before aggregation."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.explanation_only = bool(config.get("explanation_only", False))
        if not self.enabled:
            self.layers = nn.ModuleList()
            self.pair_proj = None
            self.score_head = None
            self.score_delta_scale = 0.0
            return
        num_layers = int(config.get("num_layers", 1))
        num_heads = int(config.get("num_heads", 4))
        ff_hidden_dim = int(config.get("ff_hidden_dim", hidden_dim))
        dropout = float(config.get("dropout", 0.1))
        self.score_delta_scale = float(config.get("score_delta_scale", 0.2))
        self.pair_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ff": nn.Sequential(
                            nn.Linear(hidden_dim, ff_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ff_hidden_dim, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )
        self.score_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_reprs: torch.Tensor,
        path_scores: torch.Tensor,
        *,
        bag_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return path_reprs, path_scores
        if bag_mask is None:
            bag_mask = torch.ones_like(path_scores, dtype=torch.bool)
        x = path_reprs + self.pair_proj(pair_repr).unsqueeze(1)
        key_padding_mask = ~bag_mask.bool()
        for layer in self.layers:
            attn_out, _ = layer["attn"](
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
        score_delta = self.score_head(torch.cat([path_reprs, x], dim=-1)).squeeze(-1) * self.score_delta_scale
        score_delta = score_delta.masked_fill(~bag_mask.bool(), 0.0)
        updated_scores = path_scores + score_delta
        updated_reprs = x.masked_fill(~bag_mask.unsqueeze(-1), 0.0)
        return updated_reprs, updated_scores


class ValidityGraphSidecar(nn.Module):
    """Build bag-internal path consistency features for validity calibration."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.use_in_reliability = bool(config.get("use_in_reliability", False))
        self.use_in_calibration = bool(config.get("use_in_calibration", True))
        self.summary_dim = int(config.get("summary_dim", 5))
        if not self.enabled:
            self.score_delta_scale = 0.0
            self.similarity_temperature = 1.0
            self.schema_boost = 0.0
            self.hop_boost = 0.0
            self.source_boost = 0.0
            self.repr_norm = None
            self.score_head = None
            return
        hidden = int(config.get("hidden_dim", hidden_dim))
        dropout = float(config.get("dropout", 0.1))
        self.score_delta_scale = float(config.get("score_delta_scale", 0.2))
        self.similarity_temperature = float(config.get("similarity_temperature", 0.7))
        self.schema_boost = float(config.get("schema_boost", 0.2))
        self.hop_boost = float(config.get("hop_boost", 0.1))
        self.source_boost = float(config.get("source_boost", 0.08))
        self.repr_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _metadata_boost(
        self,
        path_scores: torch.Tensor,
        path_metadata: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        boost = torch.zeros(
            path_scores.size(0),
            path_scores.size(1),
            path_scores.size(1),
            device=path_scores.device,
            dtype=path_scores.dtype,
        )
        if path_metadata is None:
            return boost
        if "schema_bucket_ids" in path_metadata and self.schema_boost > 0.0:
            schema_ids = path_metadata["schema_bucket_ids"]
            boost = boost + self.schema_boost * (schema_ids.unsqueeze(1) == schema_ids.unsqueeze(2)).float()
        if "hop_counts" in path_metadata and self.hop_boost > 0.0:
            hop_ids = path_metadata["hop_counts"]
            boost = boost + self.hop_boost * (hop_ids.unsqueeze(1) == hop_ids.unsqueeze(2)).float()
        if "path_source_ids" in path_metadata and self.source_boost > 0.0:
            source_ids = path_metadata["path_source_ids"]
            boost = boost + self.source_boost * (source_ids.unsqueeze(1) == source_ids.unsqueeze(2)).float()
        return boost

    def forward(
        self,
        *,
        path_reprs: torch.Tensor,
        base_scores: torch.Tensor,
        bag_mask: torch.Tensor,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self.enabled:
            return {
                "graph_scores": base_scores,
                "graph_features": None,
                "summary": None,
                "attention": None,
            }

        safe_mask = bag_mask.bool().clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        norm_reprs = torch.nn.functional.normalize(self.repr_norm(path_reprs), dim=-1, eps=1e-6)
        similarity = torch.bmm(norm_reprs, norm_reprs.transpose(1, 2))
        similarity = similarity + self._metadata_boost(base_scores, path_metadata)
        similarity = similarity / max(self.similarity_temperature, 1e-6)

        pair_mask = safe_mask.unsqueeze(1) & safe_mask.unsqueeze(2)
        similarity = similarity.masked_fill(~pair_mask, MASK_NEG_LARGE)
        eye_mask = torch.eye(similarity.size(1), device=similarity.device, dtype=torch.bool).unsqueeze(0)
        similarity = similarity.masked_fill(eye_mask, 0.0)

        attn = torch.softmax(similarity, dim=-1)
        attn = attn * pair_mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        neighbor_repr = torch.bmm(attn, path_reprs)
        base_probs = torch.sigmoid(base_scores)
        support_prob = torch.bmm(attn, base_probs.unsqueeze(-1)).squeeze(-1)
        pairwise_diff = (base_probs.unsqueeze(2) - base_probs.unsqueeze(1)).abs()
        disagreement = (attn * pairwise_diff).sum(dim=-1)
        consistency = torch.nn.functional.cosine_similarity(
            path_reprs,
            neighbor_repr,
            dim=-1,
            eps=1e-6,
        )
        consistency = ((consistency + 1.0) * 0.5).clamp(min=0.0, max=1.0)
        support_agreement = 1.0 - (support_prob - base_probs).abs().clamp(min=0.0, max=1.0)

        retrieval_confidence = torch.zeros_like(base_scores)
        if path_metadata is not None and "confidence" in path_metadata:
            retrieval_confidence = torch.sigmoid(path_metadata["confidence"])

        delta_inputs = torch.cat(
            [
                path_reprs,
                neighbor_repr,
                consistency.unsqueeze(-1),
                support_prob.unsqueeze(-1),
                support_agreement.unsqueeze(-1),
                retrieval_confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        score_delta = self.score_head(delta_inputs).squeeze(-1) * self.score_delta_scale
        score_delta = score_delta.masked_fill(~bag_mask.bool(), 0.0)
        graph_scores = base_scores + score_delta
        graph_scores = graph_scores.masked_fill(~bag_mask.bool(), 0.0)

        graph_features = torch.stack(
            [
                graph_scores,
                consistency,
                support_prob,
                support_agreement,
            ],
            dim=-1,
        ).masked_fill(~bag_mask.unsqueeze(-1), 0.0)

        valid_counts = bag_mask.float().sum(dim=1).clamp(min=1.0)
        summary = torch.stack(
            [
                (consistency * bag_mask.float()).sum(dim=1) / valid_counts,
                (support_prob * bag_mask.float()).sum(dim=1) / valid_counts,
                (support_agreement * bag_mask.float()).sum(dim=1) / valid_counts,
                1.0 - (disagreement * bag_mask.float()).sum(dim=1) / valid_counts,
                bag_mask.float().mean(dim=1),
            ],
            dim=-1,
        )
        summary = summary.masked_fill(empty_rows.unsqueeze(-1), 0.0)
        attn = attn.masked_fill(~pair_mask, 0.0)
        return {
            "graph_scores": graph_scores,
            "graph_features": graph_features,
            "summary": summary,
            "attention": attn,
        }


class NoGoldEvidenceExpert(nn.Module):
    """Use a dedicated evidence expert for positive-no-path bags."""

    def __init__(self, hidden_dim: int, base_config: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        if not self.enabled:
            self.aggregator = None
            self.delta_scale = 0.0
            self.residual_mix_alpha = 0.0
            self.size_threshold = 0
            self.confidence_threshold = 1.0
            self.learned_mix_gate = False
            self.mix_gate = None
            return
        hidden = int(config.get("hidden_dim", hidden_dim))
        dropout = float(config.get("dropout", 0.1))
        self.delta_scale = float(config.get("delta_scale", 0.35))
        self.residual_mix_alpha = float(config.get("residual_mix_alpha", 0.5))
        self.size_threshold = int(config.get("size_threshold", 6))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.6))
        self.learned_mix_gate = bool(config.get("learned_mix_gate", False))
        agg_cfg = dict(base_config)
        agg_cfg.update(dict(config.get("aggregator", {})))
        self.aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=agg_cfg)
        self.score_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.repr_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden_dim),
        )
        self.repr_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        if self.learned_mix_gate:
            gate_hidden = int(config.get("gate_hidden_dim", hidden))
            self.mix_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 3, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 1),
            )
        else:
            self.mix_gate = None

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        *,
        bag_mask: torch.Tensor,
        selector_scores: torch.Tensor | None = None,
        path_confidence: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        pair_expanded = pair_repr.unsqueeze(1).expand_as(path_reprs)
        if path_metadata is None:
            retrieval_confidence = torch.zeros_like(path_scores)
            retrieved_flag = torch.zeros_like(path_scores)
        else:
            retrieval_confidence = torch.sigmoid(path_metadata.get("confidence", torch.zeros_like(path_scores)))
            retrieved_flag = path_metadata.get("is_retrieved", torch.zeros_like(path_scores, dtype=torch.bool)).float()
        delta_inputs = torch.cat(
            [
                pair_expanded,
                path_reprs,
                retrieval_confidence.unsqueeze(-1),
                retrieved_flag.unsqueeze(-1),
            ],
            dim=-1,
        )
        score_delta = self.score_adapter(delta_inputs).squeeze(-1)
        expert_scores = path_scores + self.delta_scale * score_delta
        gate_inputs = torch.cat([pair_expanded, path_reprs], dim=-1)
        repr_delta = self.repr_update(gate_inputs)
        repr_gate = self.repr_gate(gate_inputs)
        expert_reprs = path_reprs + repr_gate * repr_delta
        expert_selector_scores = selector_scores + self.delta_scale * score_delta if selector_scores is not None else expert_scores
        expert_score, expert_attn, expert_repr = self.aggregator(
            pair_repr,
            expert_scores,
            expert_reprs,
            bag_mask=bag_mask,
            selector_scores=expert_selector_scores,
            path_confidence=path_confidence,
        )
        return {
            "path_score": expert_score,
            "path_repr": expert_repr,
            "path_attention": expert_attn,
            "expert_scores": expert_scores,
        }

    def compute_mix_weight(
        self,
        *,
        bag_mask: torch.Tensor,
        path_confidence: torch.Tensor | None,
        pair_repr: torch.Tensor | None = None,
        expert_repr: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return bag_mask.new_zeros((bag_mask.size(0),), dtype=torch.float)
        bag_sizes = bag_mask.float().sum(dim=1)
        if path_confidence is not None:
            conf_sums = (path_confidence * bag_mask.float()).sum(dim=1)
            mean_conf = conf_sums / bag_sizes.clamp(min=1.0)
        else:
            mean_conf = bag_mask.float().new_full((bag_mask.size(0),), self.confidence_threshold)
        if path_metadata is not None and "is_retrieved" in path_metadata:
            retrieved_ratio = (
                path_metadata["is_retrieved"].float() * bag_mask.float()
            ).sum(dim=1) / bag_sizes.clamp(min=1.0)
        else:
            retrieved_ratio = bag_mask.float().new_zeros((bag_mask.size(0),))
        if self.learned_mix_gate and self.mix_gate is not None and pair_repr is not None and expert_repr is not None:
            gate_inputs = torch.cat(
                [
                    pair_repr,
                    expert_repr,
                    (bag_sizes / max(1.0, float(self.size_threshold))).unsqueeze(-1),
                    mean_conf.unsqueeze(-1),
                    retrieved_ratio.unsqueeze(-1),
                ],
                dim=-1,
            )
            learned_gate = torch.sigmoid(self.mix_gate(gate_inputs)).squeeze(-1)
            return self.residual_mix_alpha * learned_gate
        size_factor = ((bag_sizes - float(self.size_threshold)) / max(1.0, float(self.size_threshold))).clamp(min=0.0, max=1.0)
        confidence_factor = ((self.confidence_threshold - mean_conf) / max(1e-6, self.confidence_threshold)).clamp(min=0.0, max=1.0)
        mix = self.residual_mix_alpha * size_factor * confidence_factor
        return mix


class ResidualDualEvidenceAggregator(nn.Module):
    """Blend shared evidence with gold-aware and latent no-gold experts."""

    def __init__(self, hidden_dim: int, base_config: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        if not self.enabled:
            self.gold_expert = None
            self.latent_expert = None
            self.route_gate = None
            self.route_prior_strength = 0.0
            return
        hidden = int(config.get("routing_hidden_dim", hidden_dim))
        dropout = float(config.get("dropout", 0.1))
        self.route_prior_strength = float(config.get("route_prior_strength", 1.5))
        gold_cfg = dict(config.get("gold_expert", {}))
        latent_cfg = dict(config.get("latent_expert", {}))
        gold_cfg["enabled"] = True
        latent_cfg["enabled"] = True
        self.gold_expert = NoGoldEvidenceExpert(hidden_dim=hidden_dim, base_config=base_config, config=gold_cfg)
        self.latent_expert = NoGoldEvidenceExpert(hidden_dim=hidden_dim, base_config=base_config, config=latent_cfg)
        self.route_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    @staticmethod
    def _bag_stats(
        bag_mask: torch.Tensor,
        path_confidence: torch.Tensor | None,
        path_metadata: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bag_sizes = bag_mask.float().sum(dim=1)
        if path_confidence is not None:
            conf_sums = (path_confidence * bag_mask.float()).sum(dim=1)
            mean_conf = conf_sums / bag_sizes.clamp(min=1.0)
        else:
            mean_conf = bag_mask.float().new_zeros((bag_mask.size(0),))
        if path_metadata is not None and "is_retrieved" in path_metadata:
            retrieved_ratio = (
                path_metadata["is_retrieved"].float() * bag_mask.float()
            ).sum(dim=1) / bag_sizes.clamp(min=1.0)
        else:
            retrieved_ratio = bag_mask.float().new_zeros((bag_mask.size(0),))
        return bag_sizes, mean_conf, retrieved_ratio

    def forward(
        self,
        *,
        pair_repr: torch.Tensor,
        base_score: torch.Tensor,
        base_repr: torch.Tensor,
        base_attn: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        bag_mask: torch.Tensor,
        has_gold_rationale: torch.Tensor | None,
        selector_scores: torch.Tensor | None = None,
        path_confidence: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        gold_outputs = self.gold_expert(
            pair_repr=pair_repr,
            path_scores=path_scores,
            path_reprs=path_reprs,
            bag_mask=bag_mask,
            selector_scores=selector_scores,
            path_confidence=path_confidence,
            path_metadata=path_metadata,
        )
        latent_outputs = self.latent_expert(
            pair_repr=pair_repr,
            path_scores=path_scores,
            path_reprs=path_reprs,
            bag_mask=bag_mask,
            selector_scores=selector_scores,
            path_confidence=path_confidence,
            path_metadata=path_metadata,
        )
        gold_strength = self.gold_expert.compute_mix_weight(
            bag_mask=bag_mask,
            path_confidence=path_confidence,
            pair_repr=pair_repr,
            expert_repr=gold_outputs["path_repr"],
            path_metadata=path_metadata,
        )
        latent_strength = self.latent_expert.compute_mix_weight(
            bag_mask=bag_mask,
            path_confidence=path_confidence,
            pair_repr=pair_repr,
            expert_repr=latent_outputs["path_repr"],
            path_metadata=path_metadata,
        )
        bag_sizes, mean_conf, retrieved_ratio = self._bag_stats(
            bag_mask=bag_mask,
            path_confidence=path_confidence,
            path_metadata=path_metadata,
        )
        if has_gold_rationale is None:
            has_gold = base_score.new_zeros(base_score.size(0))
        else:
            has_gold = has_gold_rationale.to(device=base_score.device, dtype=base_score.dtype)
        route_inputs = torch.cat(
            [
                pair_repr,
                base_repr,
                has_gold.unsqueeze(-1),
                (bag_sizes / bag_sizes.clamp(min=1.0).max().clamp(min=1.0)).unsqueeze(-1),
                mean_conf.unsqueeze(-1),
                retrieved_ratio.unsqueeze(-1),
            ],
            dim=-1,
        )
        route_logits = self.route_gate(route_inputs)
        route_bias = torch.stack(
            [
                has_gold * self.route_prior_strength,
                (1.0 - has_gold) * self.route_prior_strength,
            ],
            dim=-1,
        )
        route_weights = torch.softmax(route_logits + route_bias, dim=-1)
        gold_mix = route_weights[:, 0] * gold_strength
        latent_mix = route_weights[:, 1] * latent_strength
        combined_score = base_score
        combined_repr = base_repr
        combined_attn = base_attn
        combined_score = combined_score + gold_mix * (gold_outputs["path_score"] - base_score)
        combined_score = combined_score + latent_mix * (latent_outputs["path_score"] - base_score)
        combined_repr = combined_repr + gold_mix.unsqueeze(-1) * (gold_outputs["path_repr"] - base_repr)
        combined_repr = combined_repr + latent_mix.unsqueeze(-1) * (latent_outputs["path_repr"] - base_repr)
        combined_attn = combined_attn + gold_mix.unsqueeze(-1) * (gold_outputs["path_attention"] - base_attn)
        combined_attn = combined_attn + latent_mix.unsqueeze(-1) * (latent_outputs["path_attention"] - base_attn)
        return {
            "path_score": combined_score,
            "path_repr": combined_repr,
            "path_attention": combined_attn,
            "gold_mix_weight": gold_mix,
            "latent_mix_weight": latent_mix,
            "route_weights": route_weights,
            "gold_expert_scores": gold_outputs["expert_scores"],
            "latent_expert_scores": latent_outputs["expert_scores"],
        }


class LearnedPathReranker(nn.Module):
    """Learn an independent retrieval logit and shortlist over candidate paths."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.bias_scale = float(config.get("bias_scale", 0.5))
        self.shortlist_top_k = int(config.get("shortlist_top_k", 0))
        self.selection_mode = str(config.get("selection_mode", "retrieval"))
        if not self.enabled:
            self.metadata_embedding = None
            self.retrieval_encoder = None
            self.retrieval_logit_head = None
            return
        rerank_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        metadata_dim = int(config.get("metadata_dim", max(16, hidden_dim // 4)))
        dropout = float(config.get("dropout", 0.1))
        self.schema_bucket_count = int(config.get("schema_bucket_count", 2048))
        self.max_hops = int(config.get("max_hops", 8))
        self.path_source_vocab_size = int(config.get("path_source_vocab_size", 8))
        self.schema_embedding = nn.Embedding(self.schema_bucket_count + 1, metadata_dim)
        self.hop_embedding = nn.Embedding(self.max_hops + 2, metadata_dim)
        self.path_source_embedding = nn.Embedding(self.path_source_vocab_size, metadata_dim)
        input_dim = hidden_dim * 2 + 4 + metadata_dim * 3
        self.retrieval_encoder = nn.Sequential(
            nn.Linear(input_dim, rerank_hidden_dim),
            nn.LayerNorm(rerank_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.retrieval_logit_head = nn.Linear(rerank_hidden_dim, 1)

    def _metadata_features(
        self,
        evidence_scores: torch.Tensor,
        path_metadata: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if path_metadata is None:
            schema_bucket_ids = torch.zeros_like(evidence_scores, dtype=torch.long)
            hop_counts = torch.zeros_like(evidence_scores, dtype=torch.long)
            path_source_ids = torch.zeros_like(evidence_scores, dtype=torch.long)
            confidence = torch.zeros_like(evidence_scores)
            is_retrieved = torch.zeros_like(evidence_scores, dtype=torch.bool)
        else:
            schema_bucket_ids = path_metadata.get("schema_bucket_ids", torch.zeros_like(evidence_scores, dtype=torch.long))
            hop_counts = path_metadata.get("hop_counts", torch.zeros_like(evidence_scores, dtype=torch.long))
            path_source_ids = path_metadata.get("path_source_ids", torch.zeros_like(evidence_scores, dtype=torch.long))
            confidence = path_metadata.get("confidence", torch.zeros_like(evidence_scores))
            is_retrieved = path_metadata.get("is_retrieved", torch.zeros_like(evidence_scores, dtype=torch.bool))
        schema_bucket_ids = schema_bucket_ids.clamp(min=0, max=self.schema_bucket_count)
        hop_counts = hop_counts.clamp(min=0, max=self.max_hops + 1)
        path_source_ids = path_source_ids.clamp(min=0, max=self.path_source_vocab_size - 1)
        schema_features = self.schema_embedding(schema_bucket_ids)
        hop_features = self.hop_embedding(hop_counts)
        source_features = self.path_source_embedding(path_source_ids)
        return schema_features, hop_features, source_features, confidence, is_retrieved

    def forward(
        self,
        pair_repr: torch.Tensor,
        path_reprs: torch.Tensor,
        evidence_scores: torch.Tensor,
        explanation_scores: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_mask = (
            bag_mask.bool()
            if bag_mask is not None
            else torch.ones_like(evidence_scores, dtype=torch.bool)
        )
        if not self.enabled:
            return evidence_scores, torch.zeros_like(evidence_scores), base_mask, evidence_scores

        if explanation_scores is None:
            explanation_scores = evidence_scores
        schema_features, hop_features, source_features, confidence, is_retrieved = self._metadata_features(
            evidence_scores,
            path_metadata,
        )
        pair_expanded = pair_repr.unsqueeze(1).expand(-1, path_reprs.size(1), -1)
        rerank_features = torch.cat(
            [
                pair_expanded,
                path_reprs,
                evidence_scores.unsqueeze(-1),
                explanation_scores.unsqueeze(-1),
                confidence.unsqueeze(-1),
                is_retrieved.float().unsqueeze(-1),
                schema_features,
                hop_features,
                source_features,
            ],
            dim=-1,
        )
        retrieval_hidden = self.retrieval_encoder(rerank_features)
        retrieval_logits = self.retrieval_logit_head(retrieval_hidden).squeeze(-1)
        if bag_mask is not None:
            retrieval_logits = retrieval_logits.masked_fill(~bag_mask, 0.0)
        rerank_bias = retrieval_logits
        reranked_scores = evidence_scores + self.bias_scale * rerank_bias
        selection_scores = retrieval_logits if self.selection_mode == "retrieval" else reranked_scores
        shortlist_mask = base_mask
        if self.shortlist_top_k > 0:
            shortlist_mask = _masked_topk_mask(selection_scores, base_mask, self.shortlist_top_k)
        return reranked_scores, rerank_bias, shortlist_mask, retrieval_logits


class MechanismReliabilityEstimator(nn.Module):
    """Estimate how trustworthy the current path bag is for mechanistic fusion."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.top_k = int(config.get("top_k", 4))
        self.summary_dim = int(config.get("summary_dim", 8))
        if not self.enabled:
            self.feature_proj = None
            self.context_head = None
            self.reliability_head = None
            return
        reliability_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        dropout = float(config.get("dropout", 0.1))
        raw_feature_dim = 8
        self.feature_proj = nn.Sequential(
            nn.Linear(raw_feature_dim, reliability_hidden_dim),
            nn.LayerNorm(reliability_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_head = nn.Linear(reliability_hidden_dim, self.summary_dim)
        self.reliability_head = nn.Linear(reliability_hidden_dim, 1)

    def forward(
        self,
        evidence_scores: torch.Tensor,
        bag_mask: torch.Tensor,
        explanation_scores: torch.Tensor | None = None,
        binary_scores: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = evidence_scores.size(0)
        device = evidence_scores.device
        dtype = evidence_scores.dtype
        empty_rows = ~bag_mask.any(dim=1)
        if not self.enabled:
            reliability = torch.where(empty_rows, torch.zeros(batch_size, device=device, dtype=dtype), torch.ones(batch_size, device=device, dtype=dtype))
            uncertainty = 1.0 - reliability
            features = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            return {
                "reliability": reliability,
                "uncertainty": uncertainty,
                "features": features,
                "summary": None,
            }

        score_logits = binary_scores
        if score_logits is None:
            score_logits = explanation_scores if explanation_scores is not None else evidence_scores
        ref_logits = explanation_scores if explanation_scores is not None else evidence_scores
        aux_logits = binary_scores if binary_scores is not None else evidence_scores

        masked_logits = score_logits.masked_fill(~bag_mask, MASK_NEG_LARGE)
        num_paths = masked_logits.size(1)
        if num_paths == 0:
            reliability = torch.zeros(batch_size, device=device, dtype=dtype)
            uncertainty = torch.ones(batch_size, device=device, dtype=dtype)
            features = torch.zeros(batch_size, self.summary_dim + 2, device=device, dtype=dtype)
            return {
                "reliability": reliability,
                "uncertainty": uncertainty,
                "features": features,
                "summary": torch.zeros(batch_size, self.summary_dim, device=device, dtype=dtype),
            }

        k = min(max(1, self.top_k), num_paths)
        top_logits, top_idx = masked_logits.topk(k=k, dim=1)
        top_probs = torch.sigmoid(top_logits)
        top1_prob = top_probs[:, 0]
        top12_margin = top_probs[:, 0] - top_probs[:, 1] if k >= 2 else torch.zeros_like(top1_prob)
        topk_mean = top_probs.mean(dim=1)
        if k >= 2:
            topk_weights = torch.softmax(top_logits, dim=1)
            topk_entropy = -(topk_weights * torch.log(topk_weights.clamp(min=1e-8))).sum(dim=1)
            topk_entropy = topk_entropy / torch.log(torch.tensor(float(k), device=device, dtype=dtype))
        else:
            topk_entropy = torch.zeros_like(top1_prob)

        ref_probs = torch.sigmoid(ref_logits)
        aux_probs = torch.sigmoid(aux_logits)
        top_ref_probs = ref_probs.gather(1, top_idx)
        top_aux_probs = aux_probs.gather(1, top_idx)
        agreement = 1.0 - (top_ref_probs - top_aux_probs).abs().mean(dim=1).clamp(min=0.0, max=1.0)

        if path_metadata is None:
            path_metadata = {}
        confidence = path_metadata.get("confidence")
        if confidence is None:
            confidence = torch.zeros_like(evidence_scores)
        is_retrieved = path_metadata.get("is_retrieved")
        if is_retrieved is None:
            is_retrieved = torch.zeros_like(evidence_scores, dtype=torch.bool)
        top_confidence = confidence.gather(1, top_idx).mean(dim=1)
        top_retrieved_ratio = is_retrieved.float().gather(1, top_idx).mean(dim=1)
        bag_density = bag_mask.float().mean(dim=1)

        raw_features = torch.stack(
            [
                top1_prob,
                top12_margin,
                topk_mean,
                topk_entropy,
                agreement,
                top_confidence,
                top_retrieved_ratio,
                bag_density,
            ],
            dim=-1,
        )
        raw_features = raw_features.masked_fill(empty_rows.unsqueeze(-1), 0.0)
        hidden = self.feature_proj(raw_features)
        summary = torch.tanh(self.context_head(hidden))
        reliability = torch.sigmoid(self.reliability_head(hidden)).squeeze(-1)
        reliability = reliability.masked_fill(empty_rows, 0.0)
        uncertainty = 1.0 - reliability
        features = torch.cat([summary, reliability.unsqueeze(-1), uncertainty.unsqueeze(-1)], dim=-1)
        features = features.masked_fill(empty_rows.unsqueeze(-1), 0.0)
        return {
            "reliability": reliability,
            "uncertainty": uncertainty,
            "features": features,
            "summary": summary,
            "raw_features": raw_features,
        }


class PathBinaryCalibrator(nn.Module):
    """Calibrate path validity from multi-view mechanistic signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        self.pair_feedback_enabled = bool(config.get("pair_feedback_enabled", True))
        self.summary_dim = int(config.get("summary_dim", 3))
        if not self.enabled:
            self.calibrator = None
            self.delta_scale = 0.0
            self.graph_feature_dim = 0
            return
        hidden_dim = int(config.get("hidden_dim", 64))
        dropout = float(config.get("dropout", 0.1))
        self.delta_scale = float(config.get("delta_scale", 0.5))
        self.graph_feature_dim = int(config.get("graph_feature_dim", 4))
        raw_feature_dim = 8 + self.graph_feature_dim
        self.calibrator = nn.Sequential(
            nn.Linear(raw_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        evidence_scores: torch.Tensor,
        explanation_scores: torch.Tensor,
        raw_binary_scores: torch.Tensor | None,
        agreement_scores: torch.Tensor | None,
        path_confidence_scores: torch.Tensor | None,
        retrieval_logits: torch.Tensor | None,
        path_metadata: dict[str, torch.Tensor] | None,
        bag_reliability: torch.Tensor,
        bag_mask: torch.Tensor | None,
        validity_graph_scores: torch.Tensor | None = None,
        validity_graph_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if raw_binary_scores is None:
            return {
                "calibrated_scores": None,
                "summary": None,
                "raw_features": None,
            }
        if not self.enabled:
            return {
                "calibrated_scores": raw_binary_scores,
                "summary": None,
                "raw_features": None,
            }

        if bag_mask is None:
            bag_mask = torch.ones_like(raw_binary_scores, dtype=torch.bool)
        if agreement_scores is None:
            agreement_scores = 1.0 - (
                torch.sigmoid(explanation_scores) - torch.sigmoid(raw_binary_scores)
            ).abs().clamp(min=0.0, max=1.0)
        if path_confidence_scores is None:
            path_confidence_scores = torch.sigmoid(explanation_scores) * agreement_scores
        if retrieval_logits is None:
            retrieval_logits = torch.zeros_like(raw_binary_scores)
        retrieval_confidence = torch.zeros_like(raw_binary_scores)
        if path_metadata is not None and "confidence" in path_metadata:
            retrieval_confidence = torch.sigmoid(path_metadata["confidence"])

        bag_reliability_expanded = bag_reliability.unsqueeze(1).expand_as(raw_binary_scores)
        raw_features = torch.stack(
            [
                evidence_scores,
                explanation_scores,
                raw_binary_scores,
                agreement_scores,
                path_confidence_scores,
                retrieval_logits,
                retrieval_confidence,
                bag_reliability_expanded,
            ],
            dim=-1,
        )
        if self.graph_feature_dim > 0:
            if validity_graph_features is None:
                validity_graph_features = torch.zeros(
                    *raw_binary_scores.shape,
                    self.graph_feature_dim,
                    device=raw_binary_scores.device,
                    dtype=raw_binary_scores.dtype,
                )
                if validity_graph_scores is not None and self.graph_feature_dim >= 1:
                    validity_graph_features[..., 0] = validity_graph_scores
            raw_features = torch.cat([raw_features, validity_graph_features], dim=-1)
        raw_features = raw_features.masked_fill(~bag_mask.unsqueeze(-1), 0.0)
        delta = self.calibrator(raw_features).squeeze(-1)
        calibrated_scores = raw_binary_scores + self.delta_scale * delta
        calibrated_scores = calibrated_scores.masked_fill(~bag_mask, 0.0)

        calibrated_probs = torch.sigmoid(calibrated_scores)
        masked_probs = calibrated_probs.masked_fill(~bag_mask, 0.0)
        valid_counts = bag_mask.float().sum(dim=1).clamp(min=1.0)
        top1_prob = masked_probs.max(dim=1).values
        mean_prob = masked_probs.sum(dim=1) / valid_counts
        calibrated_agreement = 1.0 - (
            calibrated_probs - torch.sigmoid(explanation_scores)
        ).abs().masked_fill(~bag_mask, 0.0).sum(dim=1) / valid_counts
        summary = torch.stack([top1_prob, mean_prob, calibrated_agreement], dim=-1)
        return {
            "calibrated_scores": calibrated_scores,
            "summary": summary,
            "raw_features": raw_features,
        }


class CrossBranchInteractor(nn.Module):
    """Refine direct and mechanistic branches with explicit cross-branch gating."""

    def __init__(self, hidden_dim: int, config: dict[str, Any], reliability_dim: int = 0) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", True))
        self.alpha = float(config.get("alpha", 0.5))
        dropout = float(config.get("dropout", 0.1))
        interaction_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        self.reliability_enabled = reliability_dim > 0
        self.reliability_dim = int(reliability_dim)
        fusion_input_dim = hidden_dim * 4 + 3 + self.reliability_dim
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
        self.mech_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2 + self.reliability_dim, interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2 + self.reliability_dim, interaction_hidden_dim),
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
        reliability_features: torch.Tensor | None = None,
        mechanistic_reliability: torch.Tensor | None = None,
        mechanistic_uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mechanistic_reliability is None:
            mechanistic_reliability = torch.where(
                bag_available.squeeze(-1).bool(),
                torch.ones_like(direct_score),
                torch.zeros_like(direct_score),
            )
        if mechanistic_uncertainty is None:
            mechanistic_uncertainty = 1.0 - mechanistic_reliability
        if self.reliability_enabled:
            if reliability_features is None:
                reliability_features = torch.zeros(
                    pair_repr.size(0),
                    self.reliability_dim,
                    device=pair_repr.device,
                    dtype=pair_repr.dtype,
                )
        else:
            reliability_features = None

        if not self.enabled:
            final_score = self.alpha * direct_score + (1.0 - self.alpha) * path_score
            direct_weight = torch.where(
                bag_available.squeeze(-1).bool(),
                torch.full_like(direct_score, self.alpha),
                torch.ones_like(direct_score),
            )
            return {
                "refined_pair_repr": pair_repr,
                "refined_path_repr": path_repr,
                "refined_direct_score": direct_score,
                "refined_path_score": path_score,
                "fusion_gate": direct_weight,
                "mechanistic_gate": 1.0 - direct_weight,
                "interaction_delta": torch.zeros_like(direct_score),
                "pair_score": final_score,
                "mechanistic_reliability": mechanistic_reliability,
                "mechanistic_uncertainty": mechanistic_uncertainty,
                "reliability_features": reliability_features,
            }

        reliability_scale = (bag_available.squeeze(-1) * mechanistic_reliability).unsqueeze(-1)
        pair_path_parts = [
            pair_repr,
            path_repr,
            torch.abs(pair_repr - path_repr),
            pair_repr * path_repr,
            direct_score.unsqueeze(-1),
            path_score.unsqueeze(-1),
            bag_available,
        ]
        if reliability_features is not None:
            pair_path_parts.append(reliability_features)
        pair_path_features = torch.cat(pair_path_parts, dim=-1)
        context = self.context_mlp(pair_path_features)
        direct_gate = reliability_scale * self.direct_gate(context)
        path_gate = reliability_scale * self.path_gate(context)

        refined_pair_repr = self.direct_norm(pair_repr + direct_gate * self.direct_update(path_repr))
        refined_path_repr = self.path_norm(path_repr + path_gate * self.path_update(pair_repr))

        reliability_scalar = bag_available.squeeze(-1) * mechanistic_reliability
        refined_direct_score = direct_score + reliability_scalar * self.direct_delta_head(refined_pair_repr).squeeze(-1)
        refined_path_score = path_score + reliability_scalar * self.path_delta_head(refined_path_repr).squeeze(-1)

        fusion_parts = [
            refined_pair_repr,
            refined_path_repr,
            refined_direct_score.unsqueeze(-1),
            refined_path_score.unsqueeze(-1),
        ]
        if reliability_features is not None:
            fusion_parts.append(reliability_features)
        fusion_features = torch.cat(fusion_parts, dim=-1)
        learned_mech_gate = torch.sigmoid(self.mech_gate(fusion_features)).squeeze(-1)
        effective_mech_weight = reliability_scalar * learned_mech_gate
        direct_weight = 1.0 - effective_mech_weight
        interaction_delta = self.residual_head(fusion_features).squeeze(-1) * reliability_scalar
        final_score = (
            direct_weight * refined_direct_score
            + effective_mech_weight * refined_path_score
            + interaction_delta
        )
        return {
            "refined_pair_repr": refined_pair_repr,
            "refined_path_repr": refined_path_repr,
            "refined_direct_score": refined_direct_score,
            "refined_path_score": refined_path_score,
            "fusion_gate": direct_weight,
            "mechanistic_gate": effective_mech_weight,
            "interaction_delta": interaction_delta,
            "pair_score": final_score,
            "mechanistic_reliability": mechanistic_reliability,
            "mechanistic_uncertainty": mechanistic_uncertainty,
            "reliability_features": reliability_features,
        }


class HierarchicalPairModel(nn.Module):
    """Combine direct pair and path bag branches into the final pair score."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        pair_feature_cfg = config.get("direct_pair_features", {})
        aggregator_cfg = dict(config["aggregator"])
        dual_agg_cfg = dict(config.get("dual_aggregation", {}))
        self.agreement_aware_cfg = dict(aggregator_cfg.get("agreement_aware", {}))
        self.agreement_aware_enabled = bool(self.agreement_aware_cfg.get("enabled", False))
        self.dual_aggregation_enabled = bool(dual_agg_cfg.get("enabled", False))
        self.dual_pair_feedback_enabled = bool(dual_agg_cfg.get("pair_feedback_enabled", True))
        self.dual_use_explanation_in_reliability = bool(dual_agg_cfg.get("use_explanation_in_reliability", True))
        self.dual_evidence_only_when_no_gold = bool(dual_agg_cfg.get("evidence_only_when_no_gold", False))
        self.dual_no_gold_feedback_scale = float(dual_agg_cfg.get("no_gold_feedback_scale", 1.0))
        self.dual_agreement_v2_cfg = dict(dual_agg_cfg.get("agreement_fusion_v2", {}))
        self.dual_agreement_v2_enabled = bool(self.dual_agreement_v2_cfg.get("enabled", False))
        self.dual_agreement_feature_dim = 4 if self.dual_aggregation_enabled else 0
        cross_view_cfg = dict(dual_agg_cfg.get("cross_view_attention", {}))
        self.cross_view_attention_enabled = bool(cross_view_cfg.get("enabled", False))
        self.cross_view_feature_dim = 6 if self.cross_view_attention_enabled else 0
        hierarchical_cfg = dict(dual_agg_cfg.get("hierarchical", {}))
        evidence_hierarchical_cfg = dict(hierarchical_cfg)
        evidence_hierarchical_cfg.update(dict(dual_agg_cfg.get("evidence_hierarchical", {})))
        explanation_hierarchical_cfg = dict(hierarchical_cfg)
        explanation_hierarchical_cfg.update(dict(dual_agg_cfg.get("explanation_hierarchical", {})))
        self.evidence_hierarchical_enabled = bool(evidence_hierarchical_cfg.get("enabled", False))
        self.explanation_hierarchical_enabled = bool(explanation_hierarchical_cfg.get("enabled", False))
        self.hierarchical_dualagg_enabled = self.evidence_hierarchical_enabled or self.explanation_hierarchical_enabled
        validity_agg_cfg = dict(dual_agg_cfg.get("validity_aggregator", {}))
        self.validity_aggregation_enabled = bool(validity_agg_cfg.get("enabled", False))
        self.validity_pair_feedback_enabled = bool(validity_agg_cfg.get("pair_feedback_enabled", True))
        self.validity_use_in_reliability = bool(validity_agg_cfg.get("use_in_reliability", True))
        self.validity_pair_mix_alpha = float(validity_agg_cfg.get("pair_mix_alpha", 0.08))
        self.validity_feature_dim = 4 if self.validity_aggregation_enabled else 0
        validity_graph_cfg = dict(dual_agg_cfg.get("validity_graph_sidecar", {}))
        self.validity_graph_sidecar_enabled = bool(validity_graph_cfg.get("enabled", False))
        self.validity_graph_use_in_reliability = bool(validity_graph_cfg.get("use_in_reliability", False))
        self.validity_graph_feature_dim = int(validity_graph_cfg.get("summary_dim", 5)) if self.validity_graph_sidecar_enabled else 0
        path_interaction_cfg = dict(dual_agg_cfg.get("path_interaction", {}))
        self.path_interaction_enabled = bool(path_interaction_cfg.get("enabled", False))
        no_gold_expert_cfg = dict(dual_agg_cfg.get("no_gold_evidence_expert", {}))
        self.no_gold_expert_enabled = bool(no_gold_expert_cfg.get("enabled", False))
        self.no_gold_pair_feedback_enabled = bool(no_gold_expert_cfg.get("pair_feedback_enabled", True))
        residual_dual_expert_cfg = dict(dual_agg_cfg.get("residual_dual_evidence_expert", {}))
        self.residual_dual_expert_enabled = bool(residual_dual_expert_cfg.get("enabled", False))
        self.dual_pair_mix_alpha = float(dual_agg_cfg.get("pair_mix_alpha", 0.2))
        self.direct_pair = DirectPairEncoder(
            hidden_dim=hidden_dim,
            dropout=config["path_scorer"]["dropout"],
            pair_feature_dim=int(pair_feature_cfg.get("feature_dim", 0)) if pair_feature_cfg.get("enabled", False) else 0,
            pair_feature_hidden_dim=pair_feature_cfg.get("hidden_dim"),
        )
        self.aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=aggregator_cfg)
        self.explanation_aggregator = None
        if self.dual_aggregation_enabled:
            explanation_aggregator_cfg = dict(aggregator_cfg)
            explanation_aggregator_cfg.update(dict(dual_agg_cfg.get("explanation_aggregator", {})))
            self.explanation_aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=explanation_aggregator_cfg)
        self.hierarchical_aggregator = HierarchicalPathBagAggregator(
            hidden_dim=hidden_dim,
            base_config=aggregator_cfg,
            config=evidence_hierarchical_cfg,
        )
        self.hierarchical_explanation_aggregator = None
        if self.dual_aggregation_enabled:
            explanation_aggregator_cfg = dict(aggregator_cfg)
            explanation_aggregator_cfg.update(dict(dual_agg_cfg.get("explanation_aggregator", {})))
            self.hierarchical_explanation_aggregator = HierarchicalPathBagAggregator(
                hidden_dim=hidden_dim,
                base_config=explanation_aggregator_cfg,
                config=explanation_hierarchical_cfg,
            )
        self.validity_aggregator = None
        if self.validity_aggregation_enabled:
            validity_base_cfg = dict(aggregator_cfg)
            validity_base_cfg.update(validity_agg_cfg)
            self.validity_aggregator = PathBagAggregator(hidden_dim=hidden_dim, config=validity_base_cfg)
        self.validity_graph_sidecar = ValidityGraphSidecar(hidden_dim=hidden_dim, config=validity_graph_cfg)
        self.path_interaction = BagPathInteraction(hidden_dim=hidden_dim, config=path_interaction_cfg)
        self.no_gold_evidence_expert = NoGoldEvidenceExpert(
            hidden_dim=hidden_dim,
            base_config=aggregator_cfg,
            config=no_gold_expert_cfg,
        )
        self.residual_dual_evidence_aggregator = ResidualDualEvidenceAggregator(
            hidden_dim=hidden_dim,
            base_config=aggregator_cfg,
            config=residual_dual_expert_cfg,
        )
        self.cross_view_attention = CrossViewBagAttention(hidden_dim=hidden_dim, config=cross_view_cfg)
        self.retrieval_reranker = LearnedPathReranker(hidden_dim=hidden_dim, config=config.get("learned_retrieval", {}))
        learned_retrieval_cfg = dict(config.get("learned_retrieval", {}))
        self.explanation_shortlist_top_k = int(learned_retrieval_cfg.get("explanation_shortlist_top_k", 0))
        self.binary_shortlist_top_k = int(learned_retrieval_cfg.get("binary_shortlist_top_k", 0))
        interaction_cfg = dict(config.get("interaction", {}))
        interaction_cfg.setdefault("alpha", config["aggregator"]["alpha"])
        interaction_cfg.setdefault("dropout", config["path_scorer"]["dropout"])
        uncertainty_cfg = dict(interaction_cfg.get("uncertainty_fusion", {}))
        binary_calibration_cfg = dict(config.get("binary_calibration", {}))
        self.reliability_estimator = MechanismReliabilityEstimator(hidden_dim=hidden_dim, config=uncertainty_cfg)
        self.binary_calibrator = PathBinaryCalibrator(config=binary_calibration_cfg)
        reliability_dim = self.reliability_estimator.summary_dim + 2 if self.reliability_estimator.enabled else 0
        reliability_dim += self.dual_agreement_feature_dim if self.dual_pair_feedback_enabled else 0
        reliability_dim += self.cross_view_feature_dim if self.cross_view_attention.use_in_reliability else 0
        reliability_dim += self.validity_feature_dim if self.validity_use_in_reliability else 0
        reliability_dim += self.validity_graph_feature_dim if self.validity_graph_use_in_reliability else 0
        reliability_dim += (
            self.binary_calibrator.summary_dim
            if self.binary_calibrator.enabled and self.binary_calibrator.pair_feedback_enabled
            else 0
        )
        self.interactor = CrossBranchInteractor(hidden_dim=hidden_dim, config=interaction_cfg, reliability_dim=reliability_dim)

    def _apply_dual_shortlist(
        self,
        shortlist_mask: torch.Tensor,
        bag_mask: torch.Tensor | None,
        explanation_scores: torch.Tensor | None,
        binary_scores: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if bag_mask is not None:
            base_mask = bag_mask.bool()
        else:
            base_mask = shortlist_mask.bool()
        combined_mask = shortlist_mask.bool() & base_mask

        explanation_shortlist_mask = None
        if self.explanation_shortlist_top_k > 0 and explanation_scores is not None:
            explanation_shortlist_mask = _masked_topk_mask(
                explanation_scores,
                base_mask,
                self.explanation_shortlist_top_k,
            )
            combined_mask = combined_mask | explanation_shortlist_mask

        binary_shortlist_mask = None
        if self.binary_shortlist_top_k > 0 and binary_scores is not None:
            binary_shortlist_mask = _masked_topk_mask(
                binary_scores,
                base_mask,
                self.binary_shortlist_top_k,
            )
            combined_mask = combined_mask | binary_shortlist_mask

        return combined_mask, explanation_shortlist_mask, binary_shortlist_mask

    def _agreement_aware_path_signals(
        self,
        base_scores: torch.Tensor,
        explanation_scores: torch.Tensor | None,
        binary_scores: torch.Tensor | None,
        path_metadata: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.agreement_aware_enabled:
            return None, None, None

        explanation_logits = explanation_scores if explanation_scores is not None else base_scores
        binary_logits = binary_scores if binary_scores is not None else explanation_logits
        explanation_probs = torch.sigmoid(explanation_logits)
        binary_probs = torch.sigmoid(binary_logits)
        agreement_scores = 1.0 - (explanation_probs - binary_probs).abs().clamp(min=0.0, max=1.0)

        if path_metadata is None:
            retrieval_confidence = torch.zeros_like(base_scores)
        else:
            retrieval_confidence = torch.sigmoid(
                path_metadata.get("confidence", torch.zeros_like(base_scores))
            )

        selector_scores = base_scores
        selector_scores = selector_scores + float(self.agreement_aware_cfg.get("explanation_weight", 0.0)) * (
            (explanation_probs - 0.5) * 2.0
        )
        selector_scores = selector_scores + float(self.agreement_aware_cfg.get("binary_weight", 0.0)) * (
            (binary_probs - 0.5) * 2.0
        )
        selector_scores = selector_scores + float(self.agreement_aware_cfg.get("agreement_weight", 0.0)) * (
            (agreement_scores - 0.5) * 2.0
        )
        selector_scores = selector_scores + float(self.agreement_aware_cfg.get("confidence_weight", 0.0)) * (
            (retrieval_confidence - 0.5) * 2.0
        )

        path_confidence = explanation_probs * agreement_scores
        if binary_scores is not None:
            path_confidence = path_confidence * binary_probs
        path_confidence = path_confidence * (0.5 + 0.5 * retrieval_confidence)
        return selector_scores, path_confidence.clamp(min=0.0, max=1.0), agreement_scores

    def forward(
        self,
        drug_embedding: torch.Tensor,
        disease_embedding: torch.Tensor,
        path_scores: torch.Tensor,
        path_reprs: torch.Tensor,
        explanation_scores: torch.Tensor | None = None,
        binary_scores: torch.Tensor | None = None,
        path_metadata: dict[str, torch.Tensor] | None = None,
        pair_features: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
        has_gold_rationale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pair_repr, direct_score = self.direct_pair(drug_embedding, disease_embedding, pair_features=pair_features)
        reranked_path_scores, rerank_bias, shortlist_mask, retrieval_logits = self.retrieval_reranker(
            pair_repr=pair_repr,
            path_reprs=path_reprs,
            evidence_scores=path_scores,
            explanation_scores=explanation_scores,
            bag_mask=bag_mask,
            path_metadata=path_metadata,
        )
        combined_shortlist_mask, explanation_shortlist_mask, binary_shortlist_mask = self._apply_dual_shortlist(
            shortlist_mask=shortlist_mask,
            bag_mask=bag_mask,
            explanation_scores=explanation_scores,
            binary_scores=binary_scores,
        )
        effective_bag_mask = shortlist_mask
        if bag_mask is not None:
            effective_bag_mask = bag_mask.bool() & combined_shortlist_mask.bool()
        else:
            effective_bag_mask = combined_shortlist_mask
        aggregation_path_reprs = path_reprs
        aggregation_path_scores = reranked_path_scores
        explanation_aggregation_path_reprs = aggregation_path_reprs
        explanation_aggregation_path_scores = explanation_scores if explanation_scores is not None else aggregation_path_scores
        if self.path_interaction_enabled:
            interacted_path_reprs, interacted_path_scores = self.path_interaction(
                pair_repr=pair_repr,
                path_reprs=path_reprs,
                path_scores=reranked_path_scores,
                bag_mask=effective_bag_mask,
            )
            if self.path_interaction.explanation_only:
                explanation_aggregation_path_reprs = interacted_path_reprs
                explanation_aggregation_path_scores = interacted_path_scores
            else:
                aggregation_path_reprs = interacted_path_reprs
                aggregation_path_scores = interacted_path_scores
                explanation_aggregation_path_reprs = interacted_path_reprs
                explanation_aggregation_path_scores = explanation_scores if explanation_scores is not None else interacted_path_scores
        selector_scores, path_confidence_scores, agreement_scores = self._agreement_aware_path_signals(
            base_scores=aggregation_path_scores,
            explanation_scores=explanation_scores,
            binary_scores=binary_scores,
            path_metadata=path_metadata,
        )
        if self.evidence_hierarchical_enabled and self.hierarchical_aggregator.enabled:
            evidence_path_score, attn, evidence_path_bag_repr = self.hierarchical_aggregator(
                pair_repr,
                aggregation_path_scores,
                aggregation_path_reprs,
                bag_mask=effective_bag_mask,
                selector_scores=selector_scores,
                path_confidence=path_confidence_scores,
                path_metadata=path_metadata,
            )
        else:
            evidence_path_score, attn, evidence_path_bag_repr = self.aggregator(
                pair_repr,
                aggregation_path_scores,
                aggregation_path_reprs,
                bag_mask=effective_bag_mask,
                selector_scores=selector_scores,
                path_confidence=path_confidence_scores,
            )
        no_gold_expert_outputs = None
        residual_dual_expert_outputs = None
        if (
            self.no_gold_expert_enabled
            and not self.residual_dual_expert_enabled
            and self.no_gold_evidence_expert.enabled
            and has_gold_rationale is not None
            and effective_bag_mask is not None
        ):
            no_gold_mask = has_gold_rationale.to(device=evidence_path_score.device, dtype=evidence_path_score.dtype) < 0.5
            if no_gold_mask.any():
                no_gold_expert_outputs = self.no_gold_evidence_expert(
                    pair_repr=pair_repr,
                    path_scores=aggregation_path_scores,
                    path_reprs=aggregation_path_reprs,
                    bag_mask=effective_bag_mask,
                    selector_scores=selector_scores,
                    path_confidence=path_confidence_scores,
                    path_metadata=path_metadata,
                )
                expert_mix_weight = self.no_gold_evidence_expert.compute_mix_weight(
                    bag_mask=effective_bag_mask,
                    path_confidence=path_confidence_scores,
                    pair_repr=pair_repr,
                    expert_repr=no_gold_expert_outputs["path_repr"],
                    path_metadata=path_metadata,
                )
                expert_mix_weight = expert_mix_weight * no_gold_mask.float()
                evidence_path_score = torch.where(
                    no_gold_mask,
                    evidence_path_score + expert_mix_weight * (no_gold_expert_outputs["path_score"] - evidence_path_score),
                    evidence_path_score,
                )
                evidence_path_bag_repr = torch.where(
                    no_gold_mask.unsqueeze(-1),
                    evidence_path_bag_repr
                    + expert_mix_weight.unsqueeze(-1) * (no_gold_expert_outputs["path_repr"] - evidence_path_bag_repr),
                    evidence_path_bag_repr,
                )
                attn = torch.where(
                    no_gold_mask.unsqueeze(-1),
                    attn + expert_mix_weight.unsqueeze(-1) * (no_gold_expert_outputs["path_attention"] - attn),
                    attn,
                )
        if (
            self.residual_dual_expert_enabled
            and self.residual_dual_evidence_aggregator.enabled
            and effective_bag_mask is not None
        ):
            residual_dual_expert_outputs = self.residual_dual_evidence_aggregator(
                pair_repr=pair_repr,
                base_score=evidence_path_score,
                base_repr=evidence_path_bag_repr,
                base_attn=attn,
                path_scores=aggregation_path_scores,
                path_reprs=aggregation_path_reprs,
                bag_mask=effective_bag_mask,
                has_gold_rationale=has_gold_rationale,
                selector_scores=selector_scores,
                path_confidence=path_confidence_scores,
                path_metadata=path_metadata,
            )
            evidence_path_score = residual_dual_expert_outputs["path_score"]
            evidence_path_bag_repr = residual_dual_expert_outputs["path_repr"]
            attn = residual_dual_expert_outputs["path_attention"]
        explanation_path_score = None
        explanation_path_bag_repr = None
        explanation_attn = None
        dual_agreement_features = None
        cross_view_features = None
        evidence_cross_attention = None
        explanation_cross_attention = None
        validity_path_score = None
        validity_path_bag_repr = None
        validity_attn = None
        validity_agreement_features = None
        validity_graph_outputs = None
        path_score = evidence_path_score
        path_bag_repr = evidence_path_bag_repr
        explanation_feedback_scale = torch.ones_like(evidence_path_score)
        if has_gold_rationale is not None and self.dual_evidence_only_when_no_gold:
            has_gold_rationale = has_gold_rationale.to(device=evidence_path_score.device, dtype=evidence_path_score.dtype)
            explanation_feedback_scale = torch.where(
                has_gold_rationale > 0.5,
                torch.ones_like(evidence_path_score),
                torch.full_like(evidence_path_score, self.dual_no_gold_feedback_scale),
            )
        if self.dual_aggregation_enabled and self.explanation_aggregator is not None:
            explanation_base_scores = explanation_aggregation_path_scores
            explanation_selector_scores, explanation_confidence_scores, _ = self._agreement_aware_path_signals(
                base_scores=explanation_base_scores,
                explanation_scores=explanation_scores,
                binary_scores=binary_scores,
                path_metadata=path_metadata,
            )
            if self.explanation_hierarchical_enabled and self.hierarchical_explanation_aggregator is not None and self.hierarchical_explanation_aggregator.enabled:
                explanation_path_score, explanation_attn, explanation_path_bag_repr = self.hierarchical_explanation_aggregator(
                    pair_repr,
                    explanation_base_scores,
                    explanation_aggregation_path_reprs,
                    bag_mask=effective_bag_mask,
                    selector_scores=explanation_selector_scores,
                    path_confidence=explanation_confidence_scores,
                    path_metadata=path_metadata,
                )
            else:
                explanation_path_score, explanation_attn, explanation_path_bag_repr = self.explanation_aggregator(
                    pair_repr,
                    explanation_base_scores,
                    explanation_aggregation_path_reprs,
                    bag_mask=effective_bag_mask,
                    selector_scores=explanation_selector_scores,
                    path_confidence=explanation_confidence_scores,
                )
            if self.cross_view_attention_enabled and self.cross_view_attention.enabled:
                cross_view_outputs = self.cross_view_attention(
                    evidence_bag_repr=evidence_path_bag_repr,
                    evidence_bag_score=evidence_path_score,
                    explanation_bag_repr=explanation_path_bag_repr,
                    explanation_bag_score=explanation_path_score,
                    evidence_path_reprs=aggregation_path_reprs,
                    evidence_path_scores=aggregation_path_scores,
                    explanation_path_reprs=explanation_aggregation_path_reprs,
                    explanation_path_scores=explanation_base_scores,
                    bag_mask=effective_bag_mask,
                )
                evidence_path_bag_repr = cross_view_outputs["evidence_bag_repr"]
                evidence_path_score = cross_view_outputs["evidence_bag_score"]
                explanation_path_bag_repr = cross_view_outputs["explanation_bag_repr"]
                explanation_path_score = cross_view_outputs["explanation_bag_score"]
                cross_view_features = cross_view_outputs["summary"]
                evidence_cross_attention = cross_view_outputs["evidence_attention"]
                explanation_cross_attention = cross_view_outputs["explanation_attention"]
            score_agreement = 1.0 - (
                torch.sigmoid(evidence_path_score) - torch.sigmoid(explanation_path_score)
            ).abs().clamp(min=0.0, max=1.0)
            repr_agreement = torch.nn.functional.cosine_similarity(
                evidence_path_bag_repr,
                explanation_path_bag_repr,
                dim=-1,
                eps=1e-6,
            )
            repr_agreement = ((repr_agreement + 1.0) * 0.5).clamp(min=0.0, max=1.0)
            path_score_gap = torch.abs(evidence_path_score - explanation_path_score)
            pair_mix_weight = self.dual_pair_mix_alpha * score_agreement * repr_agreement
            if self.dual_agreement_v2_enabled:
                evidence_prob = torch.sigmoid(evidence_path_score)
                explanation_prob = torch.sigmoid(explanation_path_score)
                disagreement = (evidence_prob - explanation_prob).abs().clamp(min=0.0, max=1.0)
                v2_scale = (
                    1.0
                    - float(self.dual_agreement_v2_cfg.get("disagreement_penalty", 0.5)) * disagreement
                ).clamp(min=float(self.dual_agreement_v2_cfg.get("min_scale", 0.2)))
                v2_scale = v2_scale * (
                    1.0
                    + float(self.dual_agreement_v2_cfg.get("explanation_confidence_weight", 0.1))
                    * ((explanation_prob - 0.5) * 2.0)
                ).clamp(min=0.0)
                v2_scale = v2_scale * (
                    1.0
                    + float(self.dual_agreement_v2_cfg.get("evidence_confidence_weight", 0.1))
                    * ((evidence_prob - 0.5) * 2.0)
                ).clamp(min=0.0)
                pair_mix_weight = pair_mix_weight * v2_scale
            pair_mix_weight = pair_mix_weight * explanation_feedback_scale
            if self.dual_pair_feedback_enabled:
                path_score = evidence_path_score + pair_mix_weight * (explanation_path_score - evidence_path_score)
                path_bag_repr = evidence_path_bag_repr + pair_mix_weight.unsqueeze(-1) * (
                    explanation_path_bag_repr - evidence_path_bag_repr
                )
                dual_agreement_features = torch.stack(
                    [
                        score_agreement,
                        repr_agreement,
                        pair_mix_weight,
                        path_score_gap,
                    ],
                    dim=-1,
                )
        if self.validity_aggregation_enabled and self.validity_aggregator is not None:
            validity_base_scores = binary_scores if binary_scores is not None else (
                explanation_scores if explanation_scores is not None else aggregation_path_scores
            )
            validity_selector_scores, validity_confidence_scores, _ = self._agreement_aware_path_signals(
                base_scores=validity_base_scores,
                explanation_scores=explanation_scores if explanation_scores is not None else validity_base_scores,
                binary_scores=binary_scores if binary_scores is not None else validity_base_scores,
                path_metadata=path_metadata,
            )
            validity_path_score, validity_attn, validity_path_bag_repr = self.validity_aggregator(
                pair_repr,
                validity_base_scores,
                aggregation_path_reprs,
                bag_mask=effective_bag_mask,
                selector_scores=validity_selector_scores,
                path_confidence=validity_confidence_scores,
            )
            validity_score_agreement = 1.0 - (
                torch.sigmoid(path_score) - torch.sigmoid(validity_path_score)
            ).abs().clamp(min=0.0, max=1.0)
            validity_repr_agreement = torch.nn.functional.cosine_similarity(
                path_bag_repr,
                validity_path_bag_repr,
                dim=-1,
                eps=1e-6,
            )
            validity_repr_agreement = ((validity_repr_agreement + 1.0) * 0.5).clamp(min=0.0, max=1.0)
            validity_mix_weight = self.validity_pair_mix_alpha * validity_score_agreement * validity_repr_agreement
            if self.validity_pair_feedback_enabled:
                path_score = path_score + validity_mix_weight * (validity_path_score - path_score)
                path_bag_repr = path_bag_repr + validity_mix_weight.unsqueeze(-1) * (
                    validity_path_bag_repr - path_bag_repr
                )
            validity_agreement_features = torch.stack(
                [
                    validity_score_agreement,
                    validity_repr_agreement,
                    validity_mix_weight,
                    torch.abs(path_score - validity_path_score),
                ],
                dim=-1,
            )
        if effective_bag_mask is None:
            bag_available = torch.ones(pair_repr.size(0), 1, dtype=pair_repr.dtype, device=pair_repr.device)
        else:
            bag_available = effective_bag_mask.any(dim=1, keepdim=True).to(dtype=pair_repr.dtype)
        explanation_scores_for_reliability = explanation_scores if explanation_scores is not None else None
        if (
            explanation_scores_for_reliability is not None
            and has_gold_rationale is not None
            and self.dual_evidence_only_when_no_gold
            and self.dual_use_explanation_in_reliability
        ):
            has_gold_mask = has_gold_rationale.to(
                device=explanation_scores_for_reliability.device,
                dtype=explanation_scores_for_reliability.dtype,
            ).unsqueeze(1)
            explanation_scores_for_reliability = (
                has_gold_mask * explanation_scores_for_reliability
                + (1.0 - has_gold_mask) * reranked_path_scores
            )
        reliability_outputs = self.reliability_estimator(
            evidence_scores=reranked_path_scores,
            bag_mask=effective_bag_mask if effective_bag_mask is not None else torch.ones_like(reranked_path_scores, dtype=torch.bool),
            explanation_scores=(
                explanation_scores_for_reliability if self.dual_use_explanation_in_reliability else None
            ),
            binary_scores=binary_scores,
            path_metadata=path_metadata,
        )
        if self.validity_graph_sidecar_enabled and self.validity_graph_sidecar.enabled:
            validity_graph_base_scores = binary_scores if binary_scores is not None else (
                explanation_scores if explanation_scores is not None else reranked_path_scores
            )
            validity_graph_outputs = self.validity_graph_sidecar(
                path_reprs=aggregation_path_reprs,
                base_scores=validity_graph_base_scores,
                bag_mask=effective_bag_mask if effective_bag_mask is not None else torch.ones_like(reranked_path_scores, dtype=torch.bool),
                path_metadata=path_metadata,
            )
        binary_calibration_outputs = self.binary_calibrator(
            evidence_scores=reranked_path_scores,
            explanation_scores=explanation_scores if explanation_scores is not None else reranked_path_scores,
            raw_binary_scores=binary_scores,
            agreement_scores=agreement_scores,
            path_confidence_scores=path_confidence_scores,
            retrieval_logits=retrieval_logits,
            path_metadata=path_metadata,
            bag_reliability=reliability_outputs["reliability"],
            bag_mask=effective_bag_mask if effective_bag_mask is not None else torch.ones_like(reranked_path_scores, dtype=torch.bool),
            validity_graph_scores=(
                validity_graph_outputs["graph_scores"] if validity_graph_outputs is not None and self.validity_graph_sidecar.use_in_calibration else None
            ),
            validity_graph_features=(
                validity_graph_outputs["graph_features"] if validity_graph_outputs is not None and self.validity_graph_sidecar.use_in_calibration else None
            ),
        )
        calibrated_binary_scores = binary_calibration_outputs["calibrated_scores"]
        reliability_features = reliability_outputs["features"]
        calibration_summary = binary_calibration_outputs["summary"]
        if calibration_summary is not None and self.binary_calibrator.pair_feedback_enabled:
            if reliability_features is None:
                reliability_features = calibration_summary
            else:
                reliability_features = torch.cat([reliability_features, calibration_summary], dim=-1)
        if dual_agreement_features is not None and self.dual_pair_feedback_enabled:
            if reliability_features is None:
                reliability_features = dual_agreement_features
            else:
                reliability_features = torch.cat([reliability_features, dual_agreement_features], dim=-1)
        if cross_view_features is not None and self.cross_view_attention.use_in_reliability:
            if reliability_features is None:
                reliability_features = cross_view_features
            else:
                reliability_features = torch.cat([reliability_features, cross_view_features], dim=-1)
        if validity_agreement_features is not None and self.validity_use_in_reliability:
            if reliability_features is None:
                reliability_features = validity_agreement_features
            else:
                reliability_features = torch.cat([reliability_features, validity_agreement_features], dim=-1)
        if validity_graph_outputs is not None and self.validity_graph_use_in_reliability:
            if reliability_features is None:
                reliability_features = validity_graph_outputs["summary"]
            else:
                reliability_features = torch.cat([reliability_features, validity_graph_outputs["summary"]], dim=-1)
        interaction_outputs = self.interactor(
            pair_repr=pair_repr,
            direct_score=direct_score,
            path_repr=path_bag_repr,
            path_score=path_score,
            bag_available=bag_available,
            reliability_features=reliability_features,
            mechanistic_reliability=reliability_outputs["reliability"],
            mechanistic_uncertainty=reliability_outputs["uncertainty"],
        )
        return {
            "pair_repr": pair_repr,
            "direct_pair_score": direct_score,
            "path_bag_score": path_score,
            "path_bag_repr": path_bag_repr,
            "evidence_path_bag_score": evidence_path_score,
            "evidence_path_bag_repr": evidence_path_bag_repr,
            "explanation_path_bag_score": explanation_path_score,
            "explanation_path_bag_repr": explanation_path_bag_repr,
            "raw_path_scores": path_scores,
            "interaction_path_scores": aggregation_path_scores,
            "interaction_path_reprs": aggregation_path_reprs,
            "explanation_path_scores": explanation_scores if explanation_scores is not None else path_scores,
            "binary_path_scores": calibrated_binary_scores if calibrated_binary_scores is not None else binary_scores,
            "raw_binary_path_scores": binary_scores if binary_scores is not None else None,
            "reranked_path_scores": reranked_path_scores,
            "retrieval_rerank_bias": rerank_bias,
            "retrieval_logits": retrieval_logits,
            "retrieval_shortlist_mask": shortlist_mask,
            "explanation_shortlist_mask": explanation_shortlist_mask,
            "binary_shortlist_mask": binary_shortlist_mask,
            "combined_shortlist_mask": combined_shortlist_mask,
            "aggregation_mask": effective_bag_mask,
            "aggregation_selector_scores": selector_scores,
            "explanation_aggregation_attention": explanation_attn,
            "evidence_cross_attention": evidence_cross_attention,
            "explanation_cross_attention": explanation_cross_attention,
            "cross_view_features": cross_view_features,
            "validity_path_bag_score": validity_path_score,
            "validity_path_bag_repr": validity_path_bag_repr,
            "validity_aggregation_attention": validity_attn,
            "validity_graph_scores": (
                validity_graph_outputs["graph_scores"] if validity_graph_outputs is not None else None
            ),
            "validity_graph_features": (
                validity_graph_outputs["graph_features"] if validity_graph_outputs is not None else None
            ),
            "validity_graph_summary": (
                validity_graph_outputs["summary"] if validity_graph_outputs is not None else None
            ),
            "validity_graph_attention": (
                validity_graph_outputs["attention"] if validity_graph_outputs is not None else None
            ),
            "path_confidence_scores": path_confidence_scores,
            "path_agreement_scores": agreement_scores,
            "dual_agreement_features": dual_agreement_features,
            "validity_agreement_features": validity_agreement_features,
            "no_gold_expert_scores": (
                no_gold_expert_outputs["expert_scores"] if no_gold_expert_outputs is not None else None
            ),
            "dual_expert_route_weights": (
                residual_dual_expert_outputs["route_weights"] if residual_dual_expert_outputs is not None else None
            ),
            "dual_expert_gold_mix_weight": (
                residual_dual_expert_outputs["gold_mix_weight"] if residual_dual_expert_outputs is not None else None
            ),
            "dual_expert_latent_mix_weight": (
                residual_dual_expert_outputs["latent_mix_weight"] if residual_dual_expert_outputs is not None else None
            ),
            "dual_expert_gold_scores": (
                residual_dual_expert_outputs["gold_expert_scores"] if residual_dual_expert_outputs is not None else None
            ),
            "dual_expert_latent_scores": (
                residual_dual_expert_outputs["latent_expert_scores"] if residual_dual_expert_outputs is not None else None
            ),
            "refined_pair_repr": interaction_outputs["refined_pair_repr"],
            "refined_path_repr": interaction_outputs["refined_path_repr"],
            "refined_direct_score": interaction_outputs["refined_direct_score"],
            "refined_path_score": interaction_outputs["refined_path_score"],
            "fusion_gate": interaction_outputs["fusion_gate"],
            "mechanistic_gate": interaction_outputs["mechanistic_gate"],
            "interaction_delta": interaction_outputs["interaction_delta"],
            "pair_score": interaction_outputs["pair_score"],
            "path_attention": attn,
            "mechanistic_reliability": interaction_outputs["mechanistic_reliability"],
            "mechanistic_uncertainty": interaction_outputs["mechanistic_uncertainty"],
            "reliability_features": interaction_outputs["reliability_features"],
            "reliability_summary": reliability_outputs.get("summary"),
            "reliability_raw_features": reliability_outputs.get("raw_features"),
            "binary_calibration_summary": calibration_summary,
            "binary_calibration_raw_features": binary_calibration_outputs.get("raw_features"),
        }
