"""Pair-conditioned path encoder and ranking scorer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SchemaAwareMoERefiner(nn.Module):
    """Refine a base path logit with schema/hop/source-aware experts."""

    def __init__(self, fused_dim: int, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.num_experts = int(config.get("num_experts", 4))
        self.temperature = float(config.get("router_temperature", 1.0))
        route_hidden_dim = int(config.get("route_hidden_dim", hidden_dim))
        meta_dim = int(config.get("metadata_dim", max(16, hidden_dim // 4)))
        dropout = float(config.get("dropout", 0.1))
        self.delta_scale = float(config.get("delta_scale", 0.5))
        self.schema_bucket_count = int(config.get("schema_bucket_count", 2048))
        self.max_hops = int(config.get("max_hops", 8))
        self.path_source_vocab_size = int(config.get("path_source_vocab_size", 8))
        self.schema_embedding = nn.Embedding(self.schema_bucket_count + 1, meta_dim)
        self.hop_embedding = nn.Embedding(self.max_hops + 2, meta_dim)
        self.source_embedding = nn.Embedding(self.path_source_vocab_size, meta_dim)
        self.router = nn.Sequential(
            nn.Linear(fused_dim + meta_dim * 3, route_hidden_dim),
            nn.LayerNorm(route_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        fused: torch.Tensor,
        *,
        schema_bucket_ids: torch.Tensor | None,
        hop_counts: torch.Tensor | None,
        path_source_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if schema_bucket_ids is None:
            schema_bucket_ids = torch.zeros(fused.size(0), dtype=torch.long, device=fused.device)
        if hop_counts is None:
            hop_counts = torch.zeros_like(schema_bucket_ids)
        if path_source_ids is None:
            path_source_ids = torch.zeros_like(schema_bucket_ids)
        schema_bucket_ids = schema_bucket_ids.clamp(min=0, max=self.schema_bucket_count)
        hop_counts = hop_counts.clamp(min=0, max=self.max_hops + 1)
        path_source_ids = path_source_ids.clamp(min=0, max=self.path_source_vocab_size - 1)
        metadata = torch.cat(
            [
                self.schema_embedding(schema_bucket_ids),
                self.hop_embedding(hop_counts),
                self.source_embedding(path_source_ids),
            ],
            dim=-1,
        )
        router_logits = self.router(torch.cat([fused, metadata], dim=-1))
        router_weights = torch.softmax(router_logits / max(self.temperature, 1e-6), dim=-1)
        expert_logits = torch.cat([expert(fused) for expert in self.experts], dim=-1)
        refined_delta = (router_weights * expert_logits).sum(dim=-1)
        return self.delta_scale * refined_delta, router_weights


class EdgeAwarePathEncoder(nn.Module):
    """Encode a typed path with node states, relation ids, and node-type ids."""

    def __init__(self, hidden_dim: int, relation_dim: int, type_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(4096, relation_dim)
        self.type_embedding = nn.Embedding(256, type_dim)
        self.input_proj = nn.Linear(hidden_dim + relation_dim + type_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        relation_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = node_states.shape
        relation_emb = self.relation_embedding(relation_ids)
        zero_rel = torch.zeros(
            batch_size,
            1,
            relation_emb.size(-1),
            device=node_states.device,
            dtype=node_states.dtype,
        )
        relation_aligned = torch.cat([zero_rel, relation_emb], dim=1)
        type_emb = self.type_embedding(node_type_ids)
        fused = torch.cat([node_states, relation_aligned, type_emb], dim=-1)
        fused = self.dropout(self.input_proj(fused))
        outputs, hidden = self.gru(fused)
        if mask is None:
            final_repr = hidden[-1]
            if return_sequence:
                return final_repr, outputs
            return final_repr

        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        gathered = outputs[torch.arange(batch_size, device=outputs.device), lengths]
        if return_sequence:
            return gathered, outputs
        return gathered


class SubpathAwareExplanation(nn.Module):
    """Extract a pair-conditioned subpath summary for explanation scoring."""

    def __init__(self, hidden_dim: int, dropout: float, config: dict[str, Any]) -> None:
        super().__init__()
        attn_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        self.temperature = float(config.get("temperature", 1.0))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, attn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden_dim, 1),
        )
        self.context_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        pair_embedding: torch.Tensor,
        conditioned_path: torch.Tensor,
        path_sequence: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_expanded = pair_embedding.unsqueeze(1).expand(-1, path_sequence.size(1), -1)
        attn_features = torch.cat(
            [
                pair_expanded,
                path_sequence,
                torch.abs(pair_expanded - path_sequence),
                pair_expanded * path_sequence,
            ],
            dim=-1,
        )
        attn_logits = self.attention(attn_features).squeeze(-1)
        empty_rows = None
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))
            empty_rows = ~mask.bool().any(dim=1)
            if empty_rows.any():
                attn_logits = attn_logits.masked_fill(empty_rows.unsqueeze(1), 0.0)
        attn_weights = torch.softmax(attn_logits / max(self.temperature, 1e-6), dim=1)
        if empty_rows is not None and empty_rows.any():
            attn_weights = attn_weights.masked_fill(empty_rows.unsqueeze(1), 0.0)
        subpath_context = (attn_weights.unsqueeze(-1) * path_sequence).sum(dim=1)
        explanation_path = self.context_norm(
            conditioned_path + self.context_update(torch.cat([conditioned_path, subpath_context], dim=-1))
        )
        return explanation_path, subpath_context, attn_weights


class PrototypeAwareExplanation(nn.Module):
    """Refine explanation paths with learned mechanism prototypes."""

    def __init__(self, hidden_dim: int, dropout: float, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        if not self.enabled:
            self.num_prototypes = 0
            self.schema_embedding = None
            self.hop_embedding = None
            self.source_embedding = None
            self.prototype_bank = None
            self.router = None
            self.update = None
            self.memory_update = None
            self.memory_gate = None
            self.norm = None
            return
        self.mode = str(config.get("mode", "direct_update"))
        self.num_prototypes = int(config.get("num_prototypes", 12))
        metadata_dim = int(config.get("metadata_dim", max(16, hidden_dim // 4)))
        router_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        self.temperature = float(config.get("temperature", 1.0))
        self.memory_top_k = int(config.get("memory_top_k", min(4, self.num_prototypes)))
        self.residual_scale = float(config.get("residual_scale", 0.2))
        self.schema_bucket_count = int(config.get("schema_bucket_count", 2048))
        self.max_hops = int(config.get("max_hops", 8))
        self.path_source_vocab_size = int(config.get("path_source_vocab_size", 8))
        self.schema_embedding = nn.Embedding(self.schema_bucket_count + 1, metadata_dim)
        self.hop_embedding = nn.Embedding(self.max_hops + 2, metadata_dim)
        self.source_embedding = nn.Embedding(self.path_source_vocab_size, metadata_dim)
        self.prototype_bank = nn.Parameter(torch.randn(self.num_prototypes, hidden_dim) * 0.02)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2 + metadata_dim * 3, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, self.num_prototypes),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        pair_embedding: torch.Tensor,
        explanation_path: torch.Tensor,
        schema_bucket_ids: torch.Tensor | None,
        hop_counts: torch.Tensor | None,
        path_source_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.enabled:
            return explanation_path, None, None
        if schema_bucket_ids is None:
            schema_bucket_ids = torch.zeros(explanation_path.size(0), dtype=torch.long, device=explanation_path.device)
        if hop_counts is None:
            hop_counts = torch.zeros_like(schema_bucket_ids)
        if path_source_ids is None:
            path_source_ids = torch.zeros_like(schema_bucket_ids)
        schema_bucket_ids = schema_bucket_ids.clamp(min=0, max=self.schema_bucket_count)
        hop_counts = hop_counts.clamp(min=0, max=self.max_hops + 1)
        path_source_ids = path_source_ids.clamp(min=0, max=self.path_source_vocab_size - 1)
        metadata = torch.cat(
            [
                self.schema_embedding(schema_bucket_ids),
                self.hop_embedding(hop_counts),
                self.source_embedding(path_source_ids),
            ],
            dim=-1,
        )
        router_logits = self.router(torch.cat([pair_embedding, explanation_path, metadata], dim=-1))
        prototype_weights = torch.softmax(router_logits / max(self.temperature, 1e-6), dim=-1)
        if 0 < self.memory_top_k < self.num_prototypes:
            top_weights, top_idx = prototype_weights.topk(k=self.memory_top_k, dim=-1)
            sparse_weights = torch.zeros_like(prototype_weights)
            sparse_weights.scatter_(1, top_idx, top_weights)
            prototype_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        prototype_context = prototype_weights @ self.prototype_bank
        if self.mode == "memory_residual":
            memory_features = torch.cat(
                [
                    explanation_path,
                    prototype_context,
                    torch.abs(explanation_path - prototype_context),
                    explanation_path * prototype_context,
                ],
                dim=-1,
            )
            memory_delta = self.memory_update(memory_features)
            memory_gate = self.memory_gate(memory_features)
            updated = self.norm(explanation_path + self.residual_scale * memory_gate * memory_delta)
        else:
            updated = self.norm(
                explanation_path + self.update(torch.cat([explanation_path, prototype_context], dim=-1))
            )
        return updated, prototype_context, prototype_weights


class MultiViewPathEncoder(nn.Module):
    """Refine the base sequential path representation with auxiliary path views."""

    def __init__(self, hidden_dim: int, dropout: float, config: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = bool(config.get("enabled", False))
        if not self.enabled:
            self.endpoint_proj = None
            self.relation_proj = None
            self.type_proj = None
            self.update = None
            self.gate = None
            self.norm = None
            return
        fusion_hidden_dim = int(config.get("hidden_dim", hidden_dim))
        relation_dim = int(config.get("relation_dim", 64))
        type_dim = int(config.get("type_dim", 32))
        self.endpoint_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(relation_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        self.type_proj = nn.Sequential(
            nn.Linear(type_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 4, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        base_path_repr: torch.Tensor,
        node_states: torch.Tensor,
        relation_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        mask: torch.Tensor | None,
        relation_embedding: nn.Embedding,
        type_embedding: nn.Embedding,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if not self.enabled:
            return base_path_repr, None
        if mask is None:
            last_indices = torch.full(
                (node_states.size(0),),
                node_states.size(1) - 1,
                dtype=torch.long,
                device=node_states.device,
            )
            type_mask = torch.ones(node_states.size(0), node_states.size(1), device=node_states.device, dtype=node_states.dtype)
        else:
            last_indices = mask.long().sum(dim=1).clamp(min=1) - 1
            type_mask = mask.float()
        start_states = node_states[:, 0]
        end_states = node_states[torch.arange(node_states.size(0), device=node_states.device), last_indices]
        endpoint_view = self.endpoint_proj(torch.cat([start_states, end_states], dim=-1))

        if relation_ids.size(1) > 0:
            relation_emb = relation_embedding(relation_ids)
            relation_view = self.relation_proj(relation_emb.mean(dim=1))
        else:
            relation_view = self.relation_proj(
                torch.zeros(
                    node_states.size(0),
                    relation_embedding.embedding_dim,
                    device=node_states.device,
                    dtype=node_states.dtype,
                )
            )

        type_emb = type_embedding(node_type_ids)
        type_view = self.type_proj(
            (type_emb * type_mask.unsqueeze(-1)).sum(dim=1) / type_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        )
        fusion_input = torch.cat([base_path_repr, endpoint_view, relation_view, type_view], dim=-1)
        refined = self.norm(base_path_repr + self.gate(fusion_input) * self.update(fusion_input))
        aux = {
            "endpoint_view": endpoint_view,
            "relation_view": relation_view,
            "type_view": type_view,
        }
        return refined, aux


class PairConditionedPathScorer(nn.Module):
    """Score a candidate path conditioned on its endpoint pair representation."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        binary_head_cfg = dict(config.get("binary_head", {}))
        decoupled_cfg = dict(config.get("decoupled_heads", {}))
        moe_cfg = dict(config.get("schema_aware_moe", {}))
        subpath_cfg = dict(config.get("subpath_explanation", {}))
        prototype_cfg = dict(config.get("prototype_aware_explanation", {}))
        multiview_cfg = dict(config.get("multi_view_encoder", {}))
        self.binary_head_enabled = bool(binary_head_cfg.get("enabled", False))
        self.decoupled_heads_enabled = bool(decoupled_cfg.get("enabled", False))
        self.schema_aware_moe_enabled = bool(moe_cfg.get("enabled", False))
        self.subpath_explanation_enabled = bool(subpath_cfg.get("enabled", False))
        self.prototype_aware_explanation_enabled = bool(prototype_cfg.get("enabled", False))
        self.multi_view_enabled = bool(multiview_cfg.get("enabled", False))
        self.path_encoder = EdgeAwarePathEncoder(
            hidden_dim=hidden_dim,
            relation_dim=config["relation_dim"],
            type_dim=config["type_dim"],
            dropout=config["dropout"],
        )
        self.multi_view_encoder = MultiViewPathEncoder(
            hidden_dim=hidden_dim,
            dropout=config["dropout"],
            config={
                **multiview_cfg,
                "relation_dim": config["relation_dim"],
                "type_dim": config["type_dim"],
            },
        )
        self.pair_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.path_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_norm = nn.LayerNorm(hidden_dim)
        self.subpath_explanation = None
        if self.subpath_explanation_enabled:
            self.subpath_explanation = SubpathAwareExplanation(
                hidden_dim=hidden_dim,
                dropout=config["dropout"],
                config=subpath_cfg,
            )
        self.prototype_aware_explanation = PrototypeAwareExplanation(
            hidden_dim=hidden_dim,
            dropout=config["dropout"],
            config=prototype_cfg,
        )
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden_dim, 1),
        )
        fused_dim = hidden_dim * 4
        self.evidence_refiner = None
        self.explanation_refiner = None
        self.binary_refiner = None
        if self.schema_aware_moe_enabled:
            self.evidence_refiner = SchemaAwareMoERefiner(fused_dim=fused_dim, hidden_dim=hidden_dim, config=moe_cfg)
        self.explanation_head = None
        if self.decoupled_heads_enabled:
            explanation_hidden_dim = int(decoupled_cfg.get("hidden_dim", hidden_dim))
            explanation_dropout = float(decoupled_cfg.get("dropout", config["dropout"]))
            self.explanation_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, explanation_hidden_dim),
                nn.GELU(),
                nn.Dropout(explanation_dropout),
                nn.Linear(explanation_hidden_dim, 1),
            )
            if self.schema_aware_moe_enabled:
                self.explanation_refiner = SchemaAwareMoERefiner(
                    fused_dim=fused_dim,
                    hidden_dim=hidden_dim,
                    config=moe_cfg,
                )
        self.binary_head = None
        if self.binary_head_enabled:
            binary_hidden_dim = int(binary_head_cfg.get("hidden_dim", hidden_dim))
            binary_dropout = float(binary_head_cfg.get("dropout", config["dropout"]))
            self.binary_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, binary_hidden_dim),
                nn.GELU(),
                nn.Dropout(binary_dropout),
                nn.Linear(binary_hidden_dim, 1),
            )
            if self.schema_aware_moe_enabled:
                self.binary_refiner = SchemaAwareMoERefiner(
                    fused_dim=fused_dim,
                    hidden_dim=hidden_dim,
                    config=moe_cfg,
                )

    def forward(
        self,
        pair_embedding: torch.Tensor,
        node_states: torch.Tensor,
        relation_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        schema_bucket_ids: torch.Tensor | None = None,
        hop_counts: torch.Tensor | None = None,
        path_source_ids: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        path_sequence = None
        if self.subpath_explanation_enabled:
            path_repr, path_sequence = self.path_encoder(
                node_states,
                relation_ids,
                node_type_ids,
                mask=mask,
                return_sequence=True,
            )
        else:
            path_repr = self.path_encoder(node_states, relation_ids, node_type_ids, mask=mask)
        multiview_aux = None
        if self.multi_view_enabled:
            path_repr, multiview_aux = self.multi_view_encoder(
                base_path_repr=path_repr,
                node_states=node_states,
                relation_ids=relation_ids,
                node_type_ids=node_type_ids,
                mask=mask,
                relation_embedding=self.path_encoder.relation_embedding,
                type_embedding=self.path_encoder.type_embedding,
            )
        pair_modulation = self.pair_modulation(pair_embedding)
        gate, bias = pair_modulation.chunk(2, dim=-1)
        modulated_path = path_repr * (1.0 + 0.5 * torch.tanh(gate)) + bias
        pair_path_features = torch.cat(
            [
                pair_embedding,
                modulated_path,
                torch.abs(pair_embedding - modulated_path),
                pair_embedding * modulated_path,
            ],
            dim=-1,
        )
        conditioned_path = self.path_norm(modulated_path + self.path_interaction(pair_path_features))
        explanation_path = conditioned_path
        subpath_context = None
        subpath_attention = None
        if self.subpath_explanation is not None and path_sequence is not None:
            explanation_path, subpath_context, subpath_attention = self.subpath_explanation(
                pair_embedding=pair_embedding,
                conditioned_path=conditioned_path,
                path_sequence=path_sequence,
                mask=mask,
            )
        prototype_context = None
        prototype_attention = None
        if self.prototype_aware_explanation.enabled:
            explanation_path, prototype_context, prototype_attention = self.prototype_aware_explanation(
                pair_embedding=pair_embedding,
                explanation_path=explanation_path,
                schema_bucket_ids=schema_bucket_ids,
                hop_counts=hop_counts,
                path_source_ids=path_source_ids,
            )
        fused = torch.cat(
            [
                pair_embedding,
                conditioned_path,
                torch.abs(pair_embedding - conditioned_path),
                pair_embedding * conditioned_path,
            ],
            dim=-1,
        )
        evidence_score = self.evidence_head(fused).squeeze(-1)
        evidence_router = None
        if self.evidence_refiner is not None:
            evidence_delta, evidence_router = self.evidence_refiner(
                fused,
                schema_bucket_ids=schema_bucket_ids,
                hop_counts=hop_counts,
                path_source_ids=path_source_ids,
            )
            evidence_score = evidence_score + evidence_delta
        if self.explanation_head is not None:
            explanation_fused = torch.cat(
                [
                    pair_embedding,
                    explanation_path,
                    torch.abs(pair_embedding - explanation_path),
                    pair_embedding * explanation_path,
                ],
                dim=-1,
            )
            explanation_score = self.explanation_head(explanation_fused).squeeze(-1)
            explanation_router = None
            if self.explanation_refiner is not None:
                explanation_delta, explanation_router = self.explanation_refiner(
                    explanation_fused,
                    schema_bucket_ids=schema_bucket_ids,
                    hop_counts=hop_counts,
                    path_source_ids=path_source_ids,
                )
                explanation_score = explanation_score + explanation_delta
        else:
            explanation_score = evidence_score
            explanation_router = evidence_router
        if not return_aux:
            return evidence_score, conditioned_path

        aux: dict[str, torch.Tensor] = {}
        aux["evidence_score"] = evidence_score
        aux["explanation_score"] = explanation_score
        if multiview_aux is not None:
            aux.update(multiview_aux)
        if subpath_context is not None:
            aux["subpath_context"] = subpath_context
        if subpath_attention is not None:
            aux["subpath_attention"] = subpath_attention
        if prototype_context is not None:
            aux["prototype_context"] = prototype_context
        if prototype_attention is not None:
            aux["prototype_attention"] = prototype_attention
        if evidence_router is not None:
            aux["evidence_router_weights"] = evidence_router
        if explanation_router is not None:
            aux["explanation_router_weights"] = explanation_router
        if self.binary_head is not None:
            binary_logit = self.binary_head(fused).squeeze(-1)
            if self.binary_refiner is not None:
                binary_delta, binary_router = self.binary_refiner(
                    fused,
                    schema_bucket_ids=schema_bucket_ids,
                    hop_counts=hop_counts,
                    path_source_ids=path_source_ids,
                )
                binary_logit = binary_logit + binary_delta
                aux["binary_router_weights"] = binary_router
            aux["binary_logit"] = binary_logit
        return evidence_score, conditioned_path, aux
