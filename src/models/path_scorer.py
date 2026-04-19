"""Pair-conditioned path encoder and ranking scorer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


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
    ) -> torch.Tensor:
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
            return hidden[-1]

        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        gathered = outputs[torch.arange(batch_size, device=outputs.device), lengths]
        return gathered


class PairConditionedPathScorer(nn.Module):
    """Score a candidate path conditioned on its endpoint pair representation."""

    def __init__(self, hidden_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.path_encoder = EdgeAwarePathEncoder(
            hidden_dim=hidden_dim,
            relation_dim=config["relation_dim"],
            type_dim=config["type_dim"],
            dropout=config["dropout"],
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
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pair_embedding: torch.Tensor,
        node_states: torch.Tensor,
        relation_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path_repr = self.path_encoder(node_states, relation_ids, node_type_ids, mask=mask)
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
        fused = torch.cat(
            [
                pair_embedding,
                conditioned_path,
                torch.abs(pair_embedding - conditioned_path),
                pair_embedding * conditioned_path,
            ],
            dim=-1,
        )
        score = self.score_mlp(fused).squeeze(-1)
        return score, conditioned_path
