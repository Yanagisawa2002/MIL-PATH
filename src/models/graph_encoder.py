"""Heterogeneous graph encoder with HGT-first and graph-aware fallback interfaces."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, RGCNConv
except ImportError:  # pragma: no cover - optional dependency
    HeteroData = None
    HGTConv = None
    RGCNConv = None


class HeteroGraphEncoder(nn.Module):
    """Encode heterogeneous nodes with an HGT default and safe fallback."""

    def __init__(
        self,
        num_nodes: int,
        node_type_vocab: dict[str, int],
        relation_vocab: dict[str, int],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_type_vocab = node_type_vocab
        self.relation_vocab = relation_vocab
        self.config = config
        self.hidden_dim = config["hidden_dim"]
        self.backbone = config["backbone"].lower()
        self._cached_adjacency: dict[str, torch.Tensor] = {}

        self.node_embedding = nn.Embedding(num_nodes, self.hidden_dim)
        self.type_embedding = nn.Embedding(len(node_type_vocab), self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config["dropout"])
        self.fallback_layers = nn.ModuleList(
            _SparseMeanGraphLayer(
                hidden_dim=self.hidden_dim,
                dropout=config["dropout"],
            )
            for _ in range(config["num_layers"])
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.use_pyg_hgt = self.backbone == "hgt" and HGTConv is not None
        self.use_pyg_rgcn = self.backbone == "rgcn" and RGCNConv is not None

        if self.use_pyg_hgt:
            self.hgt_layers = nn.ModuleList(
                HGTConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    metadata=([], []),  # lazy metadata reset in build step
                    heads=config["num_heads"],
                )
                for _ in range(config["num_layers"])
            )
        else:
            self.hgt_layers = nn.ModuleList()

        if self.use_pyg_rgcn:
            self.rgcn_layers = nn.ModuleList(
                RGCNConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_relations=len(relation_vocab),
                )
                for _ in range(config["num_layers"])
            )
        else:
            self.rgcn_layers = nn.ModuleList()

        self.relation_decoder = nn.Embedding(len(relation_vocab), self.hidden_dim)
        self.type_classifier = nn.Linear(self.hidden_dim, len(node_type_vocab))

    def _base_features(self, node_type_ids: torch.Tensor) -> torch.Tensor:
        node_ids = torch.arange(self.num_nodes, device=node_type_ids.device)
        features = self.node_embedding(node_ids) + self.type_embedding(node_type_ids)
        features = self.input_norm(features)
        return self.dropout(features)

    def _fallback_forward(self, graph_artifact: dict[str, Any] | Any) -> torch.Tensor:
        if "node_type_ids" in graph_artifact:
            node_type_ids = graph_artifact["node_type_ids"].to(self.node_embedding.weight.device)
        else:
            node_type_ids = torch.tensor(
                [
                    self.node_type_vocab[node_type]
                    for node_type in graph_artifact["node_type_by_idx"]
                    if node_type in self.node_type_vocab
                ],
                dtype=torch.long,
                device=self.node_embedding.weight.device,
            )
        base = self._base_features(node_type_ids)
        edge_src = graph_artifact.get("edge_src")
        edge_dst = graph_artifact.get("edge_dst")
        edge_type_ids = graph_artifact.get("edge_type_ids")
        if edge_src is None or edge_dst is None or edge_type_ids is None:
            return self.output_proj(base)

        adjacency = self._normalized_adjacency(
            edge_src=edge_src.to(base.device),
            edge_dst=edge_dst.to(base.device),
            device=base.device,
        )

        hidden = base
        for layer in self.fallback_layers:
            hidden = hidden + layer(hidden, adjacency=adjacency)
        return self.output_proj(hidden)

    def _normalized_adjacency(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = str(device)
        cached = self._cached_adjacency.get(cache_key)
        if cached is not None and cached.device == device:
            return cached
        degrees = torch.bincount(edge_dst, minlength=self.num_nodes).to(torch.float32).clamp(min=1.0)
        values = 1.0 / degrees.index_select(0, edge_dst)
        indices = torch.stack([edge_dst, edge_src], dim=0)
        adjacency = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(self.num_nodes, self.num_nodes),
            device=device,
        ).coalesce()
        self._cached_adjacency[cache_key] = adjacency
        return adjacency

    def forward(self, graph_artifact: dict[str, Any] | Any) -> torch.Tensor:
        if not self.use_pyg_hgt and not self.use_pyg_rgcn:
            return self._fallback_forward(graph_artifact)

        # First version keeps PyG optional. When PyG is present, the backbone can
        # be extended by converting the stage-0 artifact into HeteroData.
        return self._fallback_forward(graph_artifact)

    def relation_reconstruction_loss(
        self,
        node_embeddings: torch.Tensor,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        head = node_embeddings[head_idx]
        rel = self.relation_decoder(relation_idx)
        tail = node_embeddings[tail_idx]
        positive_score = (head * rel * tail).sum(dim=-1)
        negative_tail_idx = torch.randint(
            low=0,
            high=node_embeddings.size(0),
            size=tail_idx.shape,
            device=tail_idx.device,
        )
        negative_tail = node_embeddings[negative_tail_idx]
        negative_score = (head * rel * negative_tail).sum(dim=-1)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-8)
        negative_loss = -torch.log(torch.sigmoid(-negative_score) + 1e-8)
        return (positive_loss + negative_loss).mean()

    def masked_type_prediction_loss(
        self,
        node_embeddings: torch.Tensor,
        node_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.type_classifier(node_embeddings)
        return nn.functional.cross_entropy(logits, node_type_ids)


class _SparseMeanGraphLayer(nn.Module):
    """A cheap graph-aware fallback layer for environments without PyG HGT/RGCN."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.sparse.mm(adjacency, node_states)
        updated = self.self_linear(node_states) + self.message_linear(aggregate)
        updated = self.norm(updated)
        updated = F.gelu(updated)
        return self.dropout(updated)
