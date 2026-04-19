"""KG-only RGCN pairwise baselines with optional feature and pooling augmentations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv
except ImportError:  # pragma: no cover - optional dependency
    RGCNConv = None


def build_rgcn_graph_inputs(
    graph_data: dict[str, Any],
    add_reverse_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build edge index/type tensors for a full-graph RGCN baseline."""

    edge_src = graph_data["edge_src"].to(torch.long)
    edge_dst = graph_data["edge_dst"].to(torch.long)
    edge_type = graph_data["edge_type_ids"].to(torch.long)
    num_relations = int(len(graph_data["relation_vocab"]))
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    if not add_reverse_edges:
        return edge_index, edge_type, num_relations

    reverse_edge_index = torch.stack([edge_dst, edge_src], dim=0)
    reverse_edge_type = edge_type + num_relations
    full_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    full_edge_type = torch.cat([edge_type, reverse_edge_type], dim=0)
    return full_edge_index, full_edge_type, num_relations * 2


def build_neighbor_mean_adjacency(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a sparse normalized adjacency for one-hop neighborhood mean pooling."""

    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    degrees = torch.bincount(dst, minlength=num_nodes).to(device=device, dtype=torch.float32).clamp(min=1.0)
    values = 1.0 / degrees.index_select(0, dst)
    adjacency = torch.sparse_coo_tensor(
        indices=torch.stack([dst, src], dim=0),
        values=values,
        size=(num_nodes, num_nodes),
        device=device,
    )
    return adjacency.coalesce()


class RGCNPairwiseClassifier(nn.Module):
    """Relation-aware full-graph encoder plus pairwise MLP head."""

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_types: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int | None = 8,
        dropout: float = 0.1,
        pair_feature_dim: int = 0,
        pair_feature_hidden_dim: int | None = None,
        use_neighborhood_pooling: bool = False,
    ) -> None:
        super().__init__()
        if RGCNConv is None:
            raise ImportError("torch_geometric is required for the pure RGCN baseline.")

        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.use_neighborhood_pooling = bool(use_neighborhood_pooling)
        self.pair_feature_dim = int(pair_feature_dim)
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.type_embedding = nn.Embedding(int(num_node_types), self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        basis_count = None
        if num_bases is not None:
            basis_count = min(int(num_bases), int(num_relations))
        self.convs = nn.ModuleList(
            RGCNConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_relations=int(num_relations),
                num_bases=basis_count,
                root_weight=True,
            )
            for _ in range(int(num_layers))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(int(num_layers)))
        self.dropout = nn.Dropout(dropout)
        pair_head_input_dim = self.hidden_dim * 4
        if self.use_neighborhood_pooling:
            pair_head_input_dim += self.hidden_dim * 4
        self.pair_feature_proj = None
        if self.pair_feature_dim > 0:
            feature_hidden_dim = int(pair_feature_hidden_dim or self.hidden_dim)
            self.pair_feature_proj = nn.Sequential(
                nn.Linear(self.pair_feature_dim, feature_hidden_dim),
                nn.LayerNorm(feature_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            pair_head_input_dim += feature_hidden_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_head_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def encode(
        self,
        *,
        node_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = torch.arange(self.num_nodes, device=node_type_ids.device)
        hidden = self.node_embedding(node_ids) + self.type_embedding(node_type_ids)
        hidden = self.input_dropout(self.input_norm(hidden))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            updated = conv(hidden, edge_index=edge_index, edge_type=edge_type)
            hidden = norm(hidden + self.dropout(F.gelu(updated)))
        return hidden

    def pair_logits(
        self,
        node_embeddings: torch.Tensor,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        pair_features: torch.Tensor | None = None,
        neighbor_adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        drug_embedding = node_embeddings[drug_indices]
        disease_embedding = node_embeddings[disease_indices]
        pair_parts = [
            [
                drug_embedding,
                disease_embedding,
                torch.abs(drug_embedding - disease_embedding),
                drug_embedding * disease_embedding,
            ]
        ]

        if self.use_neighborhood_pooling:
            if neighbor_adjacency is None:
                raise ValueError("neighbor_adjacency is required when neighborhood pooling is enabled")
            neighbor_embeddings = torch.sparse.mm(neighbor_adjacency, node_embeddings)
            drug_neighbor = neighbor_embeddings[drug_indices]
            disease_neighbor = neighbor_embeddings[disease_indices]
            pair_parts.append(
                [
                    drug_neighbor,
                    disease_neighbor,
                    torch.abs(drug_neighbor - disease_neighbor),
                    drug_neighbor * disease_neighbor,
                ]
            )

        if self.pair_feature_proj is not None:
            if pair_features is None:
                raise ValueError("pair_features are required when pair feature augmentation is enabled")
            pair_parts.append([self.pair_feature_proj(pair_features)])

        pair_tensor = torch.cat([tensor for group in pair_parts for tensor in group], dim=-1)
        return self.pair_head(pair_tensor).squeeze(-1)
