"""Explicit pairwise graph features for non-Mech baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class EndpointStats:
    out_degree: float
    in_degree: float
    und_degree: float
    out_type_counts: np.ndarray
    in_type_counts: np.ndarray
    und_type_counts: np.ndarray
    out_rel_counts: np.ndarray
    in_rel_counts: np.ndarray
    und_rel_counts: np.ndarray
    unique_rel_neighbor_counts: np.ndarray
    neighbor_sets_by_type: dict[str, set[int]]
    neighbor_set_all: set[int]
    relation_neighbor_sets: dict[str, set[int]]
    two_hop_sets_by_type: dict[str, set[int]]


class PairwiseFeatureBuilder:
    """Build explicit endpoint and overlap features from the KG only."""

    def __init__(self, graph_data: dict[str, Any], pair_tables: dict[str, Any]) -> None:
        self.graph_data = graph_data
        self.pair_tables = pair_tables
        self.node_types = [
            node_type
            for node_type, idx in sorted(graph_data["node_type_vocab"].items(), key=lambda item: item[1])
            if node_type != "__UNK_TYPE__"
        ]
        self.relations = [
            relation
            for relation, idx in sorted(graph_data["relation_vocab"].items(), key=lambda item: item[1])
            if relation != "__UNK_REL__"
        ]
        self.two_hop_types = [
            "pathway",
            "biological_process",
            "molecular_function",
            "cellular_component",
            "effect/phenotype",
        ]
        self.semantic_overlap_pairs = [
            ("shared_proteins", "drug_protein", "disease_protein"),
            ("shared_effect_pos", "drug_effect", "disease_phenotype_positive"),
            ("shared_effect_neg", "drug_effect", "disease_phenotype_negative"),
        ]
        self.node_type_ids = graph_data["node_type_ids"].cpu()
        self.relation_by_idx = graph_data["relation_by_idx"]
        self.adjacency_ptr = graph_data["adjacency_ptr"].cpu()
        self.adjacency_dst = graph_data["adjacency_dst"].cpu()
        self.adjacency_rel = graph_data["adjacency_rel"].cpu()
        self.reverse_ptr, self.reverse_dst, self.reverse_rel = self._build_reverse_adjacency()
        self.node_degree = self._build_node_degree()
        self.protein_function_cache: dict[int, dict[str, set[int]]] = {}
        self.feature_names = self._build_feature_names()
        self.endpoint_cache: dict[int, EndpointStats] = {}
        self._warm_endpoint_cache()

    def _build_reverse_adjacency(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_src = self.graph_data["edge_src"].cpu()
        edge_dst = self.graph_data["edge_dst"].cpu()
        edge_rel = self.graph_data["edge_type_ids"].cpu()
        num_nodes = int(self.graph_data["metadata"]["num_nodes"])
        order = torch.argsort(edge_dst, stable=True)
        sorted_dst = edge_dst[order]
        reverse_dst = edge_src[order]
        reverse_rel = edge_rel[order]
        counts = torch.bincount(sorted_dst, minlength=num_nodes)
        reverse_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        reverse_ptr[1:] = torch.cumsum(counts, dim=0)
        return reverse_ptr, reverse_dst, reverse_rel

    def _build_node_degree(self) -> np.ndarray:
        num_nodes = int(self.graph_data["metadata"]["num_nodes"])
        degrees = np.zeros(num_nodes, dtype=np.float32)
        edge_src = self.graph_data["edge_src"].cpu().numpy()
        edge_dst = self.graph_data["edge_dst"].cpu().numpy()
        np.add.at(degrees, edge_src, 1.0)
        np.add.at(degrees, edge_dst, 1.0)
        return degrees

    def _build_feature_names(self) -> list[str]:
        names = [
            "drug_out_degree",
            "drug_in_degree",
            "drug_und_degree",
            "disease_out_degree",
            "disease_in_degree",
            "disease_und_degree",
        ]
        for prefix in ["drug", "disease"]:
            for scope in ["out", "in", "und"]:
                for node_type in self.node_types:
                    names.append(f"{prefix}_{scope}_type_count::{node_type}")
            for scope in ["out", "in", "und"]:
                for relation in self.relations:
                    names.append(f"{prefix}_{scope}_rel_count::{relation}")
            for relation in self.relations:
                names.append(f"{prefix}_unique_neighbor_count::{relation}")
            for node_type in self.two_hop_types:
                names.append(f"{prefix}_two_hop_count::{node_type}")
        names.extend(
            [
                "shared_neighbors_all",
                "jaccard_neighbors_all",
                "overlap_coeff_neighbors_all",
            ]
        )
        for node_type in self.node_types:
            names.extend(
                [
                    f"shared_neighbors::{node_type}",
                    f"jaccard_neighbors::{node_type}",
                    f"overlap_coeff_neighbors::{node_type}",
                ]
            )
        for feature_prefix, _, _ in self.semantic_overlap_pairs:
            names.extend(
                [
                    f"{feature_prefix}::count",
                    f"{feature_prefix}::jaccard",
                    f"{feature_prefix}::overlap_coeff",
                    f"{feature_prefix}::resource_allocation",
                    f"{feature_prefix}::adamic_adar",
                ]
            )
        for node_type in self.two_hop_types:
            names.extend(
                [
                    f"two_hop_overlap::{node_type}::count",
                    f"two_hop_overlap::{node_type}::jaccard",
                    f"two_hop_overlap::{node_type}::overlap_coeff",
                    f"two_hop_overlap::{node_type}::resource_allocation",
                    f"two_hop_overlap::{node_type}::adamic_adar",
                ]
            )
        return names

    def _warm_endpoint_cache(self) -> None:
        endpoint_indices: set[int] = set()
        for split_table in self.pair_tables.values():
            endpoint_indices.update(split_table["drug_indices"].tolist())
            endpoint_indices.update(split_table["disease_indices"].tolist())
        for node_idx in sorted(endpoint_indices):
            self.endpoint_cache[int(node_idx)] = self._compute_endpoint_stats(int(node_idx))

    def _slice_neighbors(
        self,
        node_idx: int,
        adjacency_ptr: torch.Tensor,
        adjacency_dst: torch.Tensor,
        adjacency_rel: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = int(adjacency_ptr[node_idx].item())
        end = int(adjacency_ptr[node_idx + 1].item())
        if end <= start:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        neighbors = adjacency_dst[start:end].numpy()
        relations = adjacency_rel[start:end].numpy()
        return neighbors, relations

    def _slice_undirected_neighbors(self, node_idx: int) -> np.ndarray:
        out_neighbors, _ = self._slice_neighbors(node_idx, self.adjacency_ptr, self.adjacency_dst, self.adjacency_rel)
        in_neighbors, _ = self._slice_neighbors(node_idx, self.reverse_ptr, self.reverse_dst, self.reverse_rel)
        if out_neighbors.size == 0 and in_neighbors.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([out_neighbors, in_neighbors]))

    def _count_types(self, neighbors: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(self.node_types), dtype=np.float32)
        if neighbors.size == 0:
            return counts
        neighbor_types = self.node_type_ids[torch.from_numpy(neighbors)].numpy()
        for idx, _node_type in enumerate(self.node_types, start=1):
            counts[idx - 1] = float((neighbor_types == idx).sum())
        return counts

    def _count_relations(self, relations: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(self.relations), dtype=np.float32)
        if relations.size == 0:
            return counts
        for idx in range(1, len(self.relation_by_idx)):
            counts[idx - 1] = float((relations == idx).sum())
        return counts

    def _relation_neighbor_sets(self, neighbors: np.ndarray, relations: np.ndarray) -> dict[str, set[int]]:
        sets = {relation: set() for relation in self.relations}
        if neighbors.size == 0:
            return sets
        for neighbor_idx, relation_idx in zip(neighbors.tolist(), relations.tolist(), strict=True):
            relation_name = self.relation_by_idx[int(relation_idx)]
            if relation_name == "__UNK_REL__":
                continue
            sets[relation_name].add(int(neighbor_idx))
        return sets

    def _neighbor_sets_by_type(self, neighbors: np.ndarray) -> tuple[dict[str, set[int]], set[int]]:
        typed_sets = {node_type: set() for node_type in self.node_types}
        if neighbors.size == 0:
            return typed_sets, set()
        for neighbor_idx in neighbors.tolist():
            node_type = self.graph_data["node_type_by_idx"][neighbor_idx]
            if node_type in typed_sets:
                typed_sets[node_type].add(int(neighbor_idx))
        return typed_sets, set(int(idx) for idx in neighbors.tolist())

    def _protein_two_hop_sets(self, protein_idx: int) -> dict[str, set[int]]:
        cached = self.protein_function_cache.get(protein_idx)
        if cached is not None:
            return cached
        neighbors = self._slice_undirected_neighbors(protein_idx)
        typed_sets = {node_type: set() for node_type in self.two_hop_types}
        for neighbor_idx in neighbors.tolist():
            node_type = self.graph_data["node_type_by_idx"][neighbor_idx]
            if node_type in typed_sets:
                typed_sets[node_type].add(int(neighbor_idx))
        self.protein_function_cache[protein_idx] = typed_sets
        return typed_sets

    def _two_hop_sets_from_proteins(self, relation_neighbor_sets: dict[str, set[int]]) -> dict[str, set[int]]:
        protein_neighbors = set()
        protein_neighbors.update(relation_neighbor_sets.get("drug_protein", set()))
        protein_neighbors.update(relation_neighbor_sets.get("disease_protein", set()))
        typed_sets = {node_type: set() for node_type in self.two_hop_types}
        for protein_idx in protein_neighbors:
            expansion = self._protein_two_hop_sets(protein_idx)
            for node_type in self.two_hop_types:
                typed_sets[node_type].update(expansion[node_type])
        return typed_sets

    def _compute_endpoint_stats(self, node_idx: int) -> EndpointStats:
        out_neighbors, out_relations = self._slice_neighbors(
            node_idx,
            self.adjacency_ptr,
            self.adjacency_dst,
            self.adjacency_rel,
        )
        in_neighbors, in_relations = self._slice_neighbors(
            node_idx,
            self.reverse_ptr,
            self.reverse_dst,
            self.reverse_rel,
        )
        if out_neighbors.size or in_neighbors.size:
            und_neighbors_raw = np.concatenate([out_neighbors, in_neighbors])
            und_neighbors = np.unique(und_neighbors_raw)
            und_relations = np.concatenate([out_relations, in_relations])
        else:
            und_neighbors_raw = np.zeros(0, dtype=np.int64)
            und_neighbors = np.zeros(0, dtype=np.int64)
            und_relations = np.zeros(0, dtype=np.int64)
        typed_sets, all_neighbors = self._neighbor_sets_by_type(und_neighbors)
        relation_neighbor_sets = self._relation_neighbor_sets(und_neighbors_raw, und_relations)
        unique_rel_neighbor_counts = np.asarray(
            [float(len(relation_neighbor_sets[relation])) for relation in self.relations],
            dtype=np.float32,
        )
        two_hop_sets = self._two_hop_sets_from_proteins(relation_neighbor_sets)
        return EndpointStats(
            out_degree=float(out_neighbors.size),
            in_degree=float(in_neighbors.size),
            und_degree=float(und_neighbors.size),
            out_type_counts=self._count_types(out_neighbors),
            in_type_counts=self._count_types(in_neighbors),
            und_type_counts=self._count_types(und_neighbors),
            out_rel_counts=self._count_relations(out_relations),
            in_rel_counts=self._count_relations(in_relations),
            und_rel_counts=self._count_relations(und_relations),
            unique_rel_neighbor_counts=unique_rel_neighbor_counts,
            neighbor_sets_by_type=typed_sets,
            neighbor_set_all=all_neighbors,
            relation_neighbor_sets=relation_neighbor_sets,
            two_hop_sets_by_type=two_hop_sets,
        )

    @staticmethod
    def _set_metrics(left: set[int], right: set[int]) -> tuple[float, float, float]:
        if not left and not right:
            return 0.0, 0.0, 0.0
        intersection = float(len(left & right))
        union = float(len(left | right))
        jaccard = intersection / union if union > 0 else 0.0
        overlap_coeff = intersection / max(1.0, float(min(len(left), len(right))))
        return intersection, jaccard, overlap_coeff

    def _set_metrics_with_centrality(self, left: set[int], right: set[int]) -> tuple[float, float, float, float, float]:
        intersection, jaccard, overlap_coeff = self._set_metrics(left, right)
        shared = left & right
        if not shared:
            return intersection, jaccard, overlap_coeff, 0.0, 0.0
        shared_idx = np.fromiter(shared, dtype=np.int64)
        degrees = self.node_degree[shared_idx]
        resource_allocation = float(np.sum(1.0 / np.clip(degrees, a_min=1.0, a_max=None)))
        adamic_adar = float(np.sum(1.0 / np.log1p(np.clip(degrees, a_min=1.0, a_max=None))))
        return intersection, jaccard, overlap_coeff, resource_allocation, adamic_adar

    def _pair_features(self, drug_idx: int, disease_idx: int) -> np.ndarray:
        drug = self.endpoint_cache[int(drug_idx)]
        disease = self.endpoint_cache[int(disease_idx)]
        features: list[float] = [
            drug.out_degree,
            drug.in_degree,
            drug.und_degree,
            disease.out_degree,
            disease.in_degree,
            disease.und_degree,
        ]
        features.extend(drug.out_type_counts.tolist())
        features.extend(drug.in_type_counts.tolist())
        features.extend(drug.und_type_counts.tolist())
        features.extend(drug.out_rel_counts.tolist())
        features.extend(drug.in_rel_counts.tolist())
        features.extend(drug.und_rel_counts.tolist())
        features.extend(drug.unique_rel_neighbor_counts.tolist())
        features.extend([float(len(drug.two_hop_sets_by_type[node_type])) for node_type in self.two_hop_types])
        features.extend(disease.out_type_counts.tolist())
        features.extend(disease.in_type_counts.tolist())
        features.extend(disease.und_type_counts.tolist())
        features.extend(disease.out_rel_counts.tolist())
        features.extend(disease.in_rel_counts.tolist())
        features.extend(disease.und_rel_counts.tolist())
        features.extend(disease.unique_rel_neighbor_counts.tolist())
        features.extend([float(len(disease.two_hop_sets_by_type[node_type])) for node_type in self.two_hop_types])
        features.extend(self._set_metrics(drug.neighbor_set_all, disease.neighbor_set_all))
        for node_type in self.node_types:
            features.extend(
                self._set_metrics(
                    drug.neighbor_sets_by_type[node_type],
                    disease.neighbor_sets_by_type[node_type],
                )
            )
        for _, drug_relation, disease_relation in self.semantic_overlap_pairs:
            features.extend(
                self._set_metrics_with_centrality(
                    drug.relation_neighbor_sets[drug_relation],
                    disease.relation_neighbor_sets[disease_relation],
                )
            )
        for node_type in self.two_hop_types:
            features.extend(
                self._set_metrics_with_centrality(
                    drug.two_hop_sets_by_type[node_type],
                    disease.two_hop_sets_by_type[node_type],
                )
            )
        return np.asarray(features, dtype=np.float32)

    def transform_split(self, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        table = self.pair_tables[split]
        num_pairs = int(table["labels"].numel())
        x = np.zeros((num_pairs, len(self.feature_names)), dtype=np.float32)
        for idx in range(num_pairs):
            x[idx] = self._pair_features(
                int(table["drug_indices"][idx].item()),
                int(table["disease_indices"][idx].item()),
            )
        y = table["labels"].numpy().astype(np.int64)
        return x, y, list(table["pair_ids"])


def build_pairwise_feature_tables(
    graph_data: dict[str, Any],
    pair_tables: dict[str, Any],
) -> dict[str, Any]:
    """Build train-standardized pairwise feature tensors for all splits."""

    builder = PairwiseFeatureBuilder(graph_data=graph_data, pair_tables=pair_tables)
    train_x, _, _ = builder.transform_split("train")
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)

    split_features: dict[str, torch.Tensor] = {}
    for split_name in pair_tables:
        split_x, _, _ = builder.transform_split(split_name)
        split_x = (split_x - mean.astype(np.float32)) / std.astype(np.float32)
        split_features[split_name] = torch.from_numpy(split_x.astype(np.float32))

    return {
        "feature_names": builder.feature_names,
        "feature_dim": len(builder.feature_names),
        "train_mean": torch.from_numpy(mean.astype(np.float32)),
        "train_std": torch.from_numpy(std.astype(np.float32)),
        "split_features": split_features,
    }


def load_or_build_pairwise_feature_tables(
    processed_dir: str | Path,
    graph_data: dict[str, Any],
    pair_tables: dict[str, Any],
) -> dict[str, Any]:
    """Load cached pairwise feature tables or build them on demand."""

    processed_dir = Path(processed_dir)
    cache_path = processed_dir / "pairwise_feature_tables.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    payload = build_pairwise_feature_tables(graph_data=graph_data, pair_tables=pair_tables)
    torch.save(payload, cache_path)
    return payload
