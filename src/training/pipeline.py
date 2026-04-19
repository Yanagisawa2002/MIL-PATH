"""Real-data dataloader assembly and stage batch execution helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.baselines.pairwise_features import load_or_build_pairwise_feature_tables
from src.data.collators import GraphPretrainCollator, PairBagCollator, PathRankingCollator
from src.data.datasets import (
    ArtifactBundle,
    GraphPretrainDataset,
    LazyCandidatePathStore,
    PairBagDataset,
    PathRankingDataset,
    PseudoRationaleDataset,
    load_artifact_bundle,
)
from src.utils.config import deep_merge


def gather_node_states(node_embeddings: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    flat = node_indices.reshape(-1)
    gathered = node_embeddings[flat]
    return gathered.reshape(*node_indices.shape, node_embeddings.size(-1))


def resolve_retriever_config(config: dict[str, Any], profile: str = "default") -> tuple[dict[str, Any], str]:
    retriever_cfg = deepcopy(config["retriever"])
    profile_overrides = {
        key: retriever_cfg.pop(key)
        for key in list(retriever_cfg.keys())
        if isinstance(retriever_cfg.get(key), dict)
    }
    if profile == "default":
        return retriever_cfg, "default"
    override = profile_overrides.get(profile, {})
    if not override:
        return retriever_cfg, profile
    merged = deep_merge(retriever_cfg, override)
    cache_namespace = override.get("cache_namespace", profile)
    return merged, cache_namespace


def build_candidate_store(
    config: dict[str, Any],
    bundle: ArtifactBundle,
    profile: str = "default",
) -> LazyCandidatePathStore:
    retriever_config, cache_namespace = resolve_retriever_config(config, profile=profile)
    return LazyCandidatePathStore(
        graph_data=bundle.graph_data,
        path_tensor_store=bundle.path_tensor_store,
        schema_prior=bundle.schema_prior,
        retriever_config=retriever_config,
        cache_dir=Path(config["paths"]["cache_dir"]) / "candidate_paths_cache",
        cache_namespace=cache_namespace,
    )


def build_stage_dataloaders(config: dict[str, Any], split: str = "train") -> dict[str, Any]:
    bundle = load_artifact_bundle(config["paths"]["processed_dir"])
    direct_pair_feature_cfg = config["model"].get("direct_pair_features", {})
    if direct_pair_feature_cfg.get("enabled", False):
        feature_payload = load_or_build_pairwise_feature_tables(
            processed_dir=config["paths"]["processed_dir"],
            graph_data=bundle.graph_data,
            pair_tables=bundle.pair_tables,
        )
        for split_name, feature_tensor in feature_payload["split_features"].items():
            bundle.pair_tables[split_name]["pair_features"] = feature_tensor
        bundle.pair_tables["feature_names"] = feature_payload["feature_names"]
        bundle.pair_tables["feature_dim"] = feature_payload["feature_dim"]
    stage3_profile = (
        "stage3_cached"
        if str(config["training"].get("stage3_bag_policy", "retrieve_all")).startswith("cached_")
        else "default"
    )
    candidate_store = build_candidate_store(config, bundle, profile=stage3_profile)
    stage4_candidate_store = build_candidate_store(config, bundle, profile="stage4")
    stage3_retriever_config, stage3_cache_namespace = resolve_retriever_config(config, profile=stage3_profile)
    stage4_retriever_config, _ = resolve_retriever_config(config, profile="stage4")

    stage1_dataset = GraphPretrainDataset(
        graph_data=bundle.graph_data,
        max_edges=config["training"].get("stage1_edge_samples"),
        seed=int(config["project"].get("seed", 42)),
    )
    stage2_dataset = PathRankingDataset(
        graph_data=bundle.graph_data,
        pair_tables=bundle.pair_tables,
        path_tensor_store=bundle.path_tensor_store,
        split=split,
        num_negative_paths=config["training"].get("stage2_num_negative_paths", 4),
        negative_mix=config["training"].get("stage2_negative_mix"),
    )
    stage3_dataset = PairBagDataset(
        pair_table=bundle.pair_tables[split],
        candidate_store=candidate_store,
        split=split,
        include_gold=True,
        bag_policy=config["training"].get("stage3_bag_policy", "retrieve_all"),
    )
    stage4_dataset = PseudoRationaleDataset(
        pair_table=bundle.pair_tables[split],
        candidate_store=stage4_candidate_store,
        max_pairs=config["training"].get("stage4_max_pairs"),
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    if os.name == "nt":
        num_workers = 0

    return {
        "bundle": bundle,
        "candidate_store": candidate_store,
        "stage4_candidate_store": stage4_candidate_store,
        "stage3_cache_namespace": stage3_cache_namespace,
        "stage1": DataLoader(
            stage1_dataset,
            batch_size=max(1, len(stage1_dataset)),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=GraphPretrainCollator(bundle.graph_data),
        ),
        "stage2": DataLoader(
            stage2_dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            collate_fn=PathRankingCollator(),
        ),
        "stage3": DataLoader(
            stage3_dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            collate_fn=PairBagCollator(max_paths=stage3_retriever_config["max_candidates"]),
        ),
        "stage4": DataLoader(
            stage4_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PairBagCollator(max_paths=stage4_retriever_config["max_candidates"]),
        ),
    }
