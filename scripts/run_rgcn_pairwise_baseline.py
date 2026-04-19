"""Train a pure RGCN pairwise baseline on the built PrimeKG splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.pairwise_features import load_or_build_pairwise_feature_tables
from src.baselines.rgcn_pairwise import (
    RGCNPairwiseClassifier,
    build_neighbor_mean_adjacency,
    build_rgcn_graph_inputs,
)
from src.evaluation.evaluator import Evaluator
from src.training.pipeline import load_artifact_bundle
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/full_fast_cold_drug_rgcn_pairwise.yaml",
    )
    parser.add_argument("--output-name", type=str, default="rgcn_pairwise_baseline")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_device(config: dict[str, Any], explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    runtime_device = str(config.get("runtime", {}).get("device", "cuda")).lower()
    if runtime_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prediction_rows(pair_ids: list[str], labels: torch.Tensor, probabilities: torch.Tensor) -> list[dict[str, Any]]:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
    return [
        {
            "pair_id": pair_id,
            "label": int(label),
            "score": float(prob),
        }
        for pair_id, label, prob in zip(pair_ids, labels_np.tolist(), probs_np.tolist(), strict=True)
    ]


def _pair_tensors(pair_table: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    payload = {
        "drug_indices": pair_table["drug_indices"].to(device),
        "disease_indices": pair_table["disease_indices"].to(device),
        "labels": pair_table["labels"].to(device=device, dtype=torch.float32),
    }
    if "pair_features" in pair_table:
        payload["pair_features"] = pair_table["pair_features"].to(device=device, dtype=torch.float32)
    return payload


def _evaluate_split(
    model: RGCNPairwiseClassifier,
    *,
    node_type_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    split_name: str,
    pair_table: dict[str, Any],
    pair_tensors: dict[str, torch.Tensor],
    evaluator: Evaluator,
    neighbor_adjacency: torch.Tensor | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type=edge_type,
        )
        logits = model.pair_logits(
            node_embeddings=node_embeddings,
            drug_indices=pair_tensors["drug_indices"],
            disease_indices=pair_tensors["disease_indices"],
            pair_features=pair_tensors.get("pair_features"),
            neighbor_adjacency=neighbor_adjacency,
        )
        probabilities = torch.sigmoid(logits)
    rows = _prediction_rows(pair_table["pair_ids"], pair_tensors["labels"], probabilities)
    metrics = evaluator.evaluate_pairs(rows)
    metrics["split"] = split_name
    return metrics, rows


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    rgcn_cfg = deepcopy(config.get("rgcn_pairwise", {}))
    device = _resolve_device(config, args.device)
    seed = int(config["project"].get("seed", 42))
    _set_seed(seed)

    bundle = load_artifact_bundle(config["paths"]["processed_dir"])
    feature_payload = None
    if bool(rgcn_cfg.get("use_pair_features", False)):
        feature_payload = load_or_build_pairwise_feature_tables(
            processed_dir=config["paths"]["processed_dir"],
            graph_data=bundle.graph_data,
            pair_tables=bundle.pair_tables,
        )
        for split_name, feature_tensor in feature_payload["split_features"].items():
            bundle.pair_tables[split_name]["pair_features"] = feature_tensor
    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"] / args.output_name)
    evaluator = Evaluator(output_dir=output_dir, ks=config["evaluation"]["ks"])

    node_type_ids = bundle.graph_data["node_type_ids"].to(device)
    edge_index, edge_type, num_relations = build_rgcn_graph_inputs(
        bundle.graph_data,
        add_reverse_edges=bool(rgcn_cfg.get("add_reverse_edges", True)),
    )
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    neighbor_adjacency = None
    if bool(rgcn_cfg.get("use_neighborhood_pooling", False)):
        neighbor_adjacency = build_neighbor_mean_adjacency(
            edge_index=edge_index,
            num_nodes=int(bundle.graph_data["metadata"]["num_nodes"]),
            device=device,
        )

    train_tensors = _pair_tensors(bundle.pair_tables["train"], device)
    valid_tensors = _pair_tensors(bundle.pair_tables["valid"], device)
    test_tensors = _pair_tensors(bundle.pair_tables["test"], device)

    model = RGCNPairwiseClassifier(
        num_nodes=int(bundle.graph_data["metadata"]["num_nodes"]),
        num_node_types=len(bundle.graph_data["node_type_vocab"]),
        num_relations=num_relations,
        hidden_dim=int(rgcn_cfg.get("hidden_dim", 128)),
        num_layers=int(rgcn_cfg.get("num_layers", 2)),
        num_bases=rgcn_cfg.get("num_bases", 8),
        dropout=float(rgcn_cfg.get("dropout", 0.1)),
        pair_feature_dim=int(feature_payload["feature_dim"]) if bool(rgcn_cfg.get("use_pair_features", False)) else 0,
        pair_feature_hidden_dim=rgcn_cfg.get("pair_feature_hidden_dim"),
        use_neighborhood_pooling=bool(rgcn_cfg.get("use_neighborhood_pooling", False)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(rgcn_cfg.get("lr", 1e-3)),
        weight_decay=float(rgcn_cfg.get("weight_decay", 1e-5)),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = int(rgcn_cfg.get("epochs", 40))
    patience = int(rgcn_cfg.get("patience", 10))
    grad_clip_norm = float(rgcn_cfg.get("grad_clip_norm", 5.0))

    best_valid_auprc = float("-inf")
    best_epoch = -1
    best_state = None
    best_valid_metrics: dict[str, float] | None = None
    train_history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = model.encode(
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type=edge_type,
        )
        train_logits = model.pair_logits(
            node_embeddings=node_embeddings,
            drug_indices=train_tensors["drug_indices"],
            disease_indices=train_tensors["disease_indices"],
            pair_features=train_tensors.get("pair_features"),
            neighbor_adjacency=neighbor_adjacency,
        )
        loss = loss_fn(train_logits, train_tensors["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        valid_metrics, _ = _evaluate_split(
            model,
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            split_name="valid",
            pair_table=bundle.pair_tables["valid"],
            pair_tensors=valid_tensors,
            evaluator=evaluator,
            neighbor_adjacency=neighbor_adjacency,
        )
        train_history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "valid_auroc": float(valid_metrics["auroc"]),
                "valid_auprc": float(valid_metrics["auprc"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "valid_auroc": float(valid_metrics["auroc"]),
                "valid_auprc": float(valid_metrics["auprc"]),
            },
            flush=True,
        )

        if float(valid_metrics["auprc"]) > best_valid_auprc:
            best_valid_auprc = float(valid_metrics["auprc"])
            best_epoch = epoch
            best_valid_metrics = {key: float(value) for key, value in valid_metrics.items() if key != "split"}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("RGCN pairwise baseline did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    valid_metrics, valid_rows = _evaluate_split(
        model,
        node_type_ids=node_type_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        split_name="valid",
        pair_table=bundle.pair_tables["valid"],
        pair_tensors=valid_tensors,
        evaluator=evaluator,
        neighbor_adjacency=neighbor_adjacency,
    )
    test_metrics, test_rows = _evaluate_split(
        model,
        node_type_ids=node_type_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        split_name="test",
        pair_table=bundle.pair_tables["test"],
        pair_tensors=test_tensors,
        evaluator=evaluator,
        neighbor_adjacency=neighbor_adjacency,
    )

    save_json(
        {
            "selected_model": "pure_rgcn_pairwise",
            "device": str(device),
            "best_valid_epoch": best_epoch,
            "best_valid_metrics": best_valid_metrics,
            "valid": {key: float(value) for key, value in valid_metrics.items() if key != "split"},
            "test": {key: float(value) for key, value in test_metrics.items() if key != "split"},
            "train_history": train_history,
            "graph": {
                "num_nodes": int(bundle.graph_data["metadata"]["num_nodes"]),
                "num_edges": int(bundle.graph_data["metadata"]["num_edges"]),
                "num_relations_forward": int(len(bundle.graph_data["relation_vocab"])),
                "num_relations_model": int(num_relations),
                "add_reverse_edges": bool(rgcn_cfg.get("add_reverse_edges", True)),
            },
            "rgcn_pairwise": {
                "hidden_dim": int(rgcn_cfg.get("hidden_dim", 128)),
                "num_layers": int(rgcn_cfg.get("num_layers", 2)),
                "num_bases": rgcn_cfg.get("num_bases", 8),
                "dropout": float(rgcn_cfg.get("dropout", 0.1)),
                "use_pair_features": bool(rgcn_cfg.get("use_pair_features", False)),
                "pair_feature_hidden_dim": rgcn_cfg.get("pair_feature_hidden_dim"),
                "use_neighborhood_pooling": bool(rgcn_cfg.get("use_neighborhood_pooling", False)),
                "lr": float(rgcn_cfg.get("lr", 1e-3)),
                "weight_decay": float(rgcn_cfg.get("weight_decay", 1e-5)),
                "epochs": epochs,
                "patience": patience,
            },
        },
        output_dir / "metrics.json",
    )
    write_csv(valid_rows, output_dir / "per_pair_predictions_valid.csv")
    write_csv(test_rows, output_dir / "per_pair_predictions_test.csv")


if __name__ == "__main__":
    main()
