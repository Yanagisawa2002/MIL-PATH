"""Minimal runnable experiment skeleton with smoke forward and output writing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_dataset import build_artifacts
from src.baselines.pairwise_features import load_or_build_pairwise_feature_tables
from src.evaluation.evaluator import Evaluator
from src.models.graph_encoder import HeteroGraphEncoder
from src.models.pair_model import HierarchicalPairModel
from src.models.path_scorer import PairConditionedPathScorer
from src.training.engine import TrainingEngine
from src.training.pseudo_label import PseudoRationaleSelector
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/tiny_sanity.yaml")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def _load_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(row)
            if limit is not None and idx >= limit:
                break
    return rows


def _graph_key_for_raw_id(graph_data: dict[str, Any], raw_id: str) -> str | None:
    if raw_id in graph_data["node_to_idx"]:
        return raw_id
    suffix = f":{raw_id}"
    for key in graph_data["node_to_idx"]:
        if key.endswith(suffix):
            return key
    return None


def _pick_pair_batch(graph_data: dict[str, Any], pair_rows: list[dict[str, Any]], batch_size: int) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    selected_rows = []
    drug_indices = []
    disease_indices = []
    for row in pair_rows:
        drug_key = _graph_key_for_raw_id(graph_data, row["drug_id"])
        disease_key = _graph_key_for_raw_id(graph_data, row["disease_id"])
        if drug_key is None or disease_key is None:
            continue
        selected_rows.append(row)
        drug_indices.append(graph_data["node_to_idx"][drug_key])
        disease_indices.append(graph_data["node_to_idx"][disease_key])
        if len(selected_rows) >= batch_size:
            break
    if selected_rows:
        return (
            selected_rows,
            torch.tensor(drug_indices, dtype=torch.long),
            torch.tensor(disease_indices, dtype=torch.long),
        )

    drug_candidates = [
        idx for idx, node_type in enumerate(graph_data["node_type_by_idx"]) if node_type == "drug"
    ]
    disease_candidates = [
        idx for idx, node_type in enumerate(graph_data["node_type_by_idx"]) if node_type == "disease"
    ]
    if not drug_candidates:
        drug_candidates = list(range(min(batch_size, len(graph_data["idx_to_node"]))))
    if not disease_candidates:
        disease_candidates = list(
            range(min(batch_size, len(graph_data["idx_to_node"])))
        )[::-1]
    synthetic_size = min(batch_size, len(drug_candidates), len(disease_candidates))
    selected_rows = [
        {
            "pair_id": f"synthetic_pair_{idx}",
            "drug_id": graph_data["idx_to_node"][drug_candidates[idx]],
            "disease_id": graph_data["idx_to_node"][disease_candidates[idx]],
            "label": idx % 2,
        }
        for idx in range(synthetic_size)
    ]
    drug_indices = drug_candidates[:synthetic_size]
    disease_indices = disease_candidates[:synthetic_size]
    return (
        selected_rows,
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(disease_indices, dtype=torch.long),
    )


def _pick_pair_batch_from_tables(
    pair_tables: dict[str, Any],
    batch_size: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    payload = pair_tables["train"]
    count = min(batch_size, int(payload["labels"].numel()))
    rows = [
        {
            "pair_id": payload["pair_ids"][idx],
            "label": int(payload["labels"][idx].item()),
        }
        for idx in range(count)
    ]
    return (
        rows,
        payload["drug_indices"][:count].clone().detach(),
        payload["disease_indices"][:count].clone().detach(),
    )


def _random_path_batch(
    batch_size: int,
    num_paths: int,
    seq_len: int,
    hidden_dim: int,
    num_relations: int,
    num_types: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_states = torch.randn(batch_size, num_paths, seq_len, hidden_dim, device=device)
    relation_ids = torch.randint(0, max(1, num_relations), (batch_size, num_paths, seq_len - 1), device=device)
    node_type_ids = torch.randint(0, max(1, num_types), (batch_size, num_paths, seq_len), device=device)
    mask = torch.ones(batch_size, num_paths, seq_len, dtype=torch.bool, device=device)
    return node_states, relation_ids, node_type_ids, mask


def run_smoke_experiment(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(config["paths"]["processed_dir"])
    split_dir = Path(config["paths"]["split_dir"])
    graph_path = processed_dir / "graph_data.pt"
    pair_tables_path = processed_dir / "pair_tables.pt"
    schema_path = processed_dir / "schema_prior.json"

    graph_data = torch.load(graph_path, weights_only=False)
    pair_tables = torch.load(pair_tables_path, weights_only=False) if pair_tables_path.exists() else None
    schema_prior = []
    if schema_path.exists():
        import json

        schema_prior = json.loads(schema_path.read_text(encoding="utf-8")).get("schemas", [])

    if pair_tables is not None and int(pair_tables["train"]["labels"].numel()) > 0:
        direct_pair_feature_cfg = config["model"].get("direct_pair_features", {})
        if direct_pair_feature_cfg.get("enabled", False):
            feature_payload = load_or_build_pairwise_feature_tables(
                processed_dir=processed_dir,
                graph_data=graph_data,
                pair_tables=pair_tables,
            )
            for split_name, feature_tensor in feature_payload["split_features"].items():
                pair_tables[split_name]["pair_features"] = feature_tensor
            pair_tables["feature_dim"] = feature_payload["feature_dim"]
        selected_rows, drug_idx, disease_idx = _pick_pair_batch_from_tables(
            pair_tables=pair_tables,
            batch_size=config["training"]["batch_size"],
        )
    else:
        pair_rows = _load_csv_rows(split_dir / "pairs_train.csv", limit=config["training"]["batch_size"] * 2)
        selected_rows, drug_idx, disease_idx = _pick_pair_batch(
            graph_data=graph_data,
            pair_rows=pair_rows,
            batch_size=min(config["training"]["batch_size"], max(1, len(pair_rows))),
        )
    if not selected_rows:
        raise RuntimeError("No usable pairs found for smoke experiment.")

    device = torch.device(config["runtime"]["device"])
    encoder = HeteroGraphEncoder(
        num_nodes=graph_data["metadata"]["num_nodes"],
        node_type_vocab=graph_data["node_type_vocab"],
        relation_vocab=graph_data["relation_vocab"],
        config=config["model"]["encoder"],
    ).to(device)
    path_scorer = PairConditionedPathScorer(
        hidden_dim=config["model"]["encoder"]["hidden_dim"],
        config=config["model"]["path_scorer"],
    ).to(device)
    model_config = config["model"]
    if model_config.get("direct_pair_features", {}).get("enabled", False):
        model_config = dict(model_config)
        model_config["direct_pair_features"] = dict(model_config["direct_pair_features"])
        model_config["direct_pair_features"]["feature_dim"] = int(pair_tables.get("feature_dim", 0)) if pair_tables is not None else 0
    pair_model = HierarchicalPairModel(
        hidden_dim=config["model"]["encoder"]["hidden_dim"],
        config=model_config,
    ).to(device)
    engine = TrainingEngine(config=config, device=device)

    batch_size = len(selected_rows)
    drug_idx = drug_idx.to(device)
    disease_idx = disease_idx.to(device)
    pair_features = None
    if pair_tables is not None and "pair_features" in pair_tables["train"]:
        pair_features = pair_tables["train"]["pair_features"][:batch_size].to(device)
    node_embeddings = encoder(graph_data)
    drug_embedding = node_embeddings[drug_idx]
    disease_embedding = node_embeddings[disease_idx]
    pair_repr, _ = pair_model.direct_pair(drug_embedding, disease_embedding, pair_features=pair_features)

    num_paths = min(config["retriever"]["max_candidates"], 6)
    seq_len = min(config["retriever"]["max_hops"] + 1, 5)
    num_relations = len(graph_data["relation_vocab"])
    num_types = len(graph_data["node_type_vocab"])
    node_states, relation_ids, node_type_ids, mask = _random_path_batch(
        batch_size=batch_size,
        num_paths=num_paths,
        seq_len=seq_len,
        hidden_dim=config["model"]["encoder"]["hidden_dim"],
        num_relations=num_relations,
        num_types=num_types,
        device=device,
    )

    flat_pair = pair_repr.unsqueeze(1).repeat(1, num_paths, 1).reshape(batch_size * num_paths, -1)
    flat_states = node_states.reshape(batch_size * num_paths, seq_len, -1)
    flat_rel = relation_ids.reshape(batch_size * num_paths, seq_len - 1)
    flat_types = node_type_ids.reshape(batch_size * num_paths, seq_len)
    flat_mask = mask.reshape(batch_size * num_paths, seq_len)
    flat_scores, flat_reprs = path_scorer(
        pair_embedding=flat_pair,
        node_states=flat_states,
        relation_ids=flat_rel,
        node_type_ids=flat_types,
        mask=flat_mask,
    )
    path_scores = flat_scores.reshape(batch_size, num_paths)
    path_reprs = flat_reprs.reshape(batch_size, num_paths, -1)
    model_outputs = pair_model(
        drug_embedding=drug_embedding,
        disease_embedding=disease_embedding,
        path_scores=path_scores,
        path_reprs=path_reprs,
        pair_features=pair_features,
        bag_mask=mask[:, :, 0],
    )

    positive_batch = {
        "node_states": node_states[:, 0],
        "relation_ids": relation_ids[:, 0],
        "node_type_ids": node_type_ids[:, 0],
        "mask": mask[:, 0],
    }
    negative_batch = {
        "node_states": node_states[:, 1:],
        "relation_ids": relation_ids[:, 1:],
        "node_type_ids": node_type_ids[:, 1:],
        "mask": mask[:, 1:],
    }
    labels = torch.tensor([int(row["label"]) for row in selected_rows], dtype=torch.float32, device=device)

    stage2_losses = engine.stage2_loss(path_scorer, pair_repr, positive_batch, negative_batch)
    stage3_losses = engine.stage3_loss(model_outputs, labels)
    perturbed_outputs = pair_model(
        drug_embedding=drug_embedding,
        disease_embedding=disease_embedding,
        path_scores=path_scores + 0.05 * torch.randn_like(path_scores),
        path_reprs=path_reprs,
        pair_features=pair_features,
        bag_mask=mask[:, :, 0],
    )

    selector = PseudoRationaleSelector(
        config=config["pseudo"],
        trusted_schema_ids={schema["schema_id"] for schema in schema_prior[:8]},
    )
    top_schema_ids = [schema["schema_id"] for schema in schema_prior[: max(1, num_paths)]]
    if not top_schema_ids:
        top_schema_ids = [f"schema_{idx}" for idx in range(num_paths)]
    while len(top_schema_ids) < num_paths:
        top_schema_ids.append(top_schema_ids[-1])

    selection = selector.select(
        pair_logits=model_outputs["pair_score"],
        path_logits=path_scores,
        top_schema_ids=top_schema_ids,
        stability=torch.rand(batch_size, device=device) * 0.4 + 0.6,
    )
    stage4_losses = engine.stage4_loss(
        model_outputs_a=model_outputs,
        model_outputs_b=perturbed_outputs,
        pseudo_scores=path_scores.max(dim=1).values,
        pseudo_weights=selection.confidence.detach(),
        labels=labels,
    )

    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"])
    evaluator = Evaluator(output_dir=output_dir, ks=config["evaluation"]["ks"])

    pair_predictions = [
        (
            {
                "pair_id": row["pair_id"],
                "label": int(row["label"]),
                "score": float(model_outputs["pair_score"][idx].detach().cpu().item()),
                "direct_pair_score": float(model_outputs["direct_pair_score"][idx].detach().cpu().item()),
                "path_bag_score": float(model_outputs["path_bag_score"][idx].detach().cpu().item()),
            }
            | {
                key: float(model_outputs[key][idx].detach().cpu().item())
                for key in ("refined_direct_score", "refined_path_score", "fusion_gate", "interaction_delta")
                if key in model_outputs
            }
        )
        for idx, row in enumerate(selected_rows)
    ]
    path_rankings = []
    for batch_idx, row in enumerate(selected_rows):
        best_path = int(path_scores[batch_idx].argmax().item())
        for path_idx in range(num_paths):
            path_rankings.append(
                {
                    "pair_id": row["pair_id"],
                    "path_id": f"{row['pair_id']}::cand_{path_idx}",
                    "score": float(path_scores[batch_idx, path_idx].detach().cpu().item()),
                    "is_gold": 1 if path_idx == best_path else 0,
                }
            )

    pair_metrics = evaluator.evaluate_pairs(pair_predictions)
    path_metrics = evaluator.evaluate_paths(path_rankings)
    explanation_metrics = evaluator.evaluate_explanations(
        full_scores=model_outputs["pair_score"].detach().cpu().numpy(),
        ablated_scores=(model_outputs["pair_score"] - path_scores.max(dim=1).values).detach().cpu().numpy(),
    )
    metrics = {
        "pair": pair_metrics,
        "path": path_metrics,
        "explanation": explanation_metrics,
        "losses": {
            "stage2": {key: float(value.detach().cpu().item()) for key, value in stage2_losses.items()},
            "stage3": {key: float(value.detach().cpu().item()) for key, value in stage3_losses.items()},
            "stage4": {key: float(value.detach().cpu().item()) for key, value in stage4_losses.items()},
        },
    }

    evaluator.write_outputs(
        metrics=metrics,
        pair_rows=pair_predictions,
        path_rows=path_rankings,
        pseudo_summary=selection.summary,
    )
    save_yaml(config, output_dir / "config_snapshot.yaml")
    save_json(selection.summary, output_dir / "pseudo_label_summary.json")
    (output_dir / "ablation_summary.md").write_text(
        "\n".join(
            [
                "# Ablation Summary",
                "",
                "- Supported: no path branch",
                "- Supported: no direct pair branch",
                "- Supported: max vs topk_logsumexp vs attention vs noisy_or",
                "- Supported: no pseudo-rationale",
                "- Supported: no schema prior",
                "- Supported: HGT vs RGCN interface",
            ]
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)

    graph_path = Path(config["paths"]["processed_dir"]) / "graph_data.pt"
    if args.rebuild or not graph_path.exists():
        build_artifacts(config)

    metrics = run_smoke_experiment(config)
    print("Smoke experiment completed.")
    print(metrics)


if __name__ == "__main__":
    main()
