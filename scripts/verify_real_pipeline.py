"""Verify that Stage 1-4 can consume real artifacts and produce real batches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.graph_encoder import HeteroGraphEncoder
from src.models.pair_model import HierarchicalPairModel
from src.models.path_scorer import PairConditionedPathScorer
from src.training.engine import TrainingEngine
from src.training.pipeline import build_stage_dataloaders, gather_node_states
from src.training.pseudo_label import PseudoRationaleSelector
from src.utils.config import load_experiment_config, prepare_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/tiny_sanity.yaml")
    return parser.parse_args()


def _get_first_batch(loader):
    return next(iter(loader), None)


def main() -> None:
    args = parse_args()
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    payload = build_stage_dataloaders(config, split="train")
    bundle = payload["bundle"]
    device = torch.device(config["runtime"]["device"])

    encoder = HeteroGraphEncoder(
        num_nodes=bundle.graph_data["metadata"]["num_nodes"],
        node_type_vocab=bundle.graph_data["node_type_vocab"],
        relation_vocab=bundle.graph_data["relation_vocab"],
        config=config["model"]["encoder"],
    ).to(device)
    path_scorer = PairConditionedPathScorer(
        hidden_dim=config["model"]["encoder"]["hidden_dim"],
        config=config["model"]["path_scorer"],
    ).to(device)
    pair_model = HierarchicalPairModel(
        hidden_dim=config["model"]["encoder"]["hidden_dim"],
        config=config["model"],
    ).to(device)
    engine = TrainingEngine(config=config, device=device)

    results = {}

    stage1_batch = next(iter(payload["stage1"]))
    stage1_losses = engine.stage1_loss(encoder, stage1_batch)
    results["stage1"] = {key: float(value.detach().cpu().item()) for key, value in stage1_losses.items()}

    node_embeddings = encoder(bundle.graph_data)

    stage2_batch = _get_first_batch(payload["stage2"])
    if stage2_batch is not None:
        drug_emb = node_embeddings[stage2_batch["drug_indices"]]
        disease_emb = node_embeddings[stage2_batch["disease_indices"]]
        pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb)
        positive = stage2_batch["positive_paths"]
        negative = stage2_batch["negative_paths"]
        pos_batch = {
            "node_states": gather_node_states(node_embeddings, positive["node_indices"][:, 0]),
            "relation_ids": positive["relation_ids"][:, 0],
            "node_type_ids": positive["node_type_ids"][:, 0],
            "mask": positive["seq_mask"][:, 0],
        }
        neg_batch = {
            "node_states": gather_node_states(node_embeddings, negative["node_indices"]),
            "relation_ids": negative["relation_ids"],
            "node_type_ids": negative["node_type_ids"],
            "mask": negative["seq_mask"],
        }
        stage2_losses = engine.stage2_loss(path_scorer, pair_repr, pos_batch, neg_batch)
        results["stage2"] = {key: float(value.detach().cpu().item()) for key, value in stage2_losses.items()}
        results["stage2_dataset_size"] = len(payload["stage2"].dataset)
    else:
        results["stage2"] = "empty"

    stage3_batch = _get_first_batch(payload["stage3"])
    if stage3_batch is not None:
        drug_emb = node_embeddings[stage3_batch["drug_indices"]]
        disease_emb = node_embeddings[stage3_batch["disease_indices"]]
        bag = stage3_batch["path_bag"]
        if bag["path_mask"].size(1) > 0:
            flat_states = gather_node_states(node_embeddings, bag["node_indices"]).reshape(
                -1, bag["node_indices"].size(-1), node_embeddings.size(-1)
            )
            flat_rel = bag["relation_ids"].reshape(-1, bag["relation_ids"].size(-1))
            flat_types = bag["node_type_ids"].reshape(-1, bag["node_type_ids"].size(-1))
            flat_seq_mask = bag["seq_mask"].reshape(-1, bag["seq_mask"].size(-1))
            pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb)
            repeated_pair = pair_repr.unsqueeze(1).repeat(1, bag["node_indices"].size(1), 1).reshape(-1, pair_repr.size(-1))
            flat_scores, flat_reprs = path_scorer(
                pair_embedding=repeated_pair,
                node_states=flat_states,
                relation_ids=flat_rel,
                node_type_ids=flat_types,
                mask=flat_seq_mask,
            )
            path_scores = flat_scores.reshape(bag["node_indices"].size(0), bag["node_indices"].size(1))
            path_reprs = flat_reprs.reshape(bag["node_indices"].size(0), bag["node_indices"].size(1), -1)
            stage3_outputs = pair_model(
                drug_embedding=drug_emb,
                disease_embedding=disease_emb,
                path_scores=path_scores,
                path_reprs=path_reprs,
                bag_mask=bag["path_mask"],
            )
            stage3_losses = engine.stage3_loss(stage3_outputs, stage3_batch["labels"])
            results["stage3"] = {key: float(value.detach().cpu().item()) for key, value in stage3_losses.items()}
            results["stage3_batch_nonempty_paths"] = int(bag["path_mask"].sum().item())
        else:
            results["stage3"] = "empty_paths"

    stage4_batch = _get_first_batch(payload["stage4"])
    if stage4_batch is not None and stage4_batch["path_bag"]["path_mask"].size(1) > 0:
        drug_emb = node_embeddings[stage4_batch["drug_indices"]]
        disease_emb = node_embeddings[stage4_batch["disease_indices"]]
        bag = stage4_batch["path_bag"]
        flat_states = gather_node_states(node_embeddings, bag["node_indices"]).reshape(
            -1, bag["node_indices"].size(-1), node_embeddings.size(-1)
        )
        flat_rel = bag["relation_ids"].reshape(-1, bag["relation_ids"].size(-1))
        flat_types = bag["node_type_ids"].reshape(-1, bag["node_type_ids"].size(-1))
        flat_seq_mask = bag["seq_mask"].reshape(-1, bag["seq_mask"].size(-1))
        pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb)
        repeated_pair = pair_repr.unsqueeze(1).repeat(1, bag["node_indices"].size(1), 1).reshape(-1, pair_repr.size(-1))
        flat_scores, flat_reprs = path_scorer(
            pair_embedding=repeated_pair,
            node_states=flat_states,
            relation_ids=flat_rel,
            node_type_ids=flat_types,
            mask=flat_seq_mask,
        )
        path_scores = flat_scores.reshape(bag["node_indices"].size(0), bag["node_indices"].size(1))
        path_reprs = flat_reprs.reshape(bag["node_indices"].size(0), bag["node_indices"].size(1), -1)
        outputs_a = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores,
            path_reprs=path_reprs,
            bag_mask=bag["path_mask"],
        )
        outputs_b = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores + 0.01 * torch.randn_like(path_scores),
            path_reprs=path_reprs,
            bag_mask=bag["path_mask"],
        )
        selector = PseudoRationaleSelector(
            config=config["pseudo"],
            trusted_schema_ids=set(sum(bag["schema_ids"], [])),
        )
        top_schema_ids = [
            (schemas[0] if schemas else "empty_schema")
            for schemas in bag["schema_ids"]
        ]
        selection = selector.select(
            pair_logits=outputs_a["pair_score"],
            path_logits=path_scores,
            top_schema_ids=bag["schema_ids"],
            stability=torch.full_like(outputs_a["pair_score"], 0.9),
            bag_mask=bag["path_mask"],
        )
        stage4_losses = engine.stage4_loss(
            model_outputs_a=outputs_a,
            model_outputs_b=outputs_b,
            pseudo_scores=path_scores.max(dim=1).values,
            pseudo_weights=selection.confidence.detach(),
            labels=stage4_batch["labels"],
        )
        results["stage4"] = {key: float(value.detach().cpu().item()) for key, value in stage4_losses.items()}
        results["stage4_pseudo_summary"] = selection.summary
        results["stage4_dataset_size"] = len(payload["stage4"].dataset)

    print(results)


if __name__ == "__main__":
    main()
