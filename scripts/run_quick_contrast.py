"""Run a quick real-data contrast between DirectOnly and GoldMech."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_dataset import build_artifacts
from src.evaluation.evaluator import Evaluator
from src.models.graph_encoder import HeteroGraphEncoder
from src.models.pair_model import HierarchicalPairModel
from src.models.path_scorer import PairConditionedPathScorer
from src.training.engine import TrainingEngine
from src.training.pipeline import build_candidate_store, build_stage_dataloaders, gather_node_states
from src.training.pseudo_label import PseudoRationaleSelector
from src.utils.config import load_experiment_config, prepare_experiment_config
from src.utils.io import ensure_dir, save_json, write_csv
from src.data.datasets import split_pair_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/full_fast_random.yaml")
    parser.add_argument("--mode", choices=["direct_only", "neighbor_direct", "gold_mech"], required=True)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--stage4-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_optimizer(
    *,
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    lr: float,
    weight_decay: float,
    freeze_encoder: bool = False,
) -> torch.optim.Optimizer:
    params: list[torch.nn.Parameter] = []
    if not freeze_encoder:
        params.extend(list(encoder.parameters()))
    params.extend(list(path_scorer.parameters()))
    params.extend(list(pair_model.parameters()))
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _build_stage3_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    optimizer_cfg = config["training"].get("stage3_optimizer", {})
    scheduler_cfg = optimizer_cfg.get("scheduler", {})
    if not scheduler_cfg.get("enabled", False) or total_epochs <= 0:
        return None
    scheduler_type = scheduler_cfg.get("type", "cosine")
    if scheduler_type != "cosine":
        msg = f"Unsupported stage3 scheduler type: {scheduler_type}"
        raise ValueError(msg)
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1))

    def _lr_lambda(epoch_idx: int) -> float:
        step = epoch_idx + 1
        if warmup_epochs > 0 and step <= warmup_epochs:
            return max(1e-8, step / warmup_epochs)
        progress_den = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (step - warmup_epochs) / progress_den))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _trusted_schema_ids(config: dict[str, Any], bundle: dict[str, Any]) -> set[str]:
    output_root = Path(config["project"]["output_root"]) / config["project"]["name"]
    summary_path = output_root / "stage4_retrieval_summary_train.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return {item["key"] for item in payload.get("top_top1_schemas", [])[:6]}
    return {schema["schema_id"] for schema in bundle.schema_prior.get("schemas", [])[:6]}


def _zero_path_branch(pair_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, hidden_dim = pair_repr.shape
    path_scores = torch.zeros(batch_size, 1, device=pair_repr.device)
    path_reprs = torch.zeros(batch_size, 1, hidden_dim, device=pair_repr.device)
    bag_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=pair_repr.device)
    return path_scores, path_reprs, bag_mask


def _move_to_device(payload: Any, device: torch.device) -> Any:
    if torch.is_tensor(payload):
        return payload.to(device)
    if isinstance(payload, dict):
        return {key: _move_to_device(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(value, device) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_move_to_device(value, device) for value in payload)
    return payload


def _ensure_reverse_adjacency(graph_data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "reverse_adjacency_ptr" in graph_data and "reverse_adjacency_dst" in graph_data:
        return graph_data["reverse_adjacency_ptr"], graph_data["reverse_adjacency_dst"]
    edge_dst = graph_data["edge_dst"]
    edge_src = graph_data["edge_src"]
    num_nodes = int(graph_data["metadata"]["num_nodes"])
    order = torch.argsort(edge_dst, stable=True)
    sorted_dst = edge_dst[order]
    reverse_dst = edge_src[order]
    counts = torch.bincount(sorted_dst, minlength=num_nodes)
    reverse_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    reverse_ptr[1:] = torch.cumsum(counts, dim=0)
    graph_data["reverse_adjacency_ptr"] = reverse_ptr
    graph_data["reverse_adjacency_dst"] = reverse_dst
    return reverse_ptr, reverse_dst


def _pool_endpoint_context(
    endpoint_indices: torch.Tensor,
    node_embeddings: torch.Tensor,
    adjacency_ptr: torch.Tensor,
    adjacency_dst: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = torch.zeros(endpoint_indices.size(0), node_embeddings.size(-1), device=node_embeddings.device)
    counts = torch.zeros(endpoint_indices.size(0), device=node_embeddings.device, dtype=node_embeddings.dtype)
    ptr_cpu = adjacency_ptr.cpu()
    dst_cpu = adjacency_dst.cpu()
    for row_idx, node_idx in enumerate(endpoint_indices.detach().cpu().tolist()):
        start = int(ptr_cpu[node_idx].item())
        end = int(ptr_cpu[node_idx + 1].item())
        if end <= start:
            continue
        neighbor_idx = dst_cpu[start:end].to(node_embeddings.device)
        contexts[row_idx] = node_embeddings.index_select(0, neighbor_idx).mean(dim=0)
        counts[row_idx] = 1.0
    return contexts, counts


def _endpoint_neighborhood_context(
    endpoint_indices: torch.Tensor,
    node_embeddings: torch.Tensor,
    graph_data: dict[str, Any],
) -> torch.Tensor:
    out_ctx, out_mask = _pool_endpoint_context(
        endpoint_indices=endpoint_indices,
        node_embeddings=node_embeddings,
        adjacency_ptr=graph_data["adjacency_ptr"],
        adjacency_dst=graph_data["adjacency_dst"],
    )
    reverse_ptr, reverse_dst = _ensure_reverse_adjacency(graph_data)
    in_ctx, in_mask = _pool_endpoint_context(
        endpoint_indices=endpoint_indices,
        node_embeddings=node_embeddings,
        adjacency_ptr=reverse_ptr,
        adjacency_dst=reverse_dst,
    )
    total = out_ctx + in_ctx
    denom = (out_mask + in_mask).clamp(min=1.0).unsqueeze(-1)
    return total / denom


def _forward_pair_batch(
    batch: dict[str, Any],
    graph_data: dict[str, Any],
    node_embeddings: torch.Tensor,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    direct_only: bool,
    use_neighbor_context: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    drug_emb = node_embeddings[batch["drug_indices"]]
    disease_emb = node_embeddings[batch["disease_indices"]]
    pair_features = batch.get("pair_features")
    if pair_features is not None:
        pair_features = pair_features.to(node_embeddings.device)
    if use_neighbor_context:
        drug_emb = drug_emb + _endpoint_neighborhood_context(batch["drug_indices"], node_embeddings, graph_data)
        disease_emb = disease_emb + _endpoint_neighborhood_context(batch["disease_indices"], node_embeddings, graph_data)
    pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb, pair_features=pair_features)

    if direct_only:
        path_scores, path_reprs, bag_mask = _zero_path_branch(pair_repr)
        outputs = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores,
            path_reprs=path_reprs,
            pair_features=pair_features,
            bag_mask=bag_mask,
        )
        return outputs, path_scores, bag_mask

    bag = batch["path_bag"]
    if bag["path_mask"].size(1) == 0:
        path_scores, path_reprs, bag_mask = _zero_path_branch(pair_repr)
        outputs = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores,
            path_reprs=path_reprs,
            pair_features=pair_features,
            bag_mask=bag_mask,
        )
        return outputs, path_scores, bag_mask

    flat_states = gather_node_states(node_embeddings, bag["node_indices"]).reshape(
        -1, bag["node_indices"].size(-1), node_embeddings.size(-1)
    )
    flat_rel = bag["relation_ids"].reshape(-1, bag["relation_ids"].size(-1))
    flat_types = bag["node_type_ids"].reshape(-1, bag["node_type_ids"].size(-1))
    flat_seq_mask = bag["seq_mask"].reshape(-1, bag["seq_mask"].size(-1))
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
    outputs = pair_model(
        drug_embedding=drug_emb,
        disease_embedding=disease_emb,
        path_scores=path_scores,
        path_reprs=path_reprs,
        pair_features=pair_features,
        bag_mask=bag["path_mask"],
    )
    return outputs, path_scores, bag["path_mask"]


def _train_stage1(
    loaders: dict[str, Any],
    encoder: HeteroGraphEncoder,
    optimizer: torch.optim.Optimizer,
    engine: TrainingEngine,
    grad_clip_norm: float,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    encoder.train()
    total_steps = len(loaders["stage1"])
    for step, batch in enumerate(loaders["stage1"]):
        print({"heartbeat": "stage1_pre", "step": step + 1, "total_steps": total_steps}, flush=True)
        batch = _move_to_device(batch, engine.device)
        optimizer.zero_grad(set_to_none=True)
        losses = engine.stage1_loss(encoder, batch)
        losses["total"].backward()
        clip_grad_norm_(encoder.parameters(), grad_clip_norm)
        optimizer.step()
        history.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
        print({"heartbeat": "stage1_post", "step": step + 1, "total_steps": total_steps, "loss": history[-1]["total"]}, flush=True)
    return history


def _train_stage2(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    optimizer: torch.optim.Optimizer,
    engine: TrainingEngine,
    grad_clip_norm: float,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    encoder.train()
    path_scorer.train()
    pair_model.train()
    total_steps = len(loaders["stage2"])
    for step, batch in enumerate(loaders["stage2"]):
        batch = _move_to_device(batch, engine.device)
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = encoder(bundle.graph_data)
        drug_emb = node_embeddings[batch["drug_indices"]]
        disease_emb = node_embeddings[batch["disease_indices"]]
        pair_features = batch.get("pair_features")
        if pair_features is not None:
            pair_features = pair_features.to(node_embeddings.device)
        pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb, pair_features=pair_features)
        positive = batch["positive_paths"]
        negative = batch["negative_paths"]
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
        losses = engine.stage2_loss(path_scorer, pair_repr, pos_batch, neg_batch)
        losses["total"].backward()
        clip_grad_norm_(
            list(encoder.parameters()) + list(path_scorer.parameters()) + list(pair_model.direct_pair.parameters()),
            grad_clip_norm,
        )
        optimizer.step()
        history.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
        if (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(
                {"heartbeat": "stage2", "step": step + 1, "total_steps": total_steps, "loss": history[-1]["total"]},
                flush=True,
            )
    return history


def _train_stage3(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    optimizer: torch.optim.Optimizer,
    engine: TrainingEngine,
    grad_clip_norm: float,
    direct_only: bool,
    use_neighbor_context: bool = False,
    freeze_encoder: bool = False,
    cached_node_embeddings: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if cached_node_embeddings is not None:
        encoder.eval()
        frozen_node_embeddings = cached_node_embeddings
    elif freeze_encoder:
        encoder.eval()
        with torch.no_grad():
            frozen_node_embeddings = encoder(bundle.graph_data).detach()
    else:
        encoder.train()
        frozen_node_embeddings = None
    path_scorer.train()
    pair_model.train()
    total_steps = len(loaders["stage3"])
    for step, batch in enumerate(loaders["stage3"]):
        batch = _move_to_device(batch, engine.device)
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = frozen_node_embeddings if frozen_node_embeddings is not None else encoder(bundle.graph_data)
        outputs, _, _ = _forward_pair_batch(
            batch=batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=direct_only,
            use_neighbor_context=use_neighbor_context,
        )
        losses = engine.stage3_loss(outputs, batch["labels"])
        losses["total"].backward()
        clip_grad_norm_(
            list(encoder.parameters()) + list(path_scorer.parameters()) + list(pair_model.parameters()),
            grad_clip_norm,
        )
        optimizer.step()
        history.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
        if (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(
                {"heartbeat": "stage3", "step": step + 1, "total_steps": total_steps, "loss": history[-1]["total"]},
                flush=True,
            )
    return history


def _train_stage4(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    optimizer: torch.optim.Optimizer,
    engine: TrainingEngine,
    grad_clip_norm: float,
    pseudo_config: dict[str, Any],
    trusted_schema_ids: set[str],
    anchor_weight: float,
    freeze_encoder: bool = False,
    cached_node_embeddings: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if cached_node_embeddings is not None:
        encoder.eval()
        frozen_node_embeddings = cached_node_embeddings
    elif freeze_encoder:
        encoder.eval()
        with torch.no_grad():
            frozen_node_embeddings = encoder(bundle.graph_data).detach()
    else:
        encoder.train()
        frozen_node_embeddings = None
    path_scorer.train()
    pair_model.train()
    selector = PseudoRationaleSelector(config=pseudo_config, trusted_schema_ids=trusted_schema_ids)
    anchor_iter = iter(loaders["stage3"])

    total_steps = len(loaders["stage4"])
    for step, batch in enumerate(loaders["stage4"]):
        batch = _move_to_device(batch, engine.device)
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = frozen_node_embeddings if frozen_node_embeddings is not None else encoder(bundle.graph_data)
        outputs_a, path_scores_a, bag_mask = _forward_pair_batch(
            batch=batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=False,
        )
        outputs_b, _, _ = _forward_pair_batch(
            batch=batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=False,
        )

        selection = selector.select(
            pair_logits=outputs_a["pair_score"].detach(),
            path_logits=path_scores_a.detach(),
            top_schema_ids=batch["path_bag"]["schema_ids"],
            stability=torch.full_like(outputs_a["pair_score"], 0.9),
            bag_mask=bag_mask,
        )

        masked_scores = path_scores_a.masked_fill(~bag_mask, float("-inf"))
        pseudo_scores = masked_scores.max(dim=1).values
        pseudo_scores = torch.where(
            bag_mask.any(dim=1),
            pseudo_scores,
            torch.zeros_like(pseudo_scores),
        )
        pseudo_weights = selection.confidence.detach() * selection.accepted_mask.float().detach()

        losses = engine.stage4_loss(
            model_outputs_a=outputs_a,
            model_outputs_b=outputs_b,
            pseudo_scores=pseudo_scores,
            pseudo_weights=pseudo_weights,
            labels=batch["labels"],
        )

        try:
            anchor_batch = next(anchor_iter)
        except StopIteration:
            anchor_iter = iter(loaders["stage3"])
            anchor_batch = next(anchor_iter)
        anchor_batch = _move_to_device(anchor_batch, engine.device)
        anchor_outputs, _, _ = _forward_pair_batch(
            batch=anchor_batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=False,
        )
        anchor_losses = engine.stage3_loss(anchor_outputs, anchor_batch["labels"])

        total_loss = losses["total"] + anchor_weight * anchor_losses["total"]
        total_loss.backward()
        clip_grad_norm_(
            list(encoder.parameters()) + list(path_scorer.parameters()) + list(pair_model.parameters()),
            grad_clip_norm,
        )
        optimizer.step()
        history.append(
            {
                key: float((total_loss.detach().cpu().item() if key == "total" else value.detach().cpu().item()))
                for key, value in losses.items()
            }
            | {
                "anchor_pair_cls": float(anchor_losses["pair_cls"].detach().cpu().item()),
                "accepted_rate": float(selection.accepted_mask.float().mean().item()),
                "mean_confidence": float(selection.confidence.mean().item()),
                "num_nonempty_bags": float(selection.summary["num_nonempty_bags"]),
            }
        )
        if (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(
                {"heartbeat": "stage4", "step": step + 1, "total_steps": total_steps, "loss": history[-1]["total"]},
                flush=True,
            )
    return history


@torch.no_grad()
def _predict_pair_rows(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    direct_only: bool,
    use_neighbor_context: bool = False,
    cached_node_embeddings: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    encoder.eval()
    path_scorer.eval()
    pair_model.eval()
    node_embeddings = cached_node_embeddings if cached_node_embeddings is not None else encoder(bundle.graph_data)
    pair_rows: list[dict[str, Any]] = []
    for batch in loaders["stage3"]:
        batch = _move_to_device(batch, node_embeddings.device)
        outputs, _, bag_mask = _forward_pair_batch(
            batch=batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=direct_only,
            use_neighbor_context=use_neighbor_context,
        )
        for idx, pair_id in enumerate(batch["pair_ids"]):
            row = {
                "pair_id": pair_id,
                "label": int(batch["labels"][idx].item()),
                "score": float(outputs["pair_score"][idx].detach().cpu().item()),
                "direct_pair_score": float(outputs["direct_pair_score"][idx].detach().cpu().item()),
                "path_bag_score": float(outputs["path_bag_score"][idx].detach().cpu().item()),
                "bag_nonempty": int(bool(bag_mask[idx].any().item())),
                "bag_size": int(bag_mask[idx].sum().item()),
            }
            if "has_gold_rationale" in batch:
                row["has_gold_rationale"] = int(batch["has_gold_rationale"][idx].item())
                row["num_gold_paths"] = int(batch["num_gold_paths"][idx].item())
            for key in ("refined_direct_score", "refined_path_score", "fusion_gate", "interaction_delta"):
                if key in outputs:
                    row[key] = float(outputs[key][idx].detach().cpu().item())
            pair_rows.append(row)
    return pair_rows


@torch.no_grad()
def _evaluate_pairs(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    direct_only: bool,
    use_neighbor_context: bool,
    output_dir: Path,
    split_name: str,
    ks: list[int],
    cached_node_embeddings: torch.Tensor | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pair_rows = _predict_pair_rows(
        loaders=loaders,
        bundle=bundle,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        cached_node_embeddings=cached_node_embeddings,
    )
    evaluator = Evaluator(output_dir=output_dir, ks=ks)
    metrics = evaluator.evaluate_pairs(pair_rows)
    write_csv(pair_rows, output_dir / f"per_pair_predictions_{split_name}.csv")
    return metrics, pair_rows


def _score_valid_epoch(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    direct_only: bool,
    use_neighbor_context: bool,
    ks: list[int],
    cached_node_embeddings: torch.Tensor | None = None,
) -> dict[str, float]:
    pair_rows = _predict_pair_rows(
        loaders=loaders,
        bundle=bundle,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        cached_node_embeddings=cached_node_embeddings,
    )
    evaluator = Evaluator(output_dir=Path.cwd() / "outputs" / "_tmp_eval", ks=ks)
    return evaluator.evaluate_pairs(pair_rows)


def _evaluate_pair_strata(
    pair_rows: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    evaluator = Evaluator(output_dir=Path.cwd() / "outputs" / "_tmp_eval_strata", ks=ks)
    strata = {
        "nonempty_path_bag": [row for row in pair_rows if int(row.get("bag_nonempty", 0)) == 1],
        "empty_path_bag": [row for row in pair_rows if int(row.get("bag_nonempty", 0)) == 0],
    }
    summary: dict[str, Any] = {}
    for name, rows in strata.items():
        label_sum = int(sum(int(row["label"]) for row in rows))
        block: dict[str, Any] = {
            "count": len(rows),
            "num_positive": label_sum,
            "num_negative": int(len(rows) - label_sum),
        }
        if rows and 0 < label_sum < len(rows):
            block.update(evaluator.evaluate_pairs(rows))
        else:
            for metric_name in (
                "auroc",
                "auprc",
                *[f"recall@{k}" for k in ks],
                *[f"hits@{k}" for k in ks],
            ):
                block[metric_name] = float("nan")
        summary[name] = block
    return summary


def _snapshot_model_state(
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
) -> dict[str, Any]:
    return {
        "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
        "path_scorer": {key: value.detach().cpu().clone() for key, value in path_scorer.state_dict().items()},
        "pair_model": {key: value.detach().cpu().clone() for key, value in pair_model.state_dict().items()},
    }


def _restore_model_state(
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    snapshot: dict[str, Any],
) -> None:
    encoder.load_state_dict(snapshot["encoder"])
    path_scorer.load_state_dict(snapshot["path_scorer"])
    pair_model.load_state_dict(snapshot["pair_model"])


def _pad_path_group(path_group: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    max_seq = max(int(record["node_indices"].numel()) for record in path_group)
    num_paths = len(path_group)
    node_indices = torch.zeros(num_paths, max_seq, dtype=torch.long, device=device)
    node_type_ids = torch.zeros(num_paths, max_seq, dtype=torch.long, device=device)
    relation_ids = torch.zeros(num_paths, max(max_seq - 1, 1), dtype=torch.long, device=device)
    seq_mask = torch.zeros(num_paths, max_seq, dtype=torch.bool, device=device)
    for path_idx, record in enumerate(path_group):
        seq_len = int(record["node_indices"].numel())
        rel_len = int(record["relation_ids"].numel())
        node_indices[path_idx, :seq_len] = record["node_indices"].to(device)
        node_type_ids[path_idx, :seq_len] = record["node_type_ids"].to(device)
        if rel_len > 0:
            relation_ids[path_idx, :rel_len] = record["relation_ids"].to(device)
        seq_mask[path_idx, :seq_len] = True
    return {
        "node_indices": node_indices,
        "node_type_ids": node_type_ids,
        "relation_ids": relation_ids,
        "seq_mask": seq_mask,
    }


def _path_signature(record: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(int(item) for item in record["node_indices"].tolist()),
        tuple(int(item) for item in record["relation_ids"].tolist()),
    )


def _stable_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _clone_eval_record(
    record: dict[str, Any],
    *,
    pair_id: str,
    source: str,
    path_id_suffix: str,
) -> dict[str, Any]:
    return {
        "path_id": f"{pair_id}::{path_id_suffix}::{record['path_id']}",
        "pair_id": pair_id,
        "schema_id": record["schema_id"],
        "node_indices": record["node_indices"].clone(),
        "node_type_ids": record["node_type_ids"].clone(),
        "relation_ids": record["relation_ids"].clone(),
        "hop_count": int(record["hop_count"]),
        "path_source": source,
        "confidence": float(record.get("confidence", 0.0)),
        "resolved": True,
    }


def _build_path_eval_context(bundle: Any, split_name: str) -> dict[str, Any]:
    type_to_nodes: dict[int, list[int]] = {}
    for node_idx, type_id in enumerate(bundle.graph_data["node_type_ids"].tolist()):
        type_to_nodes.setdefault(int(type_id), []).append(int(node_idx))

    schema_pool: dict[str, list[dict[str, Any]]] = {}
    hop_pool: dict[int, list[dict[str, Any]]] = {}
    for record in bundle.path_tensor_store["paths"].values():
        if record["split"] != split_name or not record["resolved"]:
            continue
        schema_pool.setdefault(record["schema_id"], []).append(record)
        hop_pool.setdefault(int(record["hop_count"]), []).append(record)

    for pool in schema_pool.values():
        pool.sort(key=lambda item: item["path_id"])
    for pool in hop_pool.values():
        pool.sort(key=lambda item: item["path_id"])

    return {
        "type_to_nodes": type_to_nodes,
        "schema_pool": schema_pool,
        "hop_pool": hop_pool,
    }


def _generate_corruption_candidates(
    gold_record: dict[str, Any],
    pair_id: str,
    eval_context: dict[str, Any],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0:
        return []
    node_indices = gold_record["node_indices"]
    if int(node_indices.numel()) <= 2:
        return []

    original_nodes = {int(item) for item in node_indices.tolist()}
    internal_positions = list(range(1, int(node_indices.numel()) - 1))
    if not internal_positions:
        return []

    corruption_records: list[dict[str, Any]] = []
    for candidate_idx in range(max_candidates):
        position = internal_positions[candidate_idx % len(internal_positions)]
        type_id = int(gold_record["node_type_ids"][position].item())
        candidate_pool = eval_context["type_to_nodes"].get(type_id, [])
        if len(candidate_pool) <= 1:
            continue
        start_offset = _stable_index(
            f"{pair_id}::{gold_record['path_id']}::{position}::{candidate_idx}",
            len(candidate_pool),
        )
        replacement = None
        for attempt in range(len(candidate_pool)):
            node_idx = candidate_pool[(start_offset + attempt) % len(candidate_pool)]
            if node_idx not in original_nodes:
                replacement = int(node_idx)
                break
        if replacement is None:
            continue
        corrupted_nodes = gold_record["node_indices"].clone()
        corrupted_nodes[position] = replacement
        corruption_records.append(
            {
                "path_id": f"{pair_id}::corrupt::{gold_record['path_id']}::{candidate_idx}",
                "pair_id": pair_id,
                "schema_id": gold_record["schema_id"],
                "node_indices": corrupted_nodes,
                "node_type_ids": gold_record["node_type_ids"].clone(),
                "relation_ids": gold_record["relation_ids"].clone(),
                "hop_count": int(gold_record["hop_count"]),
                "path_source": "corrupt_internal",
                "confidence": 0.0,
                "resolved": True,
            }
        )
    return corruption_records


def _gather_cross_pair_candidates(
    *,
    target_pair_id: str,
    pool: list[dict[str, Any]],
    source_label: str,
    limit: int,
    seen_signatures: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for record in pool:
        if record["pair_id"] == target_pair_id:
            continue
        signature = _path_signature(record)
        if signature in seen_signatures:
            continue
        selected.append(
            _clone_eval_record(
                record,
                pair_id=target_pair_id,
                source=source_label,
                path_id_suffix=source_label,
            )
        )
        seen_signatures.add(signature)
        if len(selected) >= limit:
            break
    return selected


def _build_eval_path_group(
    *,
    pair_id: str,
    drug_id: str,
    disease_id: str,
    candidate_store: Any,
    bundle: Any,
    eval_context: dict[str, Any],
    hard_cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if hard_cfg is None:
        hard_cfg = {}
    use_cached_only = bool(hard_cfg.get("use_cached_only", True))
    use_retrieved_candidates = bool(hard_cfg.get("use_retrieved_candidates", True))
    num_corruptions_per_gold = int(hard_cfg.get("num_corruptions_per_gold", 0))
    num_cross_pair_same_schema = int(hard_cfg.get("num_cross_pair_same_schema", 0))
    num_cross_pair_same_hop = int(hard_cfg.get("num_cross_pair_same_hop", 0))

    path_group: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    if use_retrieved_candidates:
        retrieved_group = candidate_store.get_pair_paths(
            pair_id=pair_id,
            drug_id=drug_id,
            disease_id=disease_id,
            include_gold=False,
            retrieve_on_miss=not use_cached_only,
        )
        for record in retrieved_group:
            signature = _path_signature(record)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            path_group.append(record)

    gold_records = [
        bundle.path_tensor_store["paths"][path_id]
        for path_id in bundle.path_tensor_store["pair_to_path_ids"].get(pair_id, [])
        if bundle.path_tensor_store["paths"][path_id]["resolved"]
    ]
    for gold_record in gold_records:
        signature = _path_signature(gold_record)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        path_group.append(gold_record)

    if not hard_cfg.get("enabled", False):
        return path_group

    for gold_record in gold_records:
        for record in _generate_corruption_candidates(
            gold_record,
            pair_id=pair_id,
            eval_context=eval_context,
            max_candidates=num_corruptions_per_gold,
        ):
            signature = _path_signature(record)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            path_group.append(record)

        same_schema_pool = eval_context["schema_pool"].get(gold_record["schema_id"], [])
        path_group.extend(
            _gather_cross_pair_candidates(
                target_pair_id=pair_id,
                pool=same_schema_pool,
                source_label="cross_pair_same_schema",
                limit=num_cross_pair_same_schema,
                seen_signatures=seen_signatures,
            )
        )
        same_hop_pool = eval_context["hop_pool"].get(int(gold_record["hop_count"]), [])
        path_group.extend(
            _gather_cross_pair_candidates(
                target_pair_id=pair_id,
                pool=same_hop_pool,
                source_label="cross_pair_same_hop",
                limit=num_cross_pair_same_hop,
                seen_signatures=seen_signatures,
            )
        )
    return path_group


def _summarize_path_rows(path_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in path_rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    if not grouped:
        return {
            "pairs": 0,
            "mean_bag_size": 0.0,
            "median_bag_size": 0.0,
            "max_bag_size": 0,
            "pairs_with_non_gold": 0,
            "pairs_with_non_gold_ratio": 0.0,
            "mean_non_gold_per_pair": 0.0,
            "top1_gold_ratio": 0.0,
            "path_source_counts": {},
        }

    import statistics
    from collections import Counter

    bag_sizes = [len(rows) for rows in grouped.values()]
    non_gold_counts = [sum(1 for row in rows if not row["is_gold"]) for rows in grouped.values()]
    top1_gold = 0
    source_counts: Counter[str] = Counter()
    for rows in grouped.values():
        for row in rows:
            source_counts[row["path_source"]] += 1
        top_row = max(rows, key=lambda item: item["score"])
        top1_gold += int(bool(top_row["is_gold"]))

    pairs_with_non_gold = sum(1 for count in non_gold_counts if count > 0)
    return {
        "pairs": len(grouped),
        "mean_bag_size": float(statistics.mean(bag_sizes)),
        "median_bag_size": float(statistics.median(bag_sizes)),
        "max_bag_size": int(max(bag_sizes)),
        "pairs_with_non_gold": int(pairs_with_non_gold),
        "pairs_with_non_gold_ratio": float(pairs_with_non_gold / len(grouped)),
        "mean_non_gold_per_pair": float(statistics.mean(non_gold_counts)),
        "top1_gold_ratio": float(top1_gold / len(grouped)),
        "path_source_counts": dict(source_counts),
    }


@torch.no_grad()
def _evaluate_path_and_explanations(
    split_name: str,
    bundle: Any,
    config: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    direct_only: bool,
    use_neighbor_context: bool,
    output_dir: Path,
    ks: list[int],
    cached_node_embeddings: torch.Tensor | None = None,
    benchmark_name: str = "path",
    hard_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    if direct_only:
        nan_metrics = {"mrr": float("nan")}
        for k in ks:
            nan_metrics[f"hits@{k}"] = float("nan")
            nan_metrics[f"gold_recall@{k}"] = float("nan")
        return nan_metrics, {"faithfulness_drop": float("nan")}, [], {"pairs": 0}

    encoder.eval()
    path_scorer.eval()
    pair_model.eval()
    node_embeddings = cached_node_embeddings if cached_node_embeddings is not None else encoder(bundle.graph_data)
    candidate_store = build_candidate_store(config, bundle, profile="stage3_cached")
    eval_context = _build_path_eval_context(bundle, split_name)
    evaluator = Evaluator(output_dir=output_dir, ks=ks)
    split_table = bundle.pair_tables[split_name]

    path_rows: list[dict[str, Any]] = []
    full_scores: list[float] = []
    top_ablated_scores: list[float] = []
    random_full_scores: list[float] = []
    random_ablated_scores: list[float] = []

    for idx, pair_id in enumerate(split_table["pair_ids"]):
        if not bool(split_table["has_gold_rationale"][idx].item()):
            continue
        drug_id, disease_id = split_pair_id(pair_id)
        path_group = _build_eval_path_group(
            pair_id=pair_id,
            drug_id=drug_id,
            disease_id=disease_id,
            candidate_store=candidate_store,
            bundle=bundle,
            eval_context=eval_context,
            hard_cfg=hard_cfg,
        )
        if not path_group:
            continue
        if not any(record.get("path_source") == "gold" for record in path_group):
            continue

        drug_idx = split_table["drug_indices"][idx].unsqueeze(0).to(node_embeddings.device)
        disease_idx = split_table["disease_indices"][idx].unsqueeze(0).to(node_embeddings.device)
        pair_features = None
        if "pair_features" in split_table:
            pair_features = split_table["pair_features"][idx].unsqueeze(0).to(node_embeddings.device)

        drug_emb = node_embeddings[drug_idx]
        disease_emb = node_embeddings[disease_idx]
        if use_neighbor_context:
            drug_emb = drug_emb + _endpoint_neighborhood_context(drug_idx, node_embeddings, bundle.graph_data)
            disease_emb = disease_emb + _endpoint_neighborhood_context(disease_idx, node_embeddings, bundle.graph_data)
        pair_repr, _ = pair_model.direct_pair(drug_emb, disease_emb, pair_features=pair_features)

        padded = _pad_path_group(path_group, device=node_embeddings.device)
        node_states = gather_node_states(node_embeddings, padded["node_indices"])
        repeated_pair = pair_repr.expand(len(path_group), -1)
        path_scores, path_reprs = path_scorer(
            pair_embedding=repeated_pair,
            node_states=node_states,
            relation_ids=padded["relation_ids"],
            node_type_ids=padded["node_type_ids"],
            mask=padded["seq_mask"],
        )
        path_scores = path_scores.reshape(1, -1)
        path_reprs = path_reprs.reshape(1, len(path_group), -1)
        bag_mask = torch.ones(1, len(path_group), dtype=torch.bool, device=node_embeddings.device)

        outputs = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores,
            path_reprs=path_reprs,
            pair_features=pair_features,
            bag_mask=bag_mask,
        )
        top_idx = int(path_scores.squeeze(0).argmax().item())
        top_ablated_mask = bag_mask.clone()
        top_ablated_mask[0, top_idx] = False
        top_ablated_outputs = pair_model(
            drug_embedding=drug_emb,
            disease_embedding=disease_emb,
            path_scores=path_scores,
            path_reprs=path_reprs,
            pair_features=pair_features,
            bag_mask=top_ablated_mask,
        )
        full_scores.append(float(outputs["pair_score"][0].detach().cpu().item()))
        top_ablated_scores.append(float(top_ablated_outputs["pair_score"][0].detach().cpu().item()))

        if int(bag_mask.sum().item()) >= 2:
            candidate_indices = [path_idx for path_idx in range(len(path_group)) if path_idx != top_idx]
            random_offset = _stable_index(f"{benchmark_name}::{split_name}::{pair_id}", len(candidate_indices))
            random_idx = candidate_indices[random_offset]
            random_ablated_mask = bag_mask.clone()
            random_ablated_mask[0, random_idx] = False
            random_ablated_outputs = pair_model(
                drug_embedding=drug_emb,
                disease_embedding=disease_emb,
                path_scores=path_scores,
                path_reprs=path_reprs,
                pair_features=pair_features,
                bag_mask=random_ablated_mask,
            )
            random_full_scores.append(float(outputs["pair_score"][0].detach().cpu().item()))
            random_ablated_scores.append(float(random_ablated_outputs["pair_score"][0].detach().cpu().item()))

        for path_idx, record in enumerate(path_group):
            path_rows.append(
                {
                    "pair_id": pair_id,
                    "path_id": record["path_id"],
                    "schema_id": record["schema_id"],
                    "score": float(path_scores[0, path_idx].detach().cpu().item()),
                    "is_gold": int(record.get("path_source") == "gold"),
                    "path_source": record.get("path_source", "unknown"),
                }
            )

    path_metrics = evaluator.evaluate_paths(path_rows)
    explanation_metrics = evaluator.evaluate_explanations(
        full_scores=np.asarray(full_scores, dtype=np.float64),
        ablated_scores=np.asarray(top_ablated_scores, dtype=np.float64),
    ) if full_scores else {"faithfulness_drop": float("nan")}
    explanation_metrics["faithfulness_drop_top_path"] = explanation_metrics["faithfulness_drop"]
    if random_ablated_scores:
        random_drop = evaluator.evaluate_explanations(
            full_scores=np.asarray(random_full_scores, dtype=np.float64),
            ablated_scores=np.asarray(random_ablated_scores, dtype=np.float64),
        )["faithfulness_drop"]
        explanation_metrics["faithfulness_drop_random_non_top"] = random_drop
        explanation_metrics["faithfulness_gap_top_minus_random"] = (
            explanation_metrics["faithfulness_drop"] - random_drop
        )
        explanation_metrics["num_random_eligible_pairs"] = len(random_ablated_scores)
    else:
        explanation_metrics["faithfulness_drop_random_non_top"] = float("nan")
        explanation_metrics["faithfulness_gap_top_minus_random"] = float("nan")
        explanation_metrics["num_random_eligible_pairs"] = 0
    diagnostics = _summarize_path_rows(path_rows)
    write_csv(path_rows, output_dir / f"per_{benchmark_name}_ranking_{split_name}.csv")
    return path_metrics, explanation_metrics, path_rows, diagnostics


@torch.no_grad()
def _summarize_stage4(
    loaders: dict[str, Any],
    bundle: dict[str, Any],
    encoder: HeteroGraphEncoder,
    path_scorer: PairConditionedPathScorer,
    pair_model: HierarchicalPairModel,
    pseudo_config: dict[str, Any],
    trusted_schema_ids: set[str],
    direct_only: bool,
    use_neighbor_context: bool = False,
    cached_node_embeddings: torch.Tensor | None = None,
) -> dict[str, Any]:
    encoder.eval()
    path_scorer.eval()
    pair_model.eval()
    selector = PseudoRationaleSelector(config=pseudo_config, trusted_schema_ids=trusted_schema_ids)
    node_embeddings = cached_node_embeddings if cached_node_embeddings is not None else encoder(bundle.graph_data)

    total_pairs = 0
    total_accepted = 0
    total_nonempty = 0
    total_singleton = 0
    confidence_sum = 0.0

    for batch in loaders["stage4"]:
        batch = _move_to_device(batch, node_embeddings.device)
        outputs, path_scores, bag_mask = _forward_pair_batch(
            batch=batch,
            graph_data=bundle.graph_data,
            node_embeddings=node_embeddings,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=direct_only,
            use_neighbor_context=use_neighbor_context,
        )
        selection = selector.select(
            pair_logits=outputs["pair_score"],
            path_logits=path_scores,
            top_schema_ids=batch["path_bag"]["schema_ids"],
            stability=torch.full_like(outputs["pair_score"], 0.9),
            bag_mask=bag_mask,
        )
        total_pairs += selection.summary["num_pairs"]
        total_accepted += selection.summary["num_accepted"]
        total_nonempty += selection.summary["num_nonempty_bags"]
        total_singleton += selection.summary["num_singleton_bags"]
        confidence_sum += float(selection.confidence.sum().item())

    mean_confidence = confidence_sum / max(1, total_pairs)
    return {
        "num_pairs": total_pairs,
        "num_accepted": total_accepted,
        "accept_rate": float(total_accepted / max(1, total_pairs)),
        "num_nonempty_bags": total_nonempty,
        "num_singleton_bags": total_singleton,
        "mean_confidence": mean_confidence,
        "trusted_schema_count": len(trusted_schema_ids),
    }


def main() -> None:
    args = parse_args()
    print({"status": "starting", "config": args.config, "mode": args.mode}, flush=True)
    config = prepare_experiment_config(load_experiment_config(args.config), repo_root=REPO_ROOT)
    _set_global_seed(int(config["project"].get("seed", 42)))
    if args.rebuild or not (Path(config["paths"]["processed_dir"]) / "graph_data.pt").exists():
        build_artifacts(config)

    direct_only = args.mode in {"direct_only", "neighbor_direct"}
    use_neighbor_context = args.mode == "neighbor_direct"
    output_name = args.output_name or args.mode
    output_dir = ensure_dir(Path(config["project"]["output_root"]) / config["project"]["name"] / output_name)

    print({"status": "building_dataloaders"}, flush=True)
    train_loaders = build_stage_dataloaders(config, split="train")
    valid_loaders = build_stage_dataloaders(config, split="valid")
    test_loaders = build_stage_dataloaders(config, split="test")
    bundle = train_loaders["bundle"]
    print({"status": "dataloaders_ready", "train_pairs": int(bundle.pair_tables["train"]["labels"].numel())}, flush=True)

    model_config = deepcopy(config["model"])
    if direct_only:
        model_config["aggregator"]["alpha"] = 1.0
    direct_pair_feature_cfg = model_config.get("direct_pair_features", {})
    if direct_pair_feature_cfg.get("enabled", False):
        model_config["direct_pair_features"]["feature_dim"] = int(bundle.pair_tables.get("feature_dim", 0))

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
        config=model_config,
    ).to(device)
    print({"status": "models_ready"}, flush=True)

    optimizer = _build_optimizer(
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        freeze_encoder=False,
    )
    engine = TrainingEngine(config=config, device=device)
    grad_clip_norm = float(config["training"]["grad_clip_norm"])

    training_history: dict[str, list[dict[str, float]]] = {"stage1": [], "stage2": [], "stage3": [], "stage4": []}

    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else int(config["training"].get("stage1_epochs", 0))
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else int(config["training"].get("stage2_epochs", 0))
    stage3_epochs = args.stage3_epochs if args.stage3_epochs is not None else int(config["training"].get("stage3_epochs", 0))
    stage4_epochs = args.stage4_epochs if args.stage4_epochs is not None else int(config["training"].get("stage4_epochs", 0))
    stage4_anchor_weight = float(config["training"].get("stage4_anchor_weight", 0.25))
    freeze_encoder_stage3 = bool(config["training"].get("freeze_encoder_stage3", False))
    freeze_encoder_stage4 = bool(config["training"].get("freeze_encoder_stage4", freeze_encoder_stage3))
    early_stop_patience = args.early_stop_patience
    trusted_schema_ids = _trusted_schema_ids(config, bundle)
    print({"status": "training_start", "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs, "stage3_epochs": stage3_epochs}, flush=True)

    for epoch in range(stage1_epochs):
        history = _train_stage1(train_loaders, encoder, optimizer, engine, grad_clip_norm)
        training_history["stage1"].append(engine.summarize_history(history))
        print({"mode": args.mode, "stage": "stage1", "epoch": epoch + 1, **training_history["stage1"][-1]})

    if not direct_only:
        for epoch in range(stage2_epochs):
            history = _train_stage2(train_loaders, bundle, encoder, path_scorer, pair_model, optimizer, engine, grad_clip_norm)
            training_history["stage2"].append(engine.summarize_history(history))
            print({"mode": args.mode, "stage": "stage2", "epoch": epoch + 1, **training_history["stage2"][-1]})

    stage3_optimizer_cfg = config["training"].get("stage3_optimizer", {})
    stage3_lr = float(stage3_optimizer_cfg.get("lr") or config["training"]["lr"])
    stage3_weight_decay = float(stage3_optimizer_cfg.get("weight_decay") or config["training"]["weight_decay"])
    optimizer = _build_optimizer(
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        lr=stage3_lr,
        weight_decay=stage3_weight_decay,
        freeze_encoder=freeze_encoder_stage3,
    )
    stage3_scheduler = _build_stage3_scheduler(optimizer, config=config, total_epochs=stage3_epochs)

    print({"status": "initial_valid_eval"}, flush=True)
    cached_stage34_embeddings = None
    if freeze_encoder_stage3:
        encoder.eval()
        with torch.no_grad():
            cached_stage34_embeddings = encoder(bundle.graph_data).detach()
    best_valid_metrics = _score_valid_epoch(
        loaders=valid_loaders,
        bundle=valid_loaders["bundle"],
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
    )
    best_snapshot = _snapshot_model_state(encoder, path_scorer, pair_model)
    best_stage3_epoch = 0
    no_improve_epochs = 0

    for epoch in range(stage3_epochs):
        history = _train_stage3(
            train_loaders,
            bundle,
            encoder,
            path_scorer,
            pair_model,
            optimizer,
            engine,
            grad_clip_norm,
            direct_only,
            use_neighbor_context=use_neighbor_context,
            freeze_encoder=freeze_encoder_stage3,
            cached_node_embeddings=cached_stage34_embeddings,
        )
        training_history["stage3"].append(engine.summarize_history(history))
        current_lr = float(optimizer.param_groups[0]["lr"])
        training_history["stage3"][-1]["lr"] = current_lr
        current_valid_metrics = _score_valid_epoch(
            loaders=valid_loaders,
            bundle=valid_loaders["bundle"],
            encoder=encoder,
            path_scorer=path_scorer,
            pair_model=pair_model,
            direct_only=direct_only,
            use_neighbor_context=use_neighbor_context,
            ks=config["evaluation"]["ks"],
            cached_node_embeddings=cached_stage34_embeddings,
        )
        if current_valid_metrics["auprc"] > best_valid_metrics["auprc"]:
            best_valid_metrics = current_valid_metrics
            best_snapshot = _snapshot_model_state(encoder, path_scorer, pair_model)
            best_stage3_epoch = epoch + 1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        print(
            {
                "mode": args.mode,
                "stage": "stage3",
                "epoch": epoch + 1,
                **training_history["stage3"][-1],
                "valid_auprc": current_valid_metrics["auprc"],
                "best_valid_auprc": best_valid_metrics["auprc"],
            }
        )
        if stage3_scheduler is not None:
            stage3_scheduler.step()
        if early_stop_patience is not None and no_improve_epochs >= early_stop_patience:
            break

    _restore_model_state(encoder, path_scorer, pair_model, best_snapshot)

    if not direct_only and stage4_epochs > 0:
        for epoch in range(stage4_epochs):
            history = _train_stage4(
                train_loaders,
                bundle,
                encoder,
                path_scorer,
                pair_model,
                optimizer,
                engine,
                grad_clip_norm,
                pseudo_config=config["pseudo"],
                trusted_schema_ids=trusted_schema_ids,
                anchor_weight=stage4_anchor_weight,
                freeze_encoder=freeze_encoder_stage4,
                cached_node_embeddings=cached_stage34_embeddings if freeze_encoder_stage4 else None,
            )
            training_history["stage4"].append(engine.summarize_history(history))
            print({"mode": args.mode, "stage": "stage4", "epoch": epoch + 1, **training_history["stage4"][-1]})

    valid_metrics, valid_pair_rows = _evaluate_pairs(
        valid_loaders,
        valid_loaders["bundle"],
        encoder,
        path_scorer,
        pair_model,
        direct_only,
        use_neighbor_context,
        output_dir,
        split_name="valid",
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
    )
    test_metrics, test_pair_rows = _evaluate_pairs(
        test_loaders,
        test_loaders["bundle"],
        encoder,
        path_scorer,
        pair_model,
        direct_only,
        use_neighbor_context,
        output_dir,
        split_name="test",
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
    )
    valid_pair_strata = _evaluate_pair_strata(valid_pair_rows, ks=config["evaluation"]["ks"])
    test_pair_strata = _evaluate_pair_strata(test_pair_rows, ks=config["evaluation"]["ks"])
    valid_path_metrics, valid_explanation_metrics, _, valid_path_diag = _evaluate_path_and_explanations(
        split_name="valid",
        bundle=valid_loaders["bundle"],
        config=config,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        output_dir=output_dir,
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
        benchmark_name="path",
        hard_cfg={"enabled": False},
    )
    test_path_metrics, test_explanation_metrics, _, test_path_diag = _evaluate_path_and_explanations(
        split_name="test",
        bundle=test_loaders["bundle"],
        config=config,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        output_dir=output_dir,
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
        benchmark_name="path",
        hard_cfg={"enabled": False},
    )
    path_hard_cfg = deepcopy(config.get("evaluation", {}).get("path_hard", {}))
    valid_path_hard_metrics, valid_explanation_hard_metrics, _, valid_path_hard_diag = _evaluate_path_and_explanations(
        split_name="valid",
        bundle=valid_loaders["bundle"],
        config=config,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        output_dir=output_dir,
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
        benchmark_name="path_hard",
        hard_cfg=path_hard_cfg,
    )
    test_path_hard_metrics, test_explanation_hard_metrics, _, test_path_hard_diag = _evaluate_path_and_explanations(
        split_name="test",
        bundle=test_loaders["bundle"],
        config=config,
        encoder=encoder,
        path_scorer=path_scorer,
        pair_model=pair_model,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        output_dir=output_dir,
        ks=config["evaluation"]["ks"],
        cached_node_embeddings=cached_stage34_embeddings,
        benchmark_name="path_hard",
        hard_cfg=path_hard_cfg,
    )
    stage4_summary = _summarize_stage4(
        train_loaders,
        bundle,
        encoder,
        path_scorer,
        pair_model,
        pseudo_config=config["pseudo"],
        trusted_schema_ids=trusted_schema_ids,
        direct_only=direct_only,
        use_neighbor_context=use_neighbor_context,
        cached_node_embeddings=cached_stage34_embeddings,
    )

    summary = {
        "mode": args.mode,
        "best_valid_stage3_epoch": best_stage3_epoch,
        "best_valid_metrics": best_valid_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "valid_pair_strata": valid_pair_strata,
        "test_pair_strata": test_pair_strata,
        "valid_path": valid_path_metrics,
        "test_path": test_path_metrics,
        "valid_explanation": valid_explanation_metrics,
        "test_explanation": test_explanation_metrics,
        "valid_path_diagnostics": valid_path_diag,
        "test_path_diagnostics": test_path_diag,
        "valid_path_hard": valid_path_hard_metrics,
        "test_path_hard": test_path_hard_metrics,
        "valid_explanation_hard": valid_explanation_hard_metrics,
        "test_explanation_hard": test_explanation_hard_metrics,
        "valid_path_hard_diagnostics": valid_path_hard_diag,
        "test_path_hard_diagnostics": test_path_hard_diag,
        "stage4": stage4_summary,
        "training_history": training_history,
    }
    save_json(summary, output_dir / "contrast_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
