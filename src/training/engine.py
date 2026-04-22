"""Stage-wise training skeleton for the hierarchical framework."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.training.losses import (
    consistency_loss,
    head_alignment_loss,
    high_confidence_distillation_loss,
    pair_classification_loss,
    path_binary_aux_loss,
    path_ranking_loss,
    pseudo_pair_loss,
    pseudo_path_loss,
    teacher_guided_reranker_loss,
)


class TrainingEngine:
    """Run stage-specific training steps on pre-batched tensors."""

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.loss_weights = config["training"]["loss_weights"]
        self.margin = float(config["training"]["ranking_margin"])
        self.path_rank_reduction = str(config["training"].get("stage2_ranking_reduction", "mean"))
        self.path_rank_top_k = int(config["training"].get("stage2_ranking_top_k", 2))
        self.path_binary_aux_cfg = dict(config["training"].get("path_binary_aux", {}))
        self.path_head_alignment_weight = float(config["training"].get("path_head_alignment_weight", 0.0))
        self.stage3_explanation_aux_cfg = dict(config["training"].get("stage3_explanation_aux", {}))
        self.stage3_binary_calibration_aux_cfg = dict(
            config["training"].get("stage3_binary_calibration_aux", {})
        )
        self.stage3_binary_calibration_teacher_cfg = dict(
            config["training"].get("stage3_binary_calibration_teacher", {})
        )
        self.stage3_explanation_distill_cfg = dict(
            config["training"].get("stage3_explanation_distillation", {})
        )
        self.retrieval_explanation_alignment_cfg = dict(
            config["training"].get("retrieval_explanation_alignment", {})
        )
        self.retrieval_teacher_guided_cfg = dict(
            config["training"].get("retrieval_teacher_guided", {})
        )
        self.stage4_pair_cls_weight = float(
            config["training"].get("stage4_pair_cls_weight", self.loss_weights["pair_cls"])
        )
        self.stage5_pair_pu_weight = float(config["training"].get("stage5_pair_pu_weight", 0.2))

    @staticmethod
    def _topk_mask(scores: torch.Tensor, bag_mask: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            return bag_mask.bool()
        k = min(max(1, int(top_k)), int(scores.size(1)))
        masked_scores = scores.masked_fill(~bag_mask.bool(), float("-inf"))
        top_idx = masked_scores.topk(k=k, dim=1).indices
        top_mask = torch.zeros_like(bag_mask, dtype=torch.bool)
        top_mask.scatter_(1, top_idx, True)
        return top_mask & bag_mask.bool()

    @staticmethod
    def _rowwise_max(values: torch.Tensor, bag_mask: torch.Tensor) -> torch.Tensor:
        masked_values = values.masked_fill(~bag_mask.bool(), float("-inf"))
        row_max = masked_values.max(dim=1).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        return row_max

    @staticmethod
    def _apply_top1_fallback(
        *,
        distill_mask: torch.Tensor,
        bag_mask: torch.Tensor,
        ranking_scores: torch.Tensor,
        teacher_probs: torch.Tensor,
        bag_reliability: torch.Tensor | None,
        min_teacher_prob: float,
        min_bag_reliability: float,
    ) -> torch.Tensor:
        fallback_mask = distill_mask.clone()
        empty_rows = bag_mask.bool().any(dim=1) & (~distill_mask.any(dim=1))
        if not empty_rows.any():
            return fallback_mask

        row_max_teacher = TrainingEngine._rowwise_max(teacher_probs, bag_mask)
        empty_rows = empty_rows & (row_max_teacher >= float(min_teacher_prob))
        if bag_reliability is not None:
            empty_rows = empty_rows & (bag_reliability >= float(min_bag_reliability))
        if not empty_rows.any():
            return fallback_mask

        masked_scores = ranking_scores.masked_fill(~bag_mask.bool(), float("-inf"))
        top_idx = masked_scores.argmax(dim=1)
        row_idx = torch.nonzero(empty_rows, as_tuple=False).reshape(-1)
        fallback_mask[row_idx, top_idx[row_idx]] = True
        return fallback_mask

    def _bag_gold_ranking_loss(
        self,
        scores: torch.Tensor,
        bag_mask: torch.Tensor,
        is_gold: torch.Tensor,
        *,
        margin: float,
        reduction: str,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_losses: list[torch.Tensor] = []
        for row_idx in range(scores.size(0)):
            row_bag_mask = bag_mask[row_idx].bool()
            if not row_bag_mask.any():
                continue
            row_gold_mask = is_gold[row_idx].bool() & row_bag_mask
            row_neg_mask = (~is_gold[row_idx].bool()) & row_bag_mask
            if not row_gold_mask.any() or not row_neg_mask.any():
                continue
            positive_score = scores[row_idx][row_gold_mask].max().reshape(1)
            negative_scores = scores[row_idx][row_neg_mask].reshape(1, -1)
            valid_losses.append(
                path_ranking_loss(
                    positive_score,
                    negative_scores,
                    margin=margin,
                    reduction=reduction,
                    top_k=top_k,
                )
            )
        if not valid_losses:
            zero = scores.new_zeros(())
            return zero, scores.new_tensor(0.0)
        return torch.stack(valid_losses).mean(), scores.new_tensor(float(len(valid_losses)))

    def stage1_loss(self, encoder, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_embeddings = encoder(batch["graph_artifact"])
        relation_loss = encoder.relation_reconstruction_loss(
            node_embeddings=node_embeddings,
            head_idx=batch["head_idx"],
            relation_idx=batch["relation_idx"],
            tail_idx=batch["tail_idx"],
        )
        type_loss = encoder.masked_type_prediction_loss(
            node_embeddings=node_embeddings,
            node_type_ids=batch["node_type_ids"],
        )
        total = relation_loss + type_loss
        return {
            "total": total,
            "relation_reconstruction": relation_loss,
            "masked_type": type_loss,
        }

    def stage2_loss(
        self,
        path_scorer,
        pair_embedding: torch.Tensor,
        positive_batch: dict[str, torch.Tensor],
        negative_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        aux_enabled = bool(self.path_binary_aux_cfg.get("enabled", False))
        decoupled_enabled = bool(getattr(path_scorer, "decoupled_heads_enabled", False))
        use_separate_head = bool(self.path_binary_aux_cfg.get("use_separate_head", False))
        request_aux = aux_enabled or decoupled_enabled

        pos_outputs = path_scorer(pair_embedding=pair_embedding, return_aux=request_aux, **positive_batch)
        if request_aux:
            pos_scores, _, pos_aux = pos_outputs
        else:
            pos_scores, _ = pos_outputs
            pos_aux = {}
        bsz, num_neg, seq_len, hidden_dim = negative_batch["node_states"].shape
        flat_neg = {
            "node_states": negative_batch["node_states"].reshape(bsz * num_neg, seq_len, hidden_dim),
            "relation_ids": negative_batch["relation_ids"].reshape(bsz * num_neg, seq_len - 1),
            "node_type_ids": negative_batch["node_type_ids"].reshape(bsz * num_neg, seq_len),
            "mask": negative_batch["mask"].reshape(bsz * num_neg, seq_len),
            "schema_bucket_ids": negative_batch["schema_bucket_ids"].reshape(bsz * num_neg),
            "hop_counts": negative_batch["hop_counts"].reshape(bsz * num_neg),
            "path_source_ids": negative_batch["path_source_ids"].reshape(bsz * num_neg),
        }
        repeated_pair = pair_embedding.unsqueeze(1).repeat(1, num_neg, 1).reshape(bsz * num_neg, -1)
        neg_outputs = path_scorer(pair_embedding=repeated_pair, return_aux=request_aux, **flat_neg)
        if request_aux:
            neg_scores, _, neg_aux = neg_outputs
        else:
            neg_scores, _ = neg_outputs
            neg_aux = {}
        neg_scores = neg_scores.reshape(bsz, num_neg)
        positive_rank_scores = pos_aux.get("explanation_score", pos_scores)
        negative_rank_scores = neg_aux.get("explanation_score", neg_scores.reshape(-1)).reshape(bsz, num_neg)
        rank_loss = path_ranking_loss(
            positive_rank_scores,
            negative_rank_scores,
            margin=self.margin,
            reduction=self.path_rank_reduction,
            top_k=self.path_rank_top_k,
        )
        total = self.loss_weights["path_rank"] * rank_loss
        outputs: dict[str, torch.Tensor] = {
            "total": total,
            "path_rank": rank_loss,
        }

        if aux_enabled:
            allowed_sources = set(self.path_binary_aux_cfg.get("negative_sources", []))
            selected_negative_logits: list[torch.Tensor] = []
            neg_binary_logits = neg_aux.get("binary_logit")
            pos_binary_logits = pos_aux.get("binary_logit")
            if use_separate_head and neg_binary_logits is not None and pos_binary_logits is not None:
                neg_binary_logits = neg_binary_logits.reshape(bsz, num_neg)
                positive_binary_logits = pos_binary_logits.reshape(bsz)
            else:
                neg_binary_logits = negative_rank_scores
                positive_binary_logits = positive_rank_scores

            negative_sources = negative_batch.get("path_sources", [])
            for batch_idx, row_sources in enumerate(negative_sources):
                for neg_idx, source in enumerate(row_sources):
                    if source in allowed_sources:
                        selected_negative_logits.append(neg_binary_logits[batch_idx, neg_idx])

            if selected_negative_logits:
                binary_loss = path_binary_aux_loss(
                    positive_logits=positive_binary_logits,
                    negative_logits=torch.stack(selected_negative_logits),
                )
                total = total + float(self.path_binary_aux_cfg.get("weight", 0.0)) * binary_loss
                outputs["total"] = total
                outputs["path_binary_aux"] = binary_loss
                outputs["path_binary_aux_negatives"] = torch.tensor(
                    float(len(selected_negative_logits)),
                    device=positive_binary_logits.device,
                )
            else:
                outputs["path_binary_aux"] = torch.zeros_like(rank_loss)
                outputs["path_binary_aux_negatives"] = torch.tensor(0.0, device=rank_loss.device)
        if decoupled_enabled and self.path_head_alignment_weight > 0.0:
            alignment_loss = head_alignment_loss(
                evidence_logits=torch.cat([pos_scores.reshape(-1), neg_scores.reshape(-1)], dim=0),
                explanation_logits=torch.cat(
                    [positive_rank_scores.reshape(-1), negative_rank_scores.reshape(-1)],
                    dim=0,
                ),
            )
            total = total + self.path_head_alignment_weight * alignment_loss
            outputs["total"] = total
            outputs["path_head_align"] = alignment_loss
        return outputs

    def stage3_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        path_bag: dict[str, torch.Tensor] | None = None,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, torch.Tensor]:
        pair_loss = pair_classification_loss(model_outputs["pair_score"], labels)
        total = self.loss_weights["pair_cls"] * pair_loss
        outputs = {
            "total": total,
            "pair_cls": pair_loss,
        }

        if path_bag is None:
            return outputs

        bag_mask = path_bag.get("path_mask")
        is_gold = path_bag.get("is_gold")
        if bag_mask is None or is_gold is None:
            return outputs

        if bool(self.stage3_explanation_aux_cfg.get("enabled", False)) and "explanation_path_scores" in model_outputs:
            explanation_margin = self.stage3_explanation_aux_cfg.get("margin")
            explanation_loss, explanation_count = self._bag_gold_ranking_loss(
                model_outputs["explanation_path_scores"],
                bag_mask,
                is_gold,
                margin=self.margin if explanation_margin is None else float(explanation_margin),
                reduction=str(self.stage3_explanation_aux_cfg.get("reduction", self.path_rank_reduction)),
                top_k=int(self.stage3_explanation_aux_cfg.get("top_k", self.path_rank_top_k)),
            )
            total = total + float(self.stage3_explanation_aux_cfg.get("weight", 0.0)) * explanation_loss
            outputs["total"] = total
            outputs["stage3_explanation_aux"] = explanation_loss
            outputs["stage3_explanation_aux_rows"] = explanation_count

        if bool(self.stage3_binary_calibration_aux_cfg.get("enabled", False)) and "binary_path_scores" in model_outputs:
            binary_margin = self.stage3_binary_calibration_aux_cfg.get("margin")
            binary_loss, binary_count = self._bag_gold_ranking_loss(
                model_outputs["binary_path_scores"],
                bag_mask,
                is_gold,
                margin=self.margin if binary_margin is None else float(binary_margin),
                reduction=str(
                    self.stage3_binary_calibration_aux_cfg.get("reduction", self.path_rank_reduction)
                ),
                top_k=int(self.stage3_binary_calibration_aux_cfg.get("top_k", self.path_rank_top_k)),
            )
            total = total + float(self.stage3_binary_calibration_aux_cfg.get("weight", 0.0)) * binary_loss
            outputs["total"] = total
            outputs["stage3_binary_calibration_aux"] = binary_loss
            outputs["stage3_binary_calibration_aux_rows"] = binary_count

        if bool(self.stage3_binary_calibration_teacher_cfg.get("enabled", False)) and "binary_path_scores" in model_outputs:
            teacher_scores = model_outputs.get("explanation_path_scores")
            student_scores = model_outputs.get("binary_path_scores")
            if teacher_scores is not None and student_scores is not None:
                binary_teacher_loss, binary_teacher_rows = teacher_guided_reranker_loss(
                    student_logits=student_scores,
                    teacher_logits=teacher_scores,
                    bag_mask=bag_mask,
                    is_gold=is_gold,
                    is_retrieved=path_bag.get("is_retrieved"),
                    use_retrieved_only=bool(self.stage3_binary_calibration_teacher_cfg.get("use_retrieved_only", False)),
                    top_k=int(self.stage3_binary_calibration_teacher_cfg.get("top_k", 0)),
                    teacher_temperature=float(
                        self.stage3_binary_calibration_teacher_cfg.get("teacher_temperature", 1.0)
                    ),
                    student_temperature=float(
                        self.stage3_binary_calibration_teacher_cfg.get("student_temperature", 1.0)
                    ),
                    gold_boost=float(self.stage3_binary_calibration_teacher_cfg.get("gold_boost", 0.0)),
                )
                total = total + float(self.stage3_binary_calibration_teacher_cfg.get("weight", 0.0)) * binary_teacher_loss
                outputs["total"] = total
                outputs["stage3_binary_calibration_teacher"] = binary_teacher_loss
                outputs["stage3_binary_calibration_teacher_rows"] = binary_teacher_rows

        if bool(self.retrieval_explanation_alignment_cfg.get("enabled", False)) and "reranked_path_scores" in model_outputs:
            retrieval_margin = self.retrieval_explanation_alignment_cfg.get("margin")
            retrieval_loss, retrieval_count = self._bag_gold_ranking_loss(
                model_outputs["reranked_path_scores"],
                bag_mask,
                is_gold,
                margin=self.margin if retrieval_margin is None else float(retrieval_margin),
                reduction=str(self.retrieval_explanation_alignment_cfg.get("reduction", self.path_rank_reduction)),
                top_k=int(self.retrieval_explanation_alignment_cfg.get("top_k", self.path_rank_top_k)),
            )
            total = total + float(self.retrieval_explanation_alignment_cfg.get("weight", 0.0)) * retrieval_loss
            outputs["total"] = total
            outputs["retrieval_explanation_alignment"] = retrieval_loss
            outputs["retrieval_explanation_alignment_rows"] = retrieval_count

        retrieval_student_key = str(self.retrieval_teacher_guided_cfg.get("student_key", "reranked_path_scores"))
        if bool(self.retrieval_teacher_guided_cfg.get("enabled", False)) and retrieval_student_key in model_outputs:
            teacher_scores = model_outputs.get("explanation_path_scores")
            if teacher_scores is not None:
                reranker_loss, reranker_rows = teacher_guided_reranker_loss(
                    student_logits=model_outputs[retrieval_student_key],
                    teacher_logits=teacher_scores,
                    bag_mask=bag_mask,
                    is_gold=is_gold,
                    is_retrieved=path_bag.get("is_retrieved"),
                    use_retrieved_only=bool(self.retrieval_teacher_guided_cfg.get("use_retrieved_only", False)),
                    top_k=int(self.retrieval_teacher_guided_cfg.get("top_k", 0)),
                    teacher_temperature=float(self.retrieval_teacher_guided_cfg.get("teacher_temperature", 1.0)),
                    student_temperature=float(self.retrieval_teacher_guided_cfg.get("student_temperature", 1.0)),
                    gold_boost=float(self.retrieval_teacher_guided_cfg.get("gold_boost", 0.0)),
                )
                total = total + float(self.retrieval_teacher_guided_cfg.get("weight", 0.0)) * reranker_loss
                outputs["total"] = total
                outputs["retrieval_teacher_guided"] = reranker_loss
                outputs["retrieval_teacher_guided_rows"] = reranker_rows
                outputs["retrieval_teacher_guided_student"] = teacher_scores.new_tensor(
                    1.0 if retrieval_student_key == "retrieval_logits" else 0.0
                )

        if bool(self.stage3_explanation_distill_cfg.get("enabled", False)) and "explanation_path_scores" in model_outputs:
            student_key = str(self.stage3_explanation_distill_cfg.get("student_key", "raw_path_scores"))
            evidence_scores = model_outputs.get(student_key, model_outputs.get("raw_path_scores"))
            explanation_scores = model_outputs.get("explanation_path_scores")
            if evidence_scores is not None and explanation_scores is not None:
                start_epoch = int(self.stage3_explanation_distill_cfg.get("start_epoch", 1))
                if current_epoch is not None and int(current_epoch) < start_epoch:
                    outputs["stage3_explanation_distill"] = evidence_scores.new_zeros(())
                    outputs["stage3_explanation_distill_paths"] = evidence_scores.new_tensor(0.0)
                    outputs["stage3_explanation_distill_weight"] = evidence_scores.new_tensor(0.0)
                    return outputs

                base_weight = float(self.stage3_explanation_distill_cfg.get("weight", 0.0))
                ramp_epochs = int(self.stage3_explanation_distill_cfg.get("ramp_epochs", 0))
                effective_weight = base_weight
                if current_epoch is not None and ramp_epochs > 0:
                    ramp_progress = max(0, int(current_epoch) - start_epoch + 1)
                    effective_weight = base_weight * min(1.0, ramp_progress / max(1, ramp_epochs))
                if effective_weight <= 0.0:
                    outputs["stage3_explanation_distill"] = evidence_scores.new_zeros(())
                    outputs["stage3_explanation_distill_paths"] = evidence_scores.new_tensor(0.0)
                    outputs["stage3_explanation_distill_weight"] = evidence_scores.new_tensor(0.0)
                    return outputs

                teacher_probs = torch.sigmoid(explanation_scores)
                binary_scores = model_outputs.get("binary_path_scores")
                if binary_scores is not None:
                    binary_probs = torch.sigmoid(binary_scores)
                else:
                    binary_probs = teacher_probs
                agreement = 1.0 - (teacher_probs - binary_probs).abs().clamp(min=0.0, max=1.0)
                bag_reliability = model_outputs.get("mechanistic_reliability")

                distill_mask = bag_mask.bool()
                if bool(self.stage3_explanation_distill_cfg.get("use_topk_only", True)):
                    distill_mask = distill_mask & self._topk_mask(
                        explanation_scores,
                        bag_mask,
                        int(self.stage3_explanation_distill_cfg.get("top_k", 4)),
                    )
                distill_mask = distill_mask & (
                    teacher_probs >= float(self.stage3_explanation_distill_cfg.get("min_explanation_prob", 0.6))
                )
                distill_mask = distill_mask & (
                    agreement >= float(self.stage3_explanation_distill_cfg.get("min_agreement", 0.6))
                )
                if binary_scores is not None:
                    distill_mask = distill_mask & (
                        binary_probs >= float(self.stage3_explanation_distill_cfg.get("min_binary_prob", 0.55))
                    )
                if bag_reliability is not None:
                    distill_mask = distill_mask & (
                        bag_reliability.unsqueeze(1)
                        >= float(self.stage3_explanation_distill_cfg.get("min_bag_reliability", 0.0))
                    )
                if bool(self.stage3_explanation_distill_cfg.get("use_retrieved_only", False)):
                    is_retrieved = path_bag.get("is_retrieved")
                    if is_retrieved is not None:
                        distill_mask = distill_mask & (is_retrieved.bool() | is_gold.bool())
                if bool(self.stage3_explanation_distill_cfg.get("require_gold_paths", False)):
                    distill_mask = distill_mask & is_gold.bool()

                if bool(self.stage3_explanation_distill_cfg.get("use_relative_to_row_max", False)):
                    explanation_ratio = float(self.stage3_explanation_distill_cfg.get("explanation_row_ratio", 0.9))
                    binary_ratio = float(self.stage3_explanation_distill_cfg.get("binary_row_ratio", 0.9))
                    agreement_ratio = float(self.stage3_explanation_distill_cfg.get("agreement_row_ratio", 0.95))
                    row_max_teacher = self._rowwise_max(teacher_probs, bag_mask)
                    row_max_binary = self._rowwise_max(binary_probs, bag_mask)
                    row_max_agreement = self._rowwise_max(agreement, bag_mask)
                    distill_mask = distill_mask & (
                        teacher_probs >= row_max_teacher.unsqueeze(1) * explanation_ratio
                    )
                    distill_mask = distill_mask & (
                        binary_probs >= row_max_binary.unsqueeze(1) * binary_ratio
                    )
                    distill_mask = distill_mask & (
                        agreement >= row_max_agreement.unsqueeze(1) * agreement_ratio
                    )

                if bool(self.stage3_explanation_distill_cfg.get("fallback_top1_when_empty", False)):
                    distill_mask = self._apply_top1_fallback(
                        distill_mask=distill_mask,
                        bag_mask=bag_mask,
                        ranking_scores=explanation_scores,
                        teacher_probs=teacher_probs,
                        bag_reliability=bag_reliability,
                        min_teacher_prob=float(
                            self.stage3_explanation_distill_cfg.get("fallback_min_teacher_prob", 0.55)
                        ),
                        min_bag_reliability=float(
                            self.stage3_explanation_distill_cfg.get("fallback_min_bag_reliability", 0.0)
                        ),
                    )

                distill_weights = teacher_probs * agreement
                if binary_scores is not None:
                    distill_weights = distill_weights * binary_probs
                if bag_reliability is not None:
                    distill_weights = distill_weights * bag_reliability.unsqueeze(1)

                distill_loss, distill_count = high_confidence_distillation_loss(
                    evidence_logits=evidence_scores,
                    explanation_logits=explanation_scores,
                    mask=distill_mask,
                    weights=distill_weights,
                )
                total = total + effective_weight * distill_loss
                outputs["total"] = total
                outputs["stage3_explanation_distill"] = distill_loss
                outputs["stage3_explanation_distill_paths"] = distill_count
                outputs["stage3_explanation_distill_weight"] = evidence_scores.new_tensor(effective_weight)

        return outputs

    def stage4_loss(
        self,
        model_outputs_a: dict[str, torch.Tensor],
        model_outputs_b: dict[str, torch.Tensor],
        pseudo_scores: torch.Tensor,
        pseudo_weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_loss = pair_classification_loss(model_outputs_a["pair_score"], labels)
        cons_loss = consistency_loss(
            model_outputs_a["path_attention"],
            model_outputs_b["path_attention"],
        )
        pseudo_loss = pseudo_path_loss(pseudo_scores, pseudo_weights)
        total = (
            self.stage4_pair_cls_weight * pair_loss
            + self.loss_weights["consistency"] * cons_loss
            + self.loss_weights["pseudo"] * pseudo_loss
        )
        return {
            "total": total,
            "pair_cls": pair_loss,
            "consistency": cons_loss,
            "pseudo": pseudo_loss,
        }

    def stage5_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        pseudo_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pseudo_loss = pseudo_pair_loss(model_outputs["pair_score"], pseudo_weights)
        total = self.stage5_pair_pu_weight * pseudo_loss
        return {
            "total": total,
            "pseudo_pair": pseudo_loss,
        }

    def summarize_history(self, history: list[dict[str, float]]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for step in history:
            for name, value in step.items():
                buckets[name].append(float(value))
        return {name: sum(values) / len(values) for name, values in buckets.items() if values}
