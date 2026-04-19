# Minimal Runnable Experiment Plan

## Phase 1: Tiny Sanity Check

Goal:

- validate file parsing
- validate schema prior extraction
- validate split generation
- validate retriever on a small pair subset
- validate model forward pass and loss plumbing

Config:

- `configs/experiments/tiny_sanity.yaml`
- cap KG rows and rationale rows
- smaller hidden dimension
- one epoch per enabled stage

Checks:

- `graph_data.pt` saved
- split CSVs saved
- schema prior JSON saved
- candidate cache populated for a handful of rationale-labeled pairs
- training loop produces finite losses

## Phase 2: Random Split Baseline

Goal:

- obtain first end-to-end pair/path metrics
- establish the main baseline for paper tables

Config:

- `configs/experiments/random_baseline.yaml`

Run order:

1. Stage 0
2. Stage 1
3. Stage 2
4. Stage 3
5. evaluation

Outputs:

- metrics JSON
- per-pair predictions CSV
- per-path ranking CSV
- ablation-ready config snapshot

## Phase 3: Cold-Drug And Cold-Disease

Goal:

- measure generalization to unseen endpoints

Configs:

- `configs/experiments/cold_drug.yaml`
- `configs/experiments/cold_disease.yaml`

Notes:

- preserve schema prior build inside training folds only
- never leak test-pair gold rationales into train cache

## Phase 4: Pseudo-Rationale Completion

Goal:

- improve positive-no-path coverage
- test the main semi-supervised contribution

Procedure:

1. start from the best random-split checkpoint after Stage 3
2. score candidate bags for positive-no-path pairs
3. filter pseudo paths with confidence and stability gates
4. continue training with `L_pseudo + L_consistency`

Core ablations:

- no pseudo-rationale
- relaxed vs strict pseudo gates
- `max` vs `topk_logsumexp` vs `attention`
- HGT vs RGCN
