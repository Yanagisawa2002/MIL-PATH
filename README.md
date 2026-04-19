# Semi-Mechanistic Drug Repurposing

This project implements a semi-mechanistic drug repurposing framework for partially rationale-labeled prediction on mapped PrimeKG + DrugMechDB data.

The core pipeline is:

`retrieve -> rank -> aggregate -> pseudo-rationale self-training`

The implementation is modular so each block can be swapped for ablations:

- heterogenous graph encoder: `HGT` by default, `RGCN` interface retained
- candidate path retriever: schema-prior constrained BFS / beam search
- pair-conditioned path scorer: edge-aware GRU path encoder
- path-to-pair aggregation: `max`, `topk_logsumexp`, `attention`, `noisy_or`
- pseudo-rationale completion: confidence-gated teacher/student style selection

## Current Mainline

The current strongest model is:

- strong direct branch: endpoint embeddings + explicit pairwise graph features
- mechanistic branch: schema-constrained candidate retrieval + pair-conditioned path scoring
- path bag aggregation: `topk_logsumexp(top-4)`
- cross-branch interaction: direct and mech branches refine each other before final fusion
- weakly supervised path coverage: `cached_positive_only` for positive pairs without gold rationales
- tuned stage-2 hard negatives: `3 corrupt_internal + 1 cross_pair_same_schema + 0 cross_pair_same_hop`

Current best test results:

- `cold-drug`: AUROC `0.7756`, AUPRC `0.8178`
- `cold-disease`: AUROC `0.7690`, AUPRC `0.8133`

## Raw Data Assumption

The current workspace already contains:

- `all_data(indication_only)/forSemi_KG.csv`
- `all_data(indication_only)/forSemi_Mech.csv`
- `all_data(indication_only)/forSemi_Mech.json`

The `indication_only` graph still contains direct indication edges. By default the data builder can remove direct clinical `drug-disease` edges from the graph artifact used by the encoder to reduce shortcut leakage.

## Quick Start

Build processed artifacts on a tiny subset:

```bash
python scripts/build_data.py --config configs/experiments/tiny_sanity.yaml
```

Audit the current split:

```bash
python scripts/audit_splits.py --processed-dir data/processed
```

Run the experiment skeleton:

```bash
python scripts/run_experiment.py --config configs/experiments/tiny_sanity.yaml
```

Compile-check the project:

```bash
python -m compileall src scripts
```

## Directory Layout

```text
configs/
  base.yaml
  experiments/
data/
  processed/
  splits/
  cache/
docs/
  technical_design.md
  minimal_experiment_plan.md
outputs/
scripts/
src/
  data/
  models/
  training/
  evaluation/
  utils/
```

## Stage Mapping

- Stage 0: artifact build, split generation, schema prior estimation
- Stage 1: graph encoder pretraining
- Stage 2: supervised path ranking on rationale-labeled positive pairs
- Stage 3: joint pair training with direct branch + path bag branch
- Stage 4: pseudo-rationale completion for positive-no-path pairs
- Stage 5: reserved interface for later PU discovery

## Current Scope

This repository currently includes:

- stage-0 artifact building and audited split generation
- split-after-positive `1:1` negative sampling with strong clinical negatives and shared-endpoint hard negatives
- stage-1 graph encoder pretraining
- stage-2 supervised path ranking with configurable hard negatives
- stage-3 joint pair prediction with direct branch + mechanistic path bag branch
- stage-4 pseudo-rationale interfaces for positive-no-path pairs
- experiment configs and runners for `random`, `cold-drug`, and `cold-disease`

The current best gains come from supervised + weakly supervised path learning in stage 2/3; stage-4 pseudo-rationale self-training is implemented, but is not the main source of the strongest reported results.
