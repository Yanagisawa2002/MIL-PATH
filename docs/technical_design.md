# Technical Design

## 1. Task Restatement

We solve semi-supervised drug repurposing on PrimeKG with partially observed mechanism rationales mapped from DrugMechDB.

The supervision is hierarchical:

1. rationale-labeled positive pairs: positive `drug-disease` pairs with one or more mapped mechanism paths
2. rationale-unlabeled positive pairs: positive pairs without mapped mechanism paths
3. negative pairs: strong negatives when available, otherwise structure-controlled sampled negatives

The modeling chain is fixed as:

`pair-conditioned path retrieval -> path reranking -> path-to-pair aggregation -> pseudo-rationale completion`

This design preserves interpretability and keeps the first version stable enough for ablations.

## 2. Data Protocol

### 2.1 Raw Inputs

- `forSemi_KG.csv`: PrimeKG plus PrimeKG-Ext heterogeneous graph backbone
- `forSemi_Mech.csv`: path-level semi-supervised table
- `forSemi_Mech.json`: graph-structured mechanism path objects

### 2.2 Canonical Entities

The build step canonicalizes nodes with:

- `node_key = typed source-native key when available`
- fallback: `type:id`

Pairs are canonicalized as:

- `pair_id = drug_id::disease_id`

Weak negatives are sampled after positive splitting at a configurable ratio. The current default is `1:1` positive-to-negative to keep iterations fast and class balance controlled.

### 2.3 Stage-0 Outputs

The builder produces:

- `data/processed/graph_data.pt`
- `data/processed/pair_tables.pt`
- `data/processed/path_annotations.csv`
- `data/processed/schema_prior.json`
- `data/processed/split_audit.json`
- `data/splits/pairs_train.csv`
- `data/splits/pairs_valid.csv`
- `data/splits/pairs_test.csv`
- `data/cache/candidate_paths_cache/`

### 2.4 Pair Table Semantics

Each pair row carries:

- `pair_id`
- `drug_id`, `disease_id`
- `label` in `{0, 1}`
- `pair_source` such as `indication`, `contraindication`, `degree_controlled_negative`
- `has_gold_rationale`
- `num_gold_paths`
- `split`

### 2.5 Rationale Table Semantics

Each rationale row carries:

- `path_id`
- `pair_id`
- endpoint identifiers
- typed node sequence
- typed edge-hint sequence
- hop count
- schema identifier
- coarse family summary
- conservative subset flag

### 2.6 Real-Data Adjustment

The current export is close to one-path-per-pair, but not exact. In the inspected data, most pairs have one path, while a small number have multiple paths. The protocol therefore treats rationale supervision as a bag of positive rationales per pair instead of enforcing one-path-only.

## 3. Model Design

## 3.1 Module A: Heterogeneous Graph Encoder

Default: `HGT`

Fallback interface: `RGCN`

Responsibilities:

- encode typed nodes and edges into node embeddings
- expose drug/disease endpoint embeddings
- remain independent of path supervision

Stage-1 pretraining hooks:

- relation reconstruction
- masked node/edge type prediction
- optional local subgraph contrastive loss

## 3.2 Module B: Candidate Path Retriever

The retriever is pair-conditioned and schema-constrained.

Inputs:

- `(drug, disease)`
- graph artifact adjacency
- top schema prior families

Behavior:

- search only from the given drug toward the given disease
- restrict length to `2-4` hops by default
- prioritize high-support schemas mined from mapped rationales
- deduplicate by typed node sequence + relation sequence
- cap candidates per pair

The default implementation uses constrained BFS with beam pruning.

## 3.3 Module C: Pair-Conditioned Path Scorer

Inputs:

- pair embedding from the endpoint encoder
- path node states
- edge/relation ids
- node-type ids

Path encoder:

- edge-aware GRU with relation/type embeddings

Output:

- scalar `s(path | drug, disease)`
- path representation for downstream aggregation

Path supervision:

- gold path vs corruption path ranking loss
- same-pair unlabeled paths are not treated as negatives

## 3.4 Module D: Path-to-Pair Aggregator

The pair predictor has two branches:

- direct pair branch
- path bag branch

Final score:

`pair_score = alpha * direct_pair_score + (1 - alpha) * path_agg_score`

Supported aggregators:

- `max`
- `topk_logsumexp`
- `attention`
- `noisy_or`

This keeps pair prediction robust when rationale coverage is incomplete.

## 3.5 Module E: Pseudo-Rationale Self-Training

Only applied to positive pairs with no gold path.

Pseudo path selection is gated by configurable checks:

- pair score threshold
- top-1 path score threshold
- top1-top2 margin threshold
- allowed high-trust schema
- stability under dropout / edge masking
- optional multi-pass agreement

Selected pseudo rationales are added with confidence-weighted weak supervision.

## 4. Training Flow

### Stage 0

- build graph artifact
- build pair table
- build rationale table
- compute schema prior
- create split files

### Stage 1

- pretrain graph encoder on graph-only objectives

### Stage 2

- train path scorer on rationale-labeled positive pairs
- use corruption and cross-pair hard negatives

### Stage 3

- train direct pair branch + path bag branch jointly
- use all positive and negative pairs
- use MIL behavior for positive-no-path pairs

### Stage 4

- generate pseudo rationales for positive-no-path pairs
- continue training with confidence-weighted pseudo ranking loss
- add consistency loss across perturbations

### Stage 5

- reserved interface for later PU pair discovery

## 5. Negative Design

### Path-Level Negatives

- same pair, replaced intermediate node
- relation-compatible typed replacement
- cross-pair borrowed path with similar schema
- same-length type-matched but semantically mismatched path

### Pair-Level Negatives

- strong negatives if the graph variant contains them
- degree-controlled sampled non-edges
- shared-drug or shared-disease hard negatives

The current `indication_only` export removes direct contraindication edges, so the default config falls back to sampled negatives while keeping the strong-negative interface available for other graph variants.

## 6. Risks And Mitigations

### Shortcut leakage from direct indication edges

Mitigation:

- allow removing direct clinical `drug-disease` edges from the encoder graph
- keep pair labels in supervision tables only

### Gold rationale incompleteness

Mitigation:

- never treat same-pair unlabeled paths as negatives
- isolate path loss from pair classification loss

### Candidate explosion

Mitigation:

- schema prior restriction
- beam pruning
- hop limits
- candidate cap

### Sparse path supervision

Mitigation:

- retain direct pair branch
- use MIL for positive-no-path pairs
- add pseudo-rationale completion before any PU extension

### Data mismatch between assumption and export reality

Mitigation:

- protocol stores `num_gold_paths` rather than assuming exactly one
- evaluator supports multiple gold paths per pair

## 7. Final Directory Tree

```text
configs/
  base.yaml
  experiments/
    tiny_sanity.yaml
    random_baseline.yaml
    cold_drug.yaml
    cold_disease.yaml
data/
  processed/
  splits/
  cache/
docs/
  technical_design.md
  minimal_experiment_plan.md
outputs/
scripts/
  build_data.py
  run_experiment.py
src/
  data/
    protocol.py
    build_dataset.py
    path_retriever.py
  models/
    graph_encoder.py
    path_scorer.py
    pair_model.py
  training/
    losses.py
    pseudo_label.py
    engine.py
  evaluation/
    metrics.py
    evaluator.py
  utils/
    config.py
    io.py
```
