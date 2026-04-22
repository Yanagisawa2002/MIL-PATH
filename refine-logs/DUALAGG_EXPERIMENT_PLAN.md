# DualAgg Experiment Plan

**Problem**: Under partial path supervision, a single mechanistic bag aggregator cannot simultaneously optimize pair prediction and high-fidelity path explanation.
**Method Thesis**: `dualagg` should separate mechanistic evidence aggregation for pair prediction from explanation aggregation for path ranking, then use staged optimization and calibration to recover path quality without materially hurting pair performance.
**Date**: 2026-04-20

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: Dual aggregation is the right architectural abstraction for partial-path supervision. | This is the main method claim and higher-innovation alternative to loss-only fixes. | On both `cold-drug` and `cold-disease`, a minimal `dualagg` variant should improve hard path ranking over single-aggregator mainline while keeping pair AUROC/AUPRC within a small tolerance. | B1, B2 |
| C2: Stable `dualagg` requires teacher initialization and stagewise decoupling. | This rules out the anti-claim that current failures mean the architecture itself is wrong. | Teacher-initialized, explanation-protected `dualagg` should outperform scratch or fully-joint `dualagg` on hard MRR and top-1/top-5. | B2 |
| Anti-claim to rule out: gains come only from stronger path supervision, extra parameters, or retrieval perturbations. | Reviewers will otherwise say `scheme2` already explains the gain. | Hold retrieval fixed, compare against matched-capacity controls, and show that post-hoc calibration can raise path-binary without changing pair trunk. | B2, B3 |

## Paper Storyline
- Main paper must prove:
  - `dualagg` is a principled architectural solution to the pair/path conflict.
  - The gain is not just from stronger supervision or more parameters.
  - `dualagg` can be trained stably under the existing semi-supervised candidate-bag regime.
- Appendix can support:
  - split-specific variants
  - subpath-only explanation branch
  - stronger teacher/calibration refinements
  - extra faithfulness/error analyses
- Experiments intentionally cut:
  - pair-level PU
  - retrieval-logit shortlist takeover
  - fully new encoders before `dualagg` is stabilized

## Current Anchor

Current strongest reference systems:

| Split | Variant | Pair AUROC | Pair AUPRC | hard MRR | hard top-1 | hard top-5 | path binary AUROC | path binary AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cold-drug | `tuned` | 0.7744 | 0.8165 | 0.4043 | 0.1809 | 0.7171 | 0.6652 | 0.3125 |
| cold-disease | `binaryrebalance` | 0.7591 | 0.8094 | 0.4064 | 0.1394 | 0.8413 | 0.6811 | 0.3303 |

Current `dualagg` finding:
- the architecture is directionally right
- but the implementation is unstable and currently underperforms best-balanced mainline

This plan is therefore about **making `dualagg` defensible**, not replacing the whole system in one jump.

## Experiment Blocks

### Block 1: Minimal DualAgg With Strong Teacher Initialization
- Claim tested: `dualagg` itself is sound when not overloaded with extra modules.
- Why this block exists: Current `dualagg` mixes too many ideas at once; we need the clean architectural effect first.
- Dataset / split / task: `cold-drug`, `cold-disease`; pair prediction + hard path ranking + controlled path-binary.
- Compared systems:
  - `best-balanced mainline`
  - current `dualagg`
  - `dualagg-minimal-teacherinit`
- Metrics:
  - Primary: Pair AUROC, Pair AUPRC, hard MRR, hard gold_recall@1, hard gold_recall@5
  - Secondary: path-binary AUROC/AUPRC, faithfulness gap
- Setup details:
  - Keep current candidate-bag semi-supervision (`cached_positive_only`)
  - Keep retrieval entrance fixed
  - Remove subpath and in-training binary calibration from the main pair path
  - Initialize `evidence` branch from best-balanced checkpoint
  - Initialize `explanation` branch from strongest path-focused checkpoint (`teacherdistill` or `scheme2`)
- Success criterion:
  - relative to current `dualagg`, hard MRR improves materially
  - pair AUROC/AUPRC recover toward best-balanced baseline
  - no catastrophic collapse on controlled path-binary
- Failure interpretation:
  - if pair still collapses, the architecture alone is not enough and aggregation interaction is mis-specified
  - if path does not improve, teacher transfer is not reaching the explanation aggregator
- Table / figure target: Main paper ablation table, row group “architectural decoupling”
- Priority: MUST-RUN

### Block 2: Stagewise Decoupling Ablation
- Claim tested: `dualagg` needs explanation-protected training rather than fully joint optimization.
- Why this block exists: This is the most direct test of the core conflict hypothesis.
- Dataset / split / task: same as Block 1
- Compared systems:
  - `dualagg-minimal-teacherinit`
  - `dualagg-minimal-teacherinit + frozen explanation in stage3`
  - `dualagg-minimal-teacherinit + very-low-lr explanation in stage3`
- Metrics:
  - Primary: hard MRR, hard gold_recall@1/@5
  - Secondary: pair AUROC/AUPRC, path-binary AUROC/AUPRC
- Setup details:
  - identical retrieval and candidate bags
  - same initialization
  - only vary Stage 3 optimizer policy for explanation aggregator
- Success criterion:
  - frozen/low-lr explanation should dominate fully-joint training on hard ranking
  - pair should not drop more than a small tolerance relative to the non-frozen variant
- Failure interpretation:
  - if stagewise protection does not help, the issue is not optimization interference but aggregation design itself
- Table / figure target: Main paper training strategy ablation
- Priority: MUST-RUN

### Block 3: Post-hoc Binary Calibration For DualAgg
- Claim tested: controlled path-binary can be improved without modifying the pair trunk.
- Why this block exists: current `dualagg` variants often trade hard ranking against path-binary and pair.
- Dataset / split / task: same splits; focus on controlled path-binary benchmark
- Compared systems:
  - best-balanced mainline
  - `dualagg-minimal-teacherinit`
  - `dualagg-minimal-teacherinit + posthoc binary calibrator`
- Metrics:
  - Primary: path-binary AUROC/AUPRC
  - Secondary: hard MRR/top-1/top-5, pair AUROC/AUPRC should remain unchanged
- Setup details:
  - train calibrator after main model converges
  - inputs may use explanation score, evidence score, agreement, reliability, retrieval confidence
  - no gradient from calibrator into pair trunk
- Success criterion:
  - path-binary AUROC/AUPRC improves over raw dualagg explanation/binary scores
  - pair metrics remain identical up to numerical noise
- Failure interpretation:
  - if calibration cannot improve path-binary, the binary signal is not recoverable from current features
- Table / figure target: Main paper or appendix controlled path-binary table
- Priority: MUST-RUN

### Block 4: Simplicity Check Against Overbuilt DualAgg
- Claim tested: the final `dualagg` does not need subpath, online calibration, and multiple in-loop refinements all at once.
- Why this block exists: reviewers will ask whether the architecture only works when heavily engineered.
- Dataset / split / task: at least `cold-drug`; extend to `cold-disease` if promising
- Compared systems:
  - best-performing minimal dualagg
  - minimal dualagg + subpath-only explanation
  - minimal dualagg + in-loop binary calibration
- Metrics:
  - same primary pair/path metrics
- Setup details:
  - hold teacher init and stagewise policy fixed
- Success criterion:
  - minimal or lightly extended version matches or beats overbuilt variants
- Failure interpretation:
  - if larger variant is consistently better, the paper must justify the extra complexity explicitly
- Table / figure target: Appendix simplicity table
- Priority: NICE-TO-HAVE

### Block 5: Agreement and Failure Analysis
- Claim tested: evidence/explanation disagreement is informative, not noise.
- Why this block exists: this is the qualitative diagnosis that makes the paper persuasive.
- Dataset / split / task: both splits; test set only
- Compared systems:
  - best-balanced mainline
  - best-performing dualagg variant
- Metrics:
  - disagreement vs pair error
  - disagreement vs path error
  - faithfulness gap stratified by high/low agreement
  - bag-size / gold-availability strata
- Setup details:
  - no retraining needed
  - pure analysis on stored predictions
- Success criterion:
  - high disagreement should correlate with failure or uncertainty
- Failure interpretation:
  - if disagreement is uninformative, agreement-aware fusion/calibration is weaker as a claim
- Table / figure target: Main paper qualitative figure or appendix diagnostics
- Priority: NICE-TO-HAVE

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Sanity/metric correctness | Verify current baselines and dualagg outputs are reproducible | If metrics drift, stop and fix before new runs | low | stale outputs / overwritten folders |
| M1 | Minimal dualagg teacher-init | `dualagg-minimal-teacherinit` on both splits | Continue only if pair recovers vs current dualagg and hard MRR is competitive with best-balanced | medium | bad checkpoint transfer |
| M2 | Stagewise decoupling | frozen vs low-lr explanation Stage 3 | Keep the better policy only if hard ranking improves without pair collapse | medium | explanation branch under-trains |
| M3 | Post-hoc binary calibration | calibrator on saved predictions/checkpoints | Keep only if path-binary improves and pair stays unchanged | low | no recoverable binary signal |
| M4 | Simplicity check | add subpath or in-loop calibration back one at a time | Only keep extras that clearly improve at least one key metric block | medium | overfitting to one split |
| M5 | Analysis and paper packaging | disagreement, faithfulness, strata | Include only diagnostics that support the main claim | low | noisy narratives |

## Compute and Data Budget
- Total estimated GPU-hours:
  - M1-M3 must-run core: roughly equivalent to 6-8 current full `quick_contrast` jobs
  - M4-M5: another 2-4 job equivalents if warranted
- Data preparation needs:
  - none beyond current processed splits and hard path benchmark outputs
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - stable checkpoint initialization and avoiding output-path collisions

## Risks and Mitigations
- Risk: `dualagg` still underperforms even with teacher init.
- Mitigation: use that result to argue the architecture is not yet mature; do not force it into main results.

- Risk: stagewise decoupling improves path but hurts pair too much.
- Mitigation: report as supporting evidence for the conflict hypothesis and keep best-balanced mainline as the performance anchor.

- Risk: post-hoc calibrator does not rescue controlled path-binary.
- Mitigation: downgrade binary validity to an appendix-level diagnostic rather than main claim.

- Risk: too many variants make the paper story diffuse.
- Mitigation: keep only one best `dualagg` line in the main paper, move the rest to appendix.

## Final Checklist
- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] The architectural claim is separated from teacher/calibration add-ons
- [ ] Nice-to-have runs are separated from must-run runs

