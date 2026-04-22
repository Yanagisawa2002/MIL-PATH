# DualAgg Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| DUA001 | M0 | Reconfirm anchors | `tuned` | cold-drug | Pair + hard path + binary | MUST | DONE | current best-balanced drug anchor |
| DUA002 | M0 | Reconfirm anchors | `binaryrebalance` | cold-disease | Pair + hard path + binary | MUST | DONE | current best-balanced disease anchor |
| DUA003 | M0 | Reconfirm architectural baseline | current `dualagg` | both | Pair + hard path + binary | MUST | DONE | directionally right, underperforms anchor |
| DUA010 | M1 | Minimal teacher-init dualagg | `dualagg-minimal-teacherinit` | cold-drug | Pair AUROC/AUPRC, hard MRR/top-1/top-5 | MUST | TODO | evidence init from `tuned`, explanation init from `teacherdistill` |
| DUA011 | M1 | Minimal teacher-init dualagg | `dualagg-minimal-teacherinit` | cold-disease | Pair AUROC/AUPRC, hard MRR/top-1/top-5 | MUST | TODO | evidence init from `binaryrebalance`, explanation init from `teacherdistill` |
| DUA020 | M2 | Stagewise decoupling | `dualagg-minimal-teacherinit + frozen explanation` | cold-drug | Pair + hard path | MUST | TODO | freeze explanation agg in Stage 3 |
| DUA021 | M2 | Stagewise decoupling | `dualagg-minimal-teacherinit + frozen explanation` | cold-disease | Pair + hard path | MUST | TODO | freeze explanation agg in Stage 3 |
| DUA022 | M2 | Stagewise decoupling | `dualagg-minimal-teacherinit + low-lr explanation` | cold-drug | Pair + hard path | MUST | TODO | compare against full freeze |
| DUA023 | M2 | Stagewise decoupling | `dualagg-minimal-teacherinit + low-lr explanation` | cold-disease | Pair + hard path | MUST | TODO | compare against full freeze |
| DUA030 | M3 | Post-hoc calibration | `dualagg-minimal-teacherinit + posthoc calibrator` | cold-drug | path-binary AUROC/AUPRC | MUST | TODO | no gradient to pair trunk |
| DUA031 | M3 | Post-hoc calibration | `dualagg-minimal-teacherinit + posthoc calibrator` | cold-disease | path-binary AUROC/AUPRC | MUST | TODO | no gradient to pair trunk |
| DUA040 | M4 | Simplicity check | `dualagg-minimal + subpath explanation` | cold-drug | Pair + path | NICE | TODO | only if M1/M2 promising |
| DUA041 | M4 | Simplicity check | `dualagg-minimal + in-loop binary calibration` | cold-drug | Pair + path-binary | NICE | TODO | only if M3 insufficient |
| DUA050 | M5 | Diagnostics | `best dualagg` vs `best-balanced mainline` | both | disagreement, faithfulness, strata | NICE | TODO | analysis only |
