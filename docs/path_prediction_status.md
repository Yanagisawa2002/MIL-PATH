# Path Prediction Status

This note summarizes what the current experiments already show about path prediction quality, what remains weak, and what should be evaluated next.

## 1. Current main conclusion

The project already shows a clear pair-level gain from mechanistic modeling, but **path recovery is not yet as strong as pair prediction**.

In particular:

- The current mainline is strongest on pair prediction.
- The `gold_only` variant is usually stronger on hard gold-path recovery.
- Easy path metrics are nearly saturated and are **not sufficient** to claim strong rationale recovery.
- Hard path metrics are the more trustworthy indicators of explanation quality.

## 2. Mainline vs gold_only

### cold-drug

| Model | AUROC | AUPRC | hard MRR | hard top-1 | hard top-5 | hard top-10 |
|---|---:|---:|---:|---:|---:|---:|
| `gold_only` | 0.7144 | 0.7590 | 0.3940 | 0.1414 | 0.8158 | 0.9803 |
| `mainline` | 0.7756 | 0.8178 | 0.3599 | 0.1283 | 0.6908 | 0.9967 |

Sources:
- [cold-drug gold_only tuned](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_drug_pairfeat_cuda_b128_interaction/quick_contrast_gold_mech_interaction_goldonly_stage30_pathmetrics_tuned/contrast_summary.json)
- [cold-drug mainline](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_drug_pairfeat_cuda_b128_interaction/quick_contrast_gold_mech_interaction_cached_posonly_stage30_sweep310_cached/contrast_summary.json)

### cold-disease

| Model | AUROC | AUPRC | hard MRR | hard top-1 | hard top-5 | hard top-10 |
|---|---:|---:|---:|---:|---:|---:|
| `gold_only` | 0.6811 | 0.7165 | 0.3898 | 0.1635 | 0.8029 | 0.9808 |
| `mainline` | 0.7690 | 0.8133 | 0.3233 | 0.1106 | 0.5673 | 0.9808 |

Sources:
- [cold-disease gold_only](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_disease/quick_contrast_gold_mech_interaction_goldonly_stage30_pathmetrics/contrast_summary.json)
- [cold-disease mainline](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_disease/quick_contrast_gold_mech_interaction_cached_posonly_stage30_sweep310_cached/contrast_summary.json)

## 3. What these numbers mean

### What is already supported

- The model **does use mechanistic paths**.
- Easy-bag faithfulness gaps are large, so removing the top path hurts prediction much more than removing a random non-top path.
- Under harder candidate sets, gold paths still frequently remain in the top-5/top-10.

### What is still weak

- Hard top-1 rationale recovery is still limited.
- Hard MRR is only moderate.
- The best pair-prediction setting is not the same as the best gold-path recovery setting.

So the most accurate statement is:

> The current model already benefits from mechanistic evidence for pair prediction, but rationale recovery under hard candidates is still an open bottleneck.

## 4. Why easy metrics are not enough

Easy path metrics are nearly saturated:

- `cold-drug mainline`: easy MRR `0.9956`, easy top-1 recall `0.9770`
- `cold-disease mainline`: easy MRR `1.0000`, easy top-1 recall `0.9808`

These numbers are inflated because the easy candidate bags are very small and often contain very few strong distractors.

By contrast, hard candidate bags are much more informative:

- `cold-drug`: mean hard bag size `11.24`
- `cold-disease`: mean hard bag size `11.86`

Sources:
- [cold-drug mainline](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_drug_pairfeat_cuda_b128_interaction/quick_contrast_gold_mech_interaction_cached_posonly_stage30_sweep310_cached/contrast_summary.json)
- [cold-disease mainline](/D:/Models/NeurIPS/Try_5/outputs/full_fast_cold_disease/quick_contrast_gold_mech_interaction_cached_posonly_stage30_sweep310_cached/contrast_summary.json)

## 5. Why mainline can beat gold_only on pair prediction but lose on hard path ranking

This is the key trade-off:

- `gold_only` is more specialized for **recovering the labeled gold rationale**.
- `mainline` is optimized for **final pair prediction with broader mechanistic coverage**, including positive pairs without gold rationales.

As a result:

- `mainline` usually gives better `AUROC/AUPRC`
- `gold_only` usually gives better `hard MRR` and `hard top-1/top-5 recall`

This is not necessarily a bug. It reflects a genuine objective mismatch between:

- precise rationale recovery
- broad weakly supervised mechanistic support for pair prediction

## 6. What should be emphasized in future evaluation

If path prediction is to be a main contribution, the core metrics should be:

- `hard MRR`
- `hard gold_recall@1`
- `hard gold_recall@5`
- `faithfulness gap`

`hard gold_recall@10` can still be reported, but it is close to saturation and is less discriminative.

## 7. Most valuable next experiments

Priority order:

1. **Report path prediction as a first-class result**
   - Put `hard MRR`, `hard top-1`, `hard top-5`, and `faithfulness gap` in the main results table.

2. **Run path-focused ablations**
   - Compare `gold_only` vs mainline explicitly as a pair/path trade-off.

3. **Improve Stage-2 for path recovery**
   - Focus on hard-negative design and path-focused refinement, not just pair AUROC.

4. **Keep easy metrics only as supporting evidence**
   - They can show that the model uses Mech, but they should not be the main evidence for strong rationale recovery.

## 8. Bottom line

The current system already proves:

- mechanistic modeling improves pair prediction
- the model is not merely attaching explanations after the fact

But it does **not yet** prove:

- strong top-1 rationale recovery under hard candidates

So path prediction should now be treated as the main gap to close.
