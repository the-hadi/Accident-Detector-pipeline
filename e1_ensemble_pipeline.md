# E1 — conservative full-MP4 video-only ensemble

> Status: final fold-safe OOF validation completed on 2026-08-05. E1-AND was
> retained as a documented Accuracy-first diagnostic, but was not promoted to
> the default final model because its Recall fell below the required safety
> level and its small Accuracy advantage was not statistically significant.
> The default ensemble is strictly video-only and accepts an arbitrary complete
> MP4. D2 metadata fusion is excluded from that default because metadata is not
> normally available for a new video.

## Why ensemble now

The completed models make different errors on the same 120 validation MP4s:

- A2-MP: strong RGB reference; F1 0.738, Recall 0.867.
- D1: RGB + motion; F1 0.739, Recall 0.850, higher PR-AUC.
- A6: high Recall (0.950) but too many false positives; not a standalone
  finalist, possibly a recall-oriented diagnostic candidate only.
- D2 fusion: improves recall to 0.900 but adds false positives and requires
  annotations; dataset-only ablation, not part of the default ensemble.

E1 tests whether **A2-MP and D1** provide a stable video-only improvement,
not whether several models can be overfit to the one development split.

## Input/output contract

```text
input: one complete MP4
  -> A2-MP sliding-window inference
  -> D1 sliding-window RGB+motion inference
  -> each model's frozen top-3-mean video probability
  -> calibrated video-level ensemble probability
output: accident / no-accident probability and thresholded decision
```

No filename, ID, event time, technical metadata, weather, light condition, or
scene is used by the default E1 model.

## Two separate evaluation levels

### E1-D — development diagnostic (cheap)

On the existing fixed 120 validation MP4s only, create these **pre-registered,
non-learned** diagnostics:

1. A2-MP alone;
2. D1 alone;
3. equal-probability mean of A2-MP and D1;
4. conservative positive agreement (`A2 >= threshold AND D1 >= threshold`);
5. recall-oriented positive agreement (`A2 >= threshold OR D1 >= threshold`).

This checks error complementarity and gives a reproducible illustration, but
must not select learned weights, calibration, or the final production threshold.
Any score reported here is development-only.

A6 is evaluated only as an explicitly separate recall ablation; it is not
included in the default ensemble unless OOF results demonstrate an improvement
without unacceptable false positives.

### E1-F — final OOF ensemble (required before a final claim)

Use the frozen five-fold `cv_folds_v3.csv`:

1. For every outer fold, construct train windows and hard negatives using only
   its 480 outer-train videos.
2. Train A2-MP and D1 independently on that outer-train set, using an inner
   split only for early stopping.
3. Score all 120 outer-validation **complete MP4s**.
4. Save one OOF probability per video per base model; every one of the 600
   videos must occur exactly once as outer validation.
5. Fit Platt calibration and a regularized logistic stacking head only from
   fold-safe training/OOF partitions; never train it on the same OOF row it
   predicts.
6. Choose one final ensemble threshold only from the complete saved OOF table.

The final output is 600 OOF predictions, not repeated scores from the fixed
development validation set.

## CPU and cache plan

The development diagnostic uses already stored prediction CSVs and runs in
seconds.

Final OOF D1 needs motion features for all sliding sequences used by the five
folds. The existing D1 cache covers 3,214 sequences but not necessarily every
fold's full validation scope. Before a long run the script must:

- make a deterministic union manifest of needed sequence IDs;
- check and reuse all compatible RGB/motion features;
- extract only missing motion features with resumable atomic partial saves;
- show an estimate before starting the full extraction;
- never run two feature writers against the same cache.

This preserves the ability to stop and resume safely. The OOF stage is CPU
expensive; it is not started automatically by the development diagnostic.

## Calibration and thresholding

- Raw probabilities from different heads must not be averaged as a final,
  calibrated probability without OOF evidence.
- Development equal mean is an analysis only.
- Final calibration: Platt scaling on appropriately separated OOF folds.
- Final threshold objective: maximum accident F1 subject to Recall >= 0.85.
- Report fixed-threshold 0.5 results separately from the selected threshold.

## Required analysis

- F1, Recall, Precision, PR-AUC, ROC-AUC, calibration/Brier score, confusion
  matrix and inference time at MP4 level.
- 2,000-resample stratified bootstrap 95% CI for F1, Recall and PR-AUC.
- paired bootstrap and exact McNemar test: A2-MP vs D1, and best base model vs
  E1.
- FP/FN agreement table and montage; ensemble is useful only if it resolves
  enough errors without exchanging them for a larger number of new errors.

## Outputs

```text
reports_v3/e1_development_ensemble_comparison.csv
reports_v3/e1_development_ensemble_summary.json
predictions_v3/e1_development_ensemble_validation_predictions.csv
manifests_v3/e1_oof_required_sequences.csv
processed_v3/e1_motion_features_partial.pt
processed_v3/e1_motion_features.pt
predictions_v3/e1_oof_base_predictions.csv
predictions_v3/e1_oof_ensemble_predictions.csv
reports_v3/e1_oof_calibration_report.json
reports_v3/e1_oof_bootstrap_statistics.csv
reports_v3/e1_oof_paired_model_tests.json
models_v3/e1_platt_and_stacking_parameters.json
```

## Gate

- If the development diagnostic does not show useful complementary behaviour,
  do not spend CPU time on OOF extraction/training; keep A2-MP or D1 as the
  single video-only model.
- If it is promising, run E1-F and compare OOF confidence intervals and paired
  tests. E1 is selected only if its OOF benefit is stable and Recall remains at
  least 0.85.
- D2 metadata may be reported as a separate dataset benchmark only; it cannot
  be silently included in the arbitrary-MP4 default ensemble.

## Completion record — 2026-08-05

The existing final OOF A2-MP and D1 checkpoints plus their RGB/motion feature
caches were reused. No video was decoded and no base model was retrained.
For each of the five outer folds, the 48-video inner validation partition alone
selected the thresholds; predictions were then measured on that fold's 120
untouched outer-validation videos.

| Decision policy | OOF Accuracy | Recall | F1 | Decision |
|---|---:|---:|---:|---|
| E1-AND, Accuracy-first | 66.67% | 65.00% | 0.661 | Not promoted: low Recall |
| D1, nested Accuracy threshold | 66.17% | 68.67% | 0.670 | Reference for Accuracy-only comparison |
| D1, deployed F1/Recall policy | 63.33% | 87.67% | 0.705 | Default final model |
| E1-AND, inner safety-constrained | 62.00% | 83.33% | 0.687 | Rejected: Recall still < 85% |

The paired exact McNemar comparison of E1-AND Accuracy-first against nested
Accuracy-threshold D1 has p=0.813. The 0.50 percentage-point Accuracy
difference is therefore not evidence of a stable improvement.

Outputs:

```text
predictions_v3/e1_final_oof_ensemble_predictions.csv
predictions_v3/e1_final_oof_inner_predictions.csv
reports_v3/e1_final_oof_accuracy_comparison.csv
reports_v3/e1_final_oof_fold_thresholds.csv
reports_v3/e1_final_oof_bootstrap_statistics.csv
reports_v3/e1_final_oof_paired_tests.json
reports_v3/e1_final_oof_report.md
reports_v3/e1_final_oof_summary.json
```
