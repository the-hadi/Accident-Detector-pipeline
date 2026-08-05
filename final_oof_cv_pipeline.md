# Final OOF cross-validation — A2-MP and D1

> Status: pipeline prepared; implementation and long CPU execution require
> confirmation. E1 development diagnostic did not pass its gate, so this is a
> comparison of the two best **single** video-only models, not an ensemble run.

## Purpose

The fixed 480/120 development split has been used for controlled ablations.
The final defensible comparison must instead provide exactly one out-of-fold
(OOF) full-MP4 probability for each of the 600 videos.

Finalists:

```text
A2-MP: frozen ResNet18 RGB features + mean/max pooling
D1:    frozen ResNet18 RGB + frame-difference motion feature fusion
```

Development reference only:

| model | F1 | Recall | PR-AUC |
|---|---:|---:|---:|
| A2-MP | 0.738 | 0.867 | 0.723 |
| D1 | 0.739 | 0.850 | 0.741 |

The equal-mean E1 diagnostic was weaker, and the AND rule violated the
Recall >= 0.85 safety constraint. No ensemble is carried into final OOF.

## Frozen CV protocol

- Source: `manifests_v3/cv_folds_v3.csv`.
- Five outer stratified video-level folds.
- Each outer validation fold: 120 videos, exactly 60 accident and 60
  non-accident.
- Each outer train partition: 480 videos.
- From each outer train partition only, make a deterministic stratified inner
  validation subset (approximately 48 videos) for early stopping.
- All windows of a video remain in the same fold. No video ID, path, event
  timestamp, technical metadata, or D2 fields is a model input.

## Fold-local window and hard-negative policy

For every outer fold independently:

1. Create positive core/context and ordinary negative training windows from
   **only the 480 outer-train videos**.
2. Mine hard negatives only from outer-train negative windows using that
   fold's A2-MP checkpoint/inner-training model.
3. Construct the final weighted A2/D1 training rows only after mining.
4. Use all sliding windows from its 120 outer-validation MP4s for full-video
   inference and top-3-mean aggregation.

The existing global development hard-negative list is not reused as a training
decision in any outer fold.

## Resumable feature cache plan

The deterministic union of all sliding sequences for the selected 600 videos
is made first. Existing compatible caches are reused:

```text
RGB cache already available:    3,214 sequences (partial coverage)
Motion cache already available: 3,214 sequences (partial coverage)
```

All missing RGB and motion sequences are decoded/encoded exactly once into
separate resume-safe caches. Every partial cache stores:

- union-manifest SHA-256;
- encoder/checkpoint and preprocessing versions;
- completed `sequence_id -> [16,512]` features;
- atomic save after a fixed number of batches.

The script must show the missing count and CPU-time estimate before full
motion extraction starts. Failed decodes are written to CSV, never silently
dropped. No two processes may write the same partial cache.

The D1 motion extension is the expensive part; based on the previous CPU run,
it may take many hours. Model-head training after caching is comparatively
short.

## Per-fold training and prediction

For each base model and outer fold:

```text
fold-local train windows/features
 -> head training with inner validation early stopping
 -> score every sliding window in outer validation MP4s
 -> fixed top-3 mean aggregation
 -> save one raw OOF video probability
```

Save checkpoint, fold manifest checksums, selected epoch, window predictions,
video prediction, and inference time. No outer-validation threshold is used
for checkpoint selection or model tuning.

## Threshold, calibration, statistics

- Concatenate the five raw OOF prediction tables per base model: 600 rows,
  one per video.
- Select a single model threshold from all 600 OOF probabilities to maximize
  accident F1 subject to Recall >= 0.85. Report threshold 0.5 separately.
- Estimate 95% CI for F1, Recall and PR-AUC using 2,000 stratified bootstrap
  resamples.
- Compare A2-MP and D1 with paired bootstrap and exact McNemar test.
- If probability calibration is reported, use cross-fitted Platt calibration:
  calibrator for each fold is fit on OOF predictions from the other four folds
  and applied only to its held-out fold.

## Outputs

```text
manifests_v3/final_oof_union_sequences.csv
manifests_v3/final_oof_fold_{0..4}_train_windows.csv
manifests_v3/final_oof_fold_{0..4}_hard_negatives.csv
processed_v3/final_oof_rgb_features_partial.pt
processed_v3/final_oof_rgb_features.pt
processed_v3/final_oof_motion_features_partial.pt
processed_v3/final_oof_motion_features.pt
models_v3/final_oof_{a2mp,d1}_fold_{0..4}.pt
predictions_v3/final_oof_{a2mp,d1}_windows.csv
predictions_v3/final_oof_{a2mp,d1}_videos.csv
reports_v3/final_oof_model_comparison.csv
reports_v3/final_oof_bootstrap_statistics.csv
reports_v3/final_oof_paired_tests.json
reports_v3/final_oof_calibration_report.json
reports_v3/final_model_selection.md
```

## Final model hand-off

Only after OOF selection, train the chosen video-only model on all 600 videos
with the frozen recipe and build a reusable `predict_full_mp4(...)` function.
The final inference package reports the chosen OOF threshold, a probability,
and the collision/no-collision label for an arbitrary MP4.

## Acceptance gate

Select D1 only if its OOF F1/PR-AUC benefit over A2-MP is stable, Recall is at
least 0.85, and paired analysis does not contradict the improvement.
Otherwise select the simpler A2-MP. If the difference is inconclusive, report
both and choose the simpler/faster model for the default inference package.
