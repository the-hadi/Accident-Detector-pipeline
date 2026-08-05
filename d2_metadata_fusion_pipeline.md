# D2 — conservative metadata-only and video+metadata fusion

> Status: pipeline prepared; implementation and training require confirmation.
> D2 is a dataset benchmark, not a replacement for the video-only inference
> model, because arbitrary new MP4s normally do not include these annotations.

## Objective

Test whether the three human annotations available in this dataset provide
information that genuinely complements video evidence, while explicitly
checking whether they create a shortcut:

```text
D2-1  weather + light_conditions + scene -> metadata-only probability
D2-2  RGB + motion cached video features -> video-only control
D2-3  RGB + motion cached video features + metadata -> fusion probability
```

The required real-world contract remains unchanged:

```text
input: complete arbitrary MP4
default output: accident probability from a video-only model
```

D2 can only be used when all three metadata fields are supplied by a trusted
external source. It must never make the main MP4-only inference pipeline depend
on manual metadata entry.

## Frozen inputs and exclusions

### Permitted metadata

Only these categorical fields are candidates:

- `weather`: Clear, Cloudy, Rain, Snow;
- `light_conditions`: Normal, Twilight, Dark, Bright;
- `scene`: Urban, Highway, Sub-urban, Other, Industrial, Rural.

They come from `video_manifest_v2.csv`, are joined only by `video_id`, and are
fit/encoded from the 480 train videos only. A category missing in training is
represented by an explicit unknown value at validation/inference.

### Strict exclusions

The following never enter a D2 model:

- `time_of_event` or `time_to_accident` — direct/near-direct label leakage;
- video id, file name, path, split, file hash;
- duration, fps, frame count, size, resolution, codec/technical audit fields —
  potential source/capture shortcuts rather than accident evidence;
- labels, predictions, or thresholds created from validation data.

## Data and cache contract

- Fixed V3 development split: 480 train / 120 validation videos.
- Training windows: the existing 1,446 rows in
  `manifests_v3/a2mp_hn1_train_windows.csv` (positive core, ordinary negative,
  and round-1 hard negatives).
- Validation: all 1,768 sliding windows from the 120 complete validation MP4s.
- Video features: `processed_v3/a2mp_hn1_features.pt`.
- Motion features: `processed_v3/d1_motion_features.pt`.

Both caches must contain the same 3,214 sequence IDs with shape `[16, 512]`.
No MP4 decode, frame extraction, or feature extraction is allowed in D2.

## Feature construction

For every 16-frame window, make the same frozen video representation used by
D1:

```text
RGB [16, 512]       -> mean + max -> [1024]
Motion [16, 512]    -> mean + max -> [1024]
concatenate                         -> [2048]
```

Metadata is a train-fitted one-hot vector. The vocabulary and field order are
saved with the checkpoint for reproducibility.

### D2-1 — metadata only

One row per video; train a regularized logistic regression on the metadata
vector. This is deliberately simple so unusually strong performance is easy to
recognise as dataset bias rather than a video understanding result.

### D2-2 — video-only control

Train a small `LayerNorm -> Dropout -> Linear` head on the `[2048]` frozen
video feature with the exact same training windows and loss weights as D2-3.
This controls for any difference caused by the new training script.

### D2-3 — fusion

```text
[2048 video feature | one-hot metadata]
 -> LayerNorm -> metadata-vector dropout (p=0.20 during train only)
 -> Dropout(0.35) -> Linear -> window logit
```

The deliberately low-capacity fusion head makes it harder to memorize rare
metadata categories. Metadata dropout zeros the complete metadata vector for a
random subset of training windows; validation is deterministic.

## Training, evaluation, and anti-leakage rules

- `AdamW`, lr `3e-4`, weight decay `1e-4`, batch size 64, maximum 30 epochs.
- Existing A2-MP-HN1 window loss weights are used unchanged.
- Video heads use the frozen `top3_mean` aggregation for checkpoint selection
  through validation video-level PR-AUC.
- Threshold is selected on development validation for maximum accident F1 with
  Recall >= 0.85; it is not a test/final threshold.
- `max`, `mean`, `top2_mean`, `top3_mean`, `top5_mean`, `noisy_or`, and
  `logsumexp` are stored only as aggregation ablations.
- The metadata-only model is evaluated at one prediction per video.
- Final claims still need fold-local preprocessing and five-fold OOF evaluation.

## Required diagnostics

1. Compare D2-1, D2-2, D2-3 at MP4/video level: F1, Recall, Precision,
   PR-AUC, ROC-AUC, confusion matrix.
2. Save category counts and per-category errors. Categories with fewer than 10
   validation videos are descriptive only, not evidence.
3. If D2-1 alone is unexpectedly competitive with D1/A2-MP, label this as a
   shortcut/bias warning before interpreting fusion gains.
4. Compare fusion FP/FN decisions with A2-MP and D1 to assess complementarity.

## Outputs

```text
models_v3/d2_video_only_frozen_best.pt
models_v3/d2_video_metadata_fusion_frozen_best.pt
models_v3/d2_training_history.csv
predictions_v3/d2_metadata_only_validation_video_predictions.csv
predictions_v3/d2_video_only_validation_window_predictions.csv
predictions_v3/d2_video_metadata_fusion_validation_window_predictions.csv
predictions_v3/d2_video_metadata_fusion_validation_video_predictions.csv
reports_v3/d2_metadata_vocab.json
reports_v3/d2_metadata_only_metrics.json
reports_v3/d2_video_metadata_aggregation_ablation.csv
reports_v3/d2_metadata_bias_report.csv
reports_v3/d2_summary.json
```

## Decision gate

- D2 never replaces the MP4-only default model.
- Fusion is shortlist-worthy only if its frozen full-MP4 result improves the
  video-only control/D1 without breaking Recall >= 0.85, and its benefit does
  not come solely from a suspicious metadata-only shortcut.
- Otherwise D2 is retained as a transparent dataset-specific ablation.
