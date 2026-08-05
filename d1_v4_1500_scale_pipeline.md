# D1-V4 — scale from 600 to all 1,500 labelled training videos

## Goal

Measure whether the final deployable D1 architecture improves when its
supervised training set increases from 600 balanced videos to the complete
balanced `train` split of 1,500 videos (750 accident, 750 non-accident).

This is **not** a new-model claim. The first V4 experiment keeps D1's model
contract fixed so that any change can be attributed primarily to more data:

```text
complete MP4
  -> 5-second sliding windows, 2.5-second stride
  -> 16 RGB frames and 16 adjacent-frame-difference motion images
  -> shared frozen ImageNet ResNet18 encoder
  -> mean/max RGB + mean/max motion fusion head
  -> top-3 window mean at video level
```

## Preflight snapshot — 2026-08-05

| Item | Observed value |
|---|---:|
| Labelled `train` metadata | 1,500 videos: 750 positive / 750 negative |
| Already downloaded | 600 videos: 300 positive / 300 negative |
| New downloads required | 900 videos: 450 positive / 450 negative |
| Mean current MP4 size | 16.09 MB |
| Estimated new raw-video space | 14.14 GB |
| Free space on `P:` | 33.75 GB |
| Existing V3 cache | retained; never overwritten |

The expected disk space is sufficient. V4 outputs must use new names and
must never delete or overwrite the completed V2/V3 data, caches, models, or
reports.

## Realistic CPU time estimate

This estimate is based on the recorded V3 CPU run, not on an ideal GPU system.
Actual network speed and CPU load can change it.

| Stage | Estimated time |
|---|---:|
| Download 900 MP4 files (4 workers) | 1–2 hours |
| Video audit, manifest and windows | under 2 hours |
| New RGB + motion feature cache | about 28–35 hours |
| Five-fold nested OOF head training/evaluation | about 50–80 hours |
| Final all-data training, inference hand-off and reports | 3–6 hours |
| **Total wall-clock time** | **roughly 4–6 days** |

The cache stage will be resumable with atomic partial saves. The OOF stage
will be resumable at fold boundaries. VS Code may be closed, but the launched
background Python process must not be terminated; the computer must also not
sleep during active work.

## Data contract

1. Select all 750 positives and all 750 negatives from `train_metadata.csv`.
2. Keep already valid local MP4 files; download only missing files.
3. Audit every local video; save duration, FPS, dimensions, validity and error
   reason in `video_manifest_v4_1500.csv`.
4. Derive `label = 1` only from the training metadata's `time_of_event` and
   never pass event time, path, name, ID, weather/light/scene or technical
   metadata to D1 as a feature.
5. Preserve invalid records with an error reason; do not silently remove them.

## Evaluation contract

1. Freeze a new stratified five-fold video-level split over all 1,500 videos.
   Each outer validation fold contains 300 videos, balanced 150/150.
2. Within each outer-training fold, reserve a stratified 10% inner-validation
   set for early stopping/checkpoint selection and threshold selection.
3. Train the hard-negative miner and D1 only on the fold's training videos.
4. Save exactly one outer-OOF probability per video.
5. The primary requested metric is Accuracy. Report Recall, F1, PR-AUC,
   ROC-AUC, confusion matrix and 2,000-resample confidence intervals as well.
6. Compare D1-V4 with the completed 600-video D1 only as separate experiments;
   their OOF tables have different data scopes and must not be treated as a
   paired significance test.

## Cache and resource policy

- Reuse compatible D1 V3 feature records for the existing 600 videos when the
  sequence definitions match; extract only sequences belonging to the 900 new
  videos.
- Save V4 RGB and motion caches under `processed_v4/`; retain V3 caches.
- Decode RGB once per sequence whenever possible, then create the motion
  differences from those decoded frames before feature encoding.
- Save a partial cache every fixed number of batches with an atomic rename.
- Before the full run, process two sequences as a smoke test and calculate a
  revised time/space estimate. Stop if any decode or shape validation fails.

## Outputs

```text
P:\NexarCollisionData\selected_train_1500.csv
P:\NexarCollisionData\video_manifest_v4_1500.csv
P:\NexarCollisionData\manifests_v4\cv_folds_v4_1500.csv
P:\NexarCollisionData\manifests_v4\d1_v4_union_sequences.csv
P:\NexarCollisionData\processed_v4\d1_v4_rgb_features_partial.pt
P:\NexarCollisionData\processed_v4\d1_v4_motion_features_partial.pt
P:\NexarCollisionData\processed_v4\d1_v4_rgb_features.pt
P:\NexarCollisionData\processed_v4\d1_v4_motion_features.pt
P:\NexarCollisionData\models_v4\d1_v4_fold_*.pt
P:\NexarCollisionData\predictions_v4\d1_v4_oof_predictions.csv
P:\NexarCollisionData\reports_v4\d1_v4_oof_summary.json
P:\NexarCollisionData\reports_v4\d1_v4_report.md
```

## Go / no-go gate

Start the multi-day pipeline only after explicit confirmation. It is worth
running when the goal is a stronger, more statistically stable D1 result; it
does not guarantee 80% Accuracy. If V4 does not improve sufficiently, inspect
its error distribution before changing architecture or adding optical flow.
