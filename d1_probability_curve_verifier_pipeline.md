# D1-V3.1 — probability-curve verifier (no new videos)

## Objective

Test a lightweight video-level verifier that can reduce D1's false positives
without downloading data, decoding MP4s, or retraining ResNet18. The verifier
receives only the ordered D1 window probabilities for one complete MP4.

It must not receive a filename, video ID, event time, path, duration, FPS,
weather, light, scene, or any other metadata.

## Frozen inputs

- `predictions_v3/final_oof_base_windows.csv`: D1 window predictions for the
  600-video final OOF protocol.
- `models_v3/final_oof_d1_fold_*.pt` and the existing RGB/motion feature
  caches: used only to score the 48 inner-validation videos in each fold.
- The original outer-fold split remains unchanged.

## Per-video curve features

For the ordered D1 probabilities within one video, create only these eight
features:

1. maximum probability;
2. top-3 mean probability;
3. mean probability;
4. standard deviation;
5. maximum minus mean (peak prominence);
6. mean probability of the windows adjacent to the highest-probability window;
7. fraction of windows with probability at least 0.50;
8. mean absolute change between adjacent window probabilities.

These features describe the persistence and shape of suspicious evidence,
without using its absolute time position.

## Fold-safe protocol

For each of the five outer folds:

1. Score the 48 inner-validation videos using only the existing D1 head for
   that fold; these videos were not used for that head's train windows.
2. Fit a fixed `StandardScaler + LogisticRegression(C=0.1, L2)` verifier on
   only those 48 curve-feature rows.
3. Select its Accuracy-maximising threshold on the same inner set. This is a
   deliberately small, fast experiment; the small inner set is reported as a
   limitation.
4. Apply the frozen verifier and threshold once to the outer fold's 120 D1
   OOF curves.

The outer-validation labels are never used while fitting or thresholding their
verifier. The final table therefore contains exactly one verifier prediction
per one of the 600 videos.

## Evaluation and decision gate

- Primary: Accuracy.
- Also report Precision, Recall, F1, confusion matrix, bootstrap 95% CIs and
  exact McNemar comparison against D1 with fold-local Accuracy thresholds.
- Keep D1 as the deployable default unless the verifier's Accuracy improvement
  is stable and does not create an unacceptable Recall loss.
- This experiment cannot credibly promise 80% Accuracy; it is a low-cost test
  for a modest false-positive reduction.

## Expected resources

- No extra video/disk download.
- No feature extraction or ResNet training.
- Expected CPU wall-clock time: about 10–15 minutes on the current system.

## Outputs

```text
predictions_v3/d1_curve_verifier_inner_window_predictions.csv
predictions_v3/d1_curve_verifier_oof_predictions.csv
reports_v3/d1_curve_verifier_fold_thresholds.csv
reports_v3/d1_curve_verifier_comparison.csv
reports_v3/d1_curve_verifier_bootstrap.csv
reports_v3/d1_curve_verifier_paired_tests.json
reports_v3/d1_curve_verifier_report.md
reports_v3/d1_curve_verifier_chart.png
```
