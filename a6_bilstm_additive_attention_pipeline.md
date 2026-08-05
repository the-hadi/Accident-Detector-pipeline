# A6 — ResNet18 + BiLSTM + additive attention + FFN

> Status: approved for a CPU-only development experiment. This document is the
> contract before implementation; it keeps the data scope and evaluation
> comparable with D1 and A2-MP-HN1.

## Objective

The current RGB pooling models can find strong visual evidence in a window,
but do not explicitly model the order of the 16 frames. A6 tests whether a
small recurrent temporal head improves **whole-MP4 accident/no-accident
classification** without re-decoding videos or training a large video model.

```text
complete MP4
  -> 5 s sliding windows, stride 2.5 s
  -> 16 RGB frame features per window, each 512-D
  -> BiLSTM -> additive temporal attention -> FFN
  -> window probability
  -> fixed top-3 mean aggregation -> MP4 probability
```

This is offline classification of a complete video. BiLSTM can use context on
both sides of a frame because the task is not an online pre-crash warning
system.

## Frozen data and split contract

- Development split remains the frozen 480 train / 120 validation videos.
- Training windows are exactly `manifests_v3/a2mp_hn1_train_windows.csv`:
  603 positive-core, 603 ordinary negative, and 240 round-1 hard negatives.
- Validation uses every sliding window from all 120 complete validation MP4s
  (1,768 windows), then aggregates at video level.
- No frame, window, or video crosses train/validation. `video_id`, filename,
  path, event timestamp, alert timestamp, and metadata are never model inputs.
- In eventual five-fold final evaluation, hard-negative selection and training
  window construction must be repeated inside each outer-train fold.

## Reused feature cache — no new MP4 decoding

The experiment only accepts the existing frozen RGB cache:

```text
processed_v3/a2mp_hn1_features.pt
```

It contains the exact required 3,214 sequences, each shaped `[16, 512]`, with
the established contract:

- RGB after replicated-edge letterbox to 224 x 320;
- ImageNet normalization;
- frozen ImageNet ResNet18 encoder;
- 16 timestamps per five-second window.

The script validates cache compatibility and required sequence IDs before
training. It does not read MP4 files, create another frame cache, or overwrite
the RGB cache.

## A6 architecture

For one window `X` of shape `[16, 512]`:

```text
X
 -> LayerNorm(512)
 -> one-layer BiLSTM(hidden_size=128 per direction)
 -> H: [16, 256]
 -> additive attention: score_t = v^T tanh(W H_t + b)
 -> alpha = softmax(score) over the 16 time steps
 -> z = sum(alpha_t * H_t)                 # [256]
 -> LayerNorm -> Linear(256, 128) -> GELU -> Dropout(0.35) -> Linear(128, 1)
```

Only this temporal head is trainable; ResNet18 remains frozen. There is no
sequence reversal, frame shuffle, or per-frame label. Every 16-frame sequence
has one window label during training, and complete-MP4 prediction is obtained
only after sliding-window aggregation.

## Training and selection

- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Batch size: 64 feature sequences; CPU-friendly because no image decoder or
  CNN runs during training.
- Maximum 45 epochs, early stopping patience 8.
- Loss: weighted `BCEWithLogitsLoss` using the existing A2-MP-HN1 loss
  weights, including hard-negative weight 1.5 and the fixed positive balance.
- Checkpoint selection: validation video-level PR-AUC using the **frozen
  primary aggregation `top3_mean`**.
- Threshold: selected only on development validation to maximize accident F1
  subject to accident Recall >= 0.85.

`max`, `mean`, `top2_mean`, `top3_mean`, `top5_mean`, `noisy_or`, and
`logsumexp` are saved as exploratory aggregation ablations. They are not
allowed to replace the frozen primary result without confirmation in final CV.

## Required outputs

```text
models_v3/a6_bilstm_additive_attention_frozen_best.pt
models_v3/a6_bilstm_additive_attention_training_history.csv
predictions_v3/a6_bilstm_additive_attention_validation_window_predictions.csv
predictions_v3/a6_bilstm_additive_attention_validation_video_predictions.csv
predictions_v3/a6_bilstm_additive_attention_temporal_attention.csv
reports_v3/a6_bilstm_additive_attention_aggregation_ablation.csv
reports_v3/a6_bilstm_additive_attention_attention_summary.csv
reports_v3/a6_bilstm_additive_attention_summary.json
```

Attention values are saved only for interpretation: they show which *frame
positions* the trained head emphasized. They are not collision-time labels or
proof of causal explanation.

## Acceptance gate

For development shortlisting, compare the frozen `top3_mean` result against
the A2-MP full-MP4 reference (F1 0.738, Recall 0.867, PR-AUC 0.723) and D1
(F1 0.739, Recall 0.850, PR-AUC 0.741):

- primary: accident F1 with Recall >= 0.85;
- secondary: PR-AUC, precision, calibration later, and complementary FP/FN
  errors;
- final claims require the already-defined five-fold OOF protocol, not this
  single development split.

If A6 does not pass, it remains a documented negative ablation; it is not
silently discarded. If it is complementary to A2-MP or D1, it may later be a
candidate for the E1 ensemble after out-of-fold validation.
