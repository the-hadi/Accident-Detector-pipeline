"""E1-D: fixed, non-learned diagnostic ensemble on stored development predictions.

This is deliberately not a final stacking experiment.  It creates no features,
opens no MP4s, and fits no ensemble parameters on the development validation
set.  Its only learnable-looking operation is a clearly marked diagnostic
threshold scan for the pre-registered equal-probability mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from v3_a2mp_hn1 import metrics_at_threshold, select_threshold


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    recall_floor: float = 0.85

    @property
    def prediction_root(self) -> Path:
        return self.data_root / "predictions_v3"

    @property
    def report_root(self) -> Path:
        return self.data_root / "reports_v3"

    @property
    def a2_predictions_path(self) -> Path:
        return self.data_root / "inference_v2" / "a2_multipos_validation_sliding_video_predictions.csv"

    @property
    def d1_predictions_path(self) -> Path:
        return self.prediction_root / "d1_validation_video_predictions.csv"

    @property
    def predictions_path(self) -> Path:
        return self.prediction_root / "e1_development_ensemble_validation_predictions.csv"

    @property
    def comparison_path(self) -> Path:
        return self.report_root / "e1_development_ensemble_comparison.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "e1_development_ensemble_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


def prepare_directories(config: Config) -> None:
    config.prediction_root.mkdir(parents=True, exist_ok=True)
    config.report_root.mkdir(parents=True, exist_ok=True)


def load_predictions(config: Config) -> pd.DataFrame:
    for path in [config.a2_predictions_path, config.d1_predictions_path, config.registry_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing E1-D prerequisite: {path}")
    a2 = pd.read_csv(config.a2_predictions_path).copy()
    d1 = pd.read_csv(config.d1_predictions_path).copy()
    a2 = a2.rename(columns={"label": "video_label", "video_probability": "a2_probability", "prediction": "a2_prediction", "selected_threshold": "a2_threshold"})
    d1 = d1.rename(columns={"video_probability": "d1_probability", "prediction": "d1_prediction", "threshold": "d1_threshold"})
    a2 = a2[["video_id", "video_label", "a2_probability", "a2_prediction", "a2_threshold"]].copy()
    d1 = d1[["video_id", "video_label", "d1_probability", "d1_prediction", "d1_threshold"]].copy()
    for frame in [a2, d1]:
        frame["video_id"] = frame.video_id.astype(str)
        frame["video_label"] = frame.video_label.astype(int)
        if frame.video_id.duplicated().any() or len(frame) != 120:
            raise RuntimeError("E1-D requires exactly one full-MP4 prediction for each of 120 validation videos")
    joined = a2.merge(d1, on="video_id", suffixes=("", "_d1"), validate="one_to_one")
    if len(joined) != 120 or not joined.video_label.eq(joined.video_label_d1).all():
        raise RuntimeError("A2-MP and D1 validation prediction tables are incompatible")
    joined = joined.drop(columns="video_label_d1").sort_values("video_id").reset_index(drop=True)
    if joined.video_label.value_counts().to_dict() != {0: 60, 1: 60}:
        raise RuntimeError("E1-D expects a balanced fixed validation set")
    for column in ["a2_threshold", "d1_threshold"]:
        if joined[column].nunique() != 1:
            raise RuntimeError(f"E1-D expects one frozen threshold in {column}")
    return joined


def decision_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def run_diagnostic(config: Config = Config()) -> dict:
    prepare_directories(config)
    predictions = load_predictions(config)
    labels = predictions.video_label.to_numpy(dtype=int)
    a2_threshold = float(predictions.a2_threshold.iloc[0])
    d1_threshold = float(predictions.d1_threshold.iloc[0])

    # Fixed non-learned combination. The threshold scan below is an explicit
    # development-only diagnostic and cannot become the final threshold.
    predictions["equal_mean_probability"] = (predictions.a2_probability + predictions.d1_probability) / 2.0
    equal_metrics = select_threshold(labels, predictions.equal_mean_probability.to_numpy(dtype=float), config.recall_floor)
    predictions["equal_mean_selected_threshold"] = equal_metrics["threshold"]
    predictions["equal_mean_prediction"] = (predictions.equal_mean_probability >= equal_metrics["threshold"]).astype(int)
    predictions["and_prediction"] = ((predictions.a2_probability >= a2_threshold) & (predictions.d1_probability >= d1_threshold)).astype(int)
    predictions["or_prediction"] = ((predictions.a2_probability >= a2_threshold) | (predictions.d1_probability >= d1_threshold)).astype(int)
    predictions.to_csv(config.predictions_path, index=False)

    rows = []
    a2_metrics = metrics_at_threshold(labels, predictions.a2_probability.to_numpy(dtype=float), a2_threshold)
    d1_metrics = metrics_at_threshold(labels, predictions.d1_probability.to_numpy(dtype=float), d1_threshold)
    rows.append({"method": "A2-MP frozen reference", "method_type": "base_model", "threshold_rule": f"frozen {a2_threshold:.2f}", **a2_metrics, "notes": "Existing full-MP4 A2-MP output."})
    rows.append({"method": "D1 frozen RGB+motion", "method_type": "base_model", "threshold_rule": f"frozen {d1_threshold:.2f}", **d1_metrics, "notes": "Existing full-MP4 D1 output."})
    rows.append({"method": "E1-D equal raw probability mean", "method_type": "non_learned_diagnostic", "threshold_rule": f"development-selected {equal_metrics['threshold']:.2f}", **equal_metrics, "notes": "Equal weights fixed; raw probabilities are not OOF-calibrated."})
    rows.append({"method": "E1-D AND of frozen base decisions", "method_type": "non_learned_diagnostic", "threshold_rule": f"A2>={a2_threshold:.2f} AND D1>={d1_threshold:.2f}", **decision_metrics(labels, predictions.and_prediction.to_numpy(dtype=int)), "notes": "Binary decision rule; AUC values are not applicable."})
    rows.append({"method": "E1-D OR of frozen base decisions", "method_type": "non_learned_diagnostic", "threshold_rule": f"A2>={a2_threshold:.2f} OR D1>={d1_threshold:.2f}", **decision_metrics(labels, predictions.or_prediction.to_numpy(dtype=int)), "notes": "Binary decision rule; AUC values are not applicable."})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.comparison_path, index=False)

    disagreements = {
        "base_decisions_disagree": int((predictions.a2_prediction != predictions.d1_prediction).sum()),
        "a2_false_positives_resolved_by_d1": int(((predictions.video_label == 0) & (predictions.a2_prediction == 1) & (predictions.d1_prediction == 0)).sum()),
        "a2_false_negatives_resolved_by_d1": int(((predictions.video_label == 1) & (predictions.a2_prediction == 0) & (predictions.d1_prediction == 1)).sum()),
        "d1_new_false_positives_vs_a2": int(((predictions.video_label == 0) & (predictions.a2_prediction == 0) & (predictions.d1_prediction == 1)).sum()),
        "d1_new_false_negatives_vs_a2": int(((predictions.video_label == 1) & (predictions.a2_prediction == 1) & (predictions.d1_prediction == 0)).sum()),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_development_diagnostic_only",
        "scope": "fixed_120_video_development_validation",
        "base_models": ["A2-MP", "D1"],
        "default_ensemble_is_video_only": True,
        "fitted_stacker": False,
        "fitted_calibration": False,
        "final_threshold_selected": False,
        "equal_mean_development_metrics": equal_metrics,
        "disagreements": disagreements,
        "required_before_final_claim": "Run fold-local five-fold OOF base predictions, calibration, and stacking as specified in e1_ensemble_pipeline.md.",
        "prediction_path": str(config.predictions_path),
        "comparison_path": str(config.comparison_path),
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = pd.read_csv(config.registry_path)
    row = {
        "run_id": "V3_09_E1_DEVELOPMENT_DIAGNOSTIC",
        "stage": "V3-9 E1_development_diagnostic",
        "model_id": "E1-D fixed A2-MP+D1 non-learned rules",
        "dataset_version": "existing_full_mp4_validation_predictions",
        "split_version": "fixed_120_development_validation",
        "window_version": "base_models_frozen_top3_mean",
        "feature_version": "not_applicable",
        "augmentation_version": "none",
        "checkpoint_path": "base model predictions only",
        "config_path": "notebooks/39_v3_e1_development_ensemble.ipynb",
        "git_commit": "not_available",
        "status": "completed_development_diagnostic_only",
        "primary_metric": "diagnostic_equal_mean_f1_not_final",
        "primary_value": equal_metrics["f1"],
        "notes": "No learned weights/calibration. Final E1 requires fold-local five-fold OOF evaluation.",
    }
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"predictions": predictions, "comparison": comparison, "summary": summary}


def context_report(config: Config = Config()) -> dict:
    return {
        "a2_full_mp4_predictions": str(config.a2_predictions_path),
        "d1_full_mp4_predictions": str(config.d1_predictions_path),
        "feature_extraction_required": False,
        "mp4_decoding_required": False,
        "learned_stacker": False,
        "final_oof_evaluation": False,
    }
