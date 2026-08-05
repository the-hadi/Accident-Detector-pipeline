"""Fold-safe final OOF validation for the E1 A2-MP + D1 AND ensemble.

No video is decoded and no base model is trained here.  It reuses the frozen
five-fold A2-MP/D1 checkpoints and feature caches.  In every fold, thresholds
are selected only on the 48 inner-validation videos and then applied once to
the 120 untouched outer-validation videos.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from v3_a2mp_hn1 import aggregate_videos
from v3_final_oof_cv import Config, build_context, load_trained_head, score_rows, split_fold_videos


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
RECALL_FLOOR = 0.85


def decisions_and(a2_probability: np.ndarray, d1_probability: np.ndarray, a2_threshold: float, d1_threshold: float) -> np.ndarray:
    return ((a2_probability >= a2_threshold) & (d1_probability >= d1_threshold)).astype(int)


def decisions_single(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (probability >= threshold).astype(int)


def decision_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def choose_single_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = []
    for threshold in THRESHOLDS:
        metrics = decision_metrics(labels, decisions_single(probabilities, float(threshold)))
        candidates.append({"threshold": float(threshold), **metrics})
    # Accuracy is the user-requested primary objective; recall/F1 are only
    # deterministic tie-breakers, never tuned with outer-validation data.
    return max(candidates, key=lambda row: (row["accuracy"], row["recall"], row["f1"], -row["threshold"]))


def choose_and(labels: np.ndarray, a2_probability: np.ndarray, d1_probability: np.ndarray, *, safety_constrained: bool) -> dict:
    candidates: list[dict] = []
    for a2_threshold in THRESHOLDS:
        a2_mask = a2_probability >= a2_threshold
        for d1_threshold in THRESHOLDS:
            metrics = decision_metrics(labels, (a2_mask & (d1_probability >= d1_threshold)).astype(int))
            candidates.append({"a2_threshold": float(a2_threshold), "d1_threshold": float(d1_threshold), **metrics})
    pool = [row for row in candidates if row["recall"] >= RECALL_FLOOR] if safety_constrained else candidates
    if not pool:
        # Kept explicit in the output rather than silently discarding the
        # safety requirement when a fold cannot meet it.
        pool = candidates
        fallback = True
    else:
        fallback = False
    selected = max(pool, key=lambda row: (row["accuracy"], row["recall"], row["f1"], -row["a2_threshold"], -row["d1_threshold"]))
    return {"safety_fallback": fallback, **selected}


def score_inner_videos(context: dict, fold: int) -> pd.DataFrame:
    _, inner_ids, _ = split_fold_videos(context, fold)
    rows = context["union"].loc[context["union"].video_id.isin(inner_ids)].copy()
    outputs = []
    for kind in ("a2mp", "d1"):
        model = load_trained_head(context, kind, fold, context["config"].fold_model_path(kind, fold))
        windows = score_rows(model, kind, rows, context)
        videos = aggregate_videos(windows, context["config"].primary_aggregation)
        videos = videos[["video_id", "video_label", "video_probability"]].rename(columns={"video_probability": f"{kind}_probability"})
        outputs.append(videos)
    merged = outputs[0].merge(outputs[1], on=["video_id", "video_label"], validate="one_to_one")
    if len(merged) != 48 or merged.video_label.value_counts().to_dict() != {0: 24, 1: 24}:
        raise RuntimeError(f"Fold {fold} inner predictions are invalid")
    return merged.sort_values("video_id").reset_index(drop=True)


def load_outer_oof(config: Config) -> pd.DataFrame:
    table = pd.read_csv(config.oof_videos_path)
    table["video_id"] = table.video_id.astype(str)
    pivot = table.pivot(index=["video_id", "video_label", "outer_fold"], columns="model", values="video_probability").reset_index()
    if len(pivot) != 600 or pivot.video_label.value_counts().to_dict() != {0: 300, 1: 300} or set(pivot.outer_fold) != {0, 1, 2, 3, 4}:
        raise RuntimeError("Saved base OOF table is not the required 600-video five-fold table")
    return pivot.sort_values(["outer_fold", "video_id"]).reset_index(drop=True)


def exact_mcnemar(first: np.ndarray, second: np.ndarray, labels: np.ndarray) -> dict:
    first_correct = first == labels
    second_correct = second == labels
    first_correct_second_wrong = int(np.sum(first_correct & ~second_correct))
    first_wrong_second_correct = int(np.sum(~first_correct & second_correct))
    n = first_correct_second_wrong + first_wrong_second_correct
    if n == 0:
        p_value = 1.0
    else:
        lower_tail = sum(comb(n, value) for value in range(0, min(first_correct_second_wrong, first_wrong_second_correct) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "first_correct_second_wrong": first_correct_second_wrong,
        "first_wrong_second_correct": first_wrong_second_correct,
        "discordant": n,
        "exact_two_sided_p": p_value,
    }


def bootstrap(rows: pd.DataFrame, prediction_columns: list[str], seed: int = 42, resamples: int = 2000) -> pd.DataFrame:
    labels = rows.video_label.to_numpy(dtype=int)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values = {column: {metric: [] for metric in ("accuracy", "recall", "f1")} for column in prediction_columns}
    for _ in range(resamples):
        indices = np.concatenate([rng.choice(negative, size=len(negative), replace=True), rng.choice(positive, size=len(positive), replace=True)])
        sampled_labels = labels[indices]
        for column in prediction_columns:
            metric = decision_metrics(sampled_labels, rows[column].to_numpy(dtype=int)[indices])
            for name in values[column]:
                values[column][name].append(metric[name])
    records = []
    for column in prediction_columns:
        point = decision_metrics(labels, rows[column].to_numpy(dtype=int))
        for metric, samples in values[column].items():
            records.append({"model": column, "metric": metric, "estimate": point[metric], "ci95_low": float(np.percentile(samples, 2.5)), "ci95_high": float(np.percentile(samples, 97.5)), "resamples": resamples})
    return pd.DataFrame(records)


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_report(config: Config, comparison: pd.DataFrame, fold_selection: pd.DataFrame, paired: dict, selected: str) -> Path:
    rows = []
    for _, item in comparison.iterrows():
        rows.append(f"| {item.model} | {format_percent(item.accuracy)} | {format_percent(item.precision)} | {format_percent(item.recall)} | {item.f1:.3f} | {item.confusion_matrix} | {item.notes} |")
    fold_lines = []
    for _, item in fold_selection.iterrows():
        fold_lines.append(f"| {int(item.outer_fold)} | {item.d1_accuracy_threshold:.2f} | {item.e1_accuracy_a2_threshold:.2f} | {item.e1_accuracy_d1_threshold:.2f} | {item.e1_safe_a2_threshold:.2f} | {item.e1_safe_d1_threshold:.2f} | {bool(item.e1_safe_fallback)} |")
    report = f"""# E1-AND — final fold-safe OOF validation

## Protocol

- Five existing outer folds are reused; each outer-validation video was scored
  only by A2-MP and D1 heads trained without that video.
- In every fold, thresholds are selected only from that fold's 48
  inner-validation videos. No outer-validation label is used for threshold
  selection.
- The requested primary metric is Accuracy. A second safety-constrained E1
  variant is also retained to show the cost of enforcing Recall ≥ {RECALL_FLOOR:.2f}.
- No MP4 was decoded and no base head was retrained in this validation run.

## Result

The Accuracy-first winner under this fold-safe protocol is **{selected}**.
Its comparison with the nested Accuracy-threshold D1 is evaluated over all 600
out-of-fold videos. McNemar test: `{json.dumps(paired, ensure_ascii=False)}`.

| Model / decision policy | Accuracy | Precision | Recall | F1 | Confusion matrix [[TN,FP],[FN,TP]] | Notes |
|---|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

## Fold-local thresholds

| Outer fold | D1 accuracy threshold | E1-AND accuracy A2 | E1-AND accuracy D1 | E1-AND safety A2 | E1-AND safety D1 | Safety fallback |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fold_lines)}

## Decision gate

- An E1 result may be called *OOF-validated* because its thresholds are chosen
  without the labels of its outer fold.
- It can replace D1 for an Accuracy-only objective only when its OOF Accuracy
  exceeds nested D1 Accuracy. It remains inappropriate for the original safety
  objective when its OOF Recall is below {RECALL_FLOOR:.2f}.
- A deployable final E1 requires a separate all-data A2-MP head, reuse of the
  final all-data D1 head, and a shared inference wrapper. That hand-off should
  be created only if the selected OOF policy passes the intended objective.
"""
    path = config.report_root / "e1_final_oof_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_chart(config: Config, comparison: pd.DataFrame) -> Path:
    chart = comparison.loc[comparison.model.ne("D1 current deployed policy")].copy().sort_values("accuracy")
    colors = ["#1b9e77" if row.model.startswith("E1-AND accuracy") else "#d95f02" if "safety" in row.model else "#7570b3" for _, row in chart.iterrows()]
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    bars = axis.barh(chart.model, chart.accuracy * 100, color=colors)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("E1-AND final fold-safe OOF comparison", weight="bold")
    axis.grid(axis="x", alpha=0.25)
    for bar, (_, row) in zip(bars, chart.iterrows()):
        axis.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{row.accuracy * 100:.1f}% | Recall {row.recall * 100:.1f}%", va="center")
    path = config.report_root / "e1_final_oof_accuracy_comparison.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run() -> dict:
    config = Config()
    context = build_context(config)
    outer = load_outer_oof(config)
    selection_records, merged_inners = [], []
    for fold in range(5):
        inner = score_inner_videos(context, fold)
        merged_inners.append(inner.assign(outer_fold=fold))
        labels = inner.video_label.to_numpy(dtype=int)
        d1_choice = choose_single_accuracy(labels, inner.d1_probability.to_numpy(dtype=float))
        e1_accuracy_choice = choose_and(labels, inner.a2mp_probability.to_numpy(dtype=float), inner.d1_probability.to_numpy(dtype=float), safety_constrained=False)
        e1_safe_choice = choose_and(labels, inner.a2mp_probability.to_numpy(dtype=float), inner.d1_probability.to_numpy(dtype=float), safety_constrained=True)
        selection_records.append({
            "outer_fold": fold,
            "inner_videos": len(inner),
            "d1_accuracy_threshold": d1_choice["threshold"],
            "d1_inner_accuracy": d1_choice["accuracy"],
            "e1_accuracy_a2_threshold": e1_accuracy_choice["a2_threshold"],
            "e1_accuracy_d1_threshold": e1_accuracy_choice["d1_threshold"],
            "e1_accuracy_inner_accuracy": e1_accuracy_choice["accuracy"],
            "e1_accuracy_inner_recall": e1_accuracy_choice["recall"],
            "e1_safe_a2_threshold": e1_safe_choice["a2_threshold"],
            "e1_safe_d1_threshold": e1_safe_choice["d1_threshold"],
            "e1_safe_inner_accuracy": e1_safe_choice["accuracy"],
            "e1_safe_inner_recall": e1_safe_choice["recall"],
            "e1_safe_fallback": e1_safe_choice["safety_fallback"],
        })
    selections = pd.DataFrame(selection_records).sort_values("outer_fold").reset_index(drop=True)
    selections.to_csv(config.report_root / "e1_final_oof_fold_thresholds.csv", index=False)
    pd.concat(merged_inners, ignore_index=True).to_csv(config.prediction_root / "e1_final_oof_inner_predictions.csv", index=False)

    outer = outer.merge(selections, on="outer_fold", validate="many_to_one")
    outer["d1_nested_accuracy_prediction"] = [
        int(probability >= threshold) for probability, threshold in zip(outer.d1, outer.d1_accuracy_threshold)
    ]
    outer["e1_and_accuracy_prediction"] = decisions_and(
        outer.a2mp.to_numpy(dtype=float), outer.d1.to_numpy(dtype=float), outer.e1_accuracy_a2_threshold.to_numpy(dtype=float), outer.e1_accuracy_d1_threshold.to_numpy(dtype=float)
    )
    outer["e1_and_safety_prediction"] = decisions_and(
        outer.a2mp.to_numpy(dtype=float), outer.d1.to_numpy(dtype=float), outer.e1_safe_a2_threshold.to_numpy(dtype=float), outer.e1_safe_d1_threshold.to_numpy(dtype=float)
    )

    final_summary = json.loads(config.summary_path.read_text(encoding="utf-8"))
    deployed_threshold = float(final_summary["selected_thresholds"]["d1"])
    outer["d1_current_deployed_prediction"] = decisions_single(outer.d1.to_numpy(dtype=float), deployed_threshold)
    outer.to_csv(config.prediction_root / "e1_final_oof_ensemble_predictions.csv", index=False)

    labels = outer.video_label.to_numpy(dtype=int)
    models = [
        ("D1 current deployed policy", "d1_current_deployed_prediction", f"global OOF threshold {deployed_threshold:.2f}; original F1/Recall policy"),
        ("D1 nested Accuracy policy", "d1_nested_accuracy_prediction", "threshold selected from each fold's inner-validation Accuracy"),
        ("E1-AND accuracy-first", "e1_and_accuracy_prediction", "two AND thresholds selected from each fold's inner-validation Accuracy"),
        ("E1-AND safety-constrained", "e1_and_safety_prediction", f"inner threshold selection maximizes Accuracy subject to Recall >= {RECALL_FLOOR:.2f}"),
    ]
    comparison_rows = []
    for model, column, notes in models:
        comparison_rows.append({"model": model, **decision_metrics(labels, outer[column].to_numpy(dtype=int)), "notes": notes})
    comparison = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    comparison.to_csv(config.report_root / "e1_final_oof_accuracy_comparison.csv", index=False)

    paired = {
        "comparison": "E1-AND accuracy-first versus D1 nested Accuracy policy",
        **exact_mcnemar(outer.d1_nested_accuracy_prediction.to_numpy(dtype=int), outer.e1_and_accuracy_prediction.to_numpy(dtype=int), labels),
    }
    (config.report_root / "e1_final_oof_paired_tests.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    bootstrap_table = bootstrap(outer, [column for _, column, _ in models])
    bootstrap_table.to_csv(config.report_root / "e1_final_oof_bootstrap_statistics.csv", index=False)

    d1_nested_accuracy = float(comparison.loc[comparison.model.eq("D1 nested Accuracy policy"), "accuracy"].iloc[0])
    e1_accuracy = float(comparison.loc[comparison.model.eq("E1-AND accuracy-first"), "accuracy"].iloc[0])
    e1_recall = float(comparison.loc[comparison.model.eq("E1-AND accuracy-first"), "recall"].iloc[0])
    if e1_accuracy > d1_nested_accuracy:
        selected = "E1-AND accuracy-first"
    else:
        selected = "D1 nested Accuracy policy"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "scope": "five_fold_nested_threshold_selection_outer_oof_full_mp4",
        "primary_metric": "accuracy",
        "recall_floor_for_safety_variant": RECALL_FLOOR,
        "selected_accuracy_first": selected,
        "e1_accuracy_first_recall": e1_recall,
        "base_checkpoints": "existing final_oof_{a2mp,d1}_fold_*.pt; no retraining",
        "feature_cache": "existing final_oof_rgb_features.pt + final_oof_motion_features.pt; no decoding",
        "comparison": comparison.to_dict(orient="records"),
        "paired_tests": paired,
        "paths": {
            "predictions": str(config.prediction_root / "e1_final_oof_ensemble_predictions.csv"),
            "fold_thresholds": str(config.report_root / "e1_final_oof_fold_thresholds.csv"),
            "bootstrap": str(config.report_root / "e1_final_oof_bootstrap_statistics.csv"),
        },
        "gate": "Accuracy-first E1 can be considered only under an Accuracy-only objective. If Recall >= 0.85 remains required, use the safety variant result and retain D1 unless it wins there.",
    }
    (config.report_root / "e1_final_oof_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    chart = write_chart(config, comparison)
    report = write_report(config, comparison, selections, paired, selected)
    return {"selected_accuracy_first": selected, "comparison": comparison.to_dict(orient="records"), "chart": str(chart), "report": str(report)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
