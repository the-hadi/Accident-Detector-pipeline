"""Fast, fold-safe D1 probability-curve verifier using existing V3 caches."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from v3_final_oof_cv import Config, build_context, load_trained_head, score_rows, split_fold_videos


FEATURE_COLUMNS = [
    "curve_max",
    "curve_top3_mean",
    "curve_mean",
    "curve_std",
    "curve_peak_prominence",
    "curve_peak_neighbor_mean",
    "curve_fraction_ge_050",
    "curve_mean_abs_adjacent_change",
]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def decision_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def select_accuracy_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = []
    for threshold in THRESHOLDS:
        metrics = decision_metrics(labels, (probabilities >= threshold).astype(int))
        candidates.append({"threshold": float(threshold), **metrics})
    return max(candidates, key=lambda row: (row["accuracy"], row["recall"], row["f1"], -row["threshold"]))


def curve_feature_table(windows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for video_id, group in windows.groupby("video_id", sort=True):
        ordered = group.sort_values("window_index")
        probability = ordered.positive_probability.to_numpy(dtype=float)
        peak = int(np.argmax(probability))
        neighbours = [index for index in (peak - 1, peak + 1) if 0 <= index < len(probability)]
        neighbour_mean = float(probability[neighbours].mean()) if neighbours else float(probability[peak])
        records.append(
            {
                "video_id": str(video_id),
                "video_label": int(ordered.video_label.iloc[0]),
                "window_count": len(probability),
                "curve_max": float(probability.max()),
                "curve_top3_mean": float(np.sort(probability)[-min(3, len(probability)):].mean()),
                "curve_mean": float(probability.mean()),
                "curve_std": float(probability.std()),
                "curve_peak_prominence": float(probability.max() - probability.mean()),
                "curve_peak_neighbor_mean": neighbour_mean,
                "curve_fraction_ge_050": float(np.mean(probability >= 0.50)),
                "curve_mean_abs_adjacent_change": float(np.abs(np.diff(probability)).mean()) if len(probability) > 1 else 0.0,
            }
        )
    return pd.DataFrame(records)


def score_inner_windows(context: dict, fold: int) -> pd.DataFrame:
    _, inner_ids, _ = split_fold_videos(context, fold)
    inner_rows = context["union"].loc[context["union"].video_id.isin(inner_ids)].copy()
    model = load_trained_head(context, "d1", fold, context["config"].fold_model_path("d1", fold))
    windows = score_rows(model, "d1", inner_rows, context)
    windows["selection_outer_fold"] = fold
    inner_videos = windows[["video_id", "video_label"]].drop_duplicates("video_id")
    if len(inner_videos) != 48 or inner_videos.video_label.value_counts().to_dict() != {0: 24, 1: 24}:
        raise RuntimeError(f"Invalid inner D1 window predictions for fold {fold}")
    return windows


def exact_mcnemar(first: np.ndarray, second: np.ndarray, labels: np.ndarray) -> dict:
    first_correct = first == labels
    second_correct = second == labels
    first_correct_second_wrong = int(np.sum(first_correct & ~second_correct))
    first_wrong_second_correct = int(np.sum(~first_correct & second_correct))
    n = first_correct_second_wrong + first_wrong_second_correct
    lower = sum(comb(n, value) for value in range(min(first_correct_second_wrong, first_wrong_second_correct) + 1)) / (2**n) if n else 0.5
    return {
        "first_correct_second_wrong": first_correct_second_wrong,
        "first_wrong_second_correct": first_wrong_second_correct,
        "discordant": n,
        "exact_two_sided_p": 1.0 if n == 0 else min(1.0, 2.0 * lower),
    }


def bootstrap(table: pd.DataFrame, columns: list[str], seed: int = 42, resamples: int = 2000) -> pd.DataFrame:
    labels = table.video_label.to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    values = {column: {metric: [] for metric in ("accuracy", "recall", "f1")} for column in columns}
    for _ in range(resamples):
        indices = np.concatenate([rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)])
        for column in columns:
            metrics = decision_metrics(labels[indices], table[column].to_numpy(dtype=int)[indices])
            for metric in values[column]:
                values[column][metric].append(metrics[metric])
    rows = []
    for column in columns:
        point = decision_metrics(labels, table[column].to_numpy(dtype=int))
        for metric, samples in values[column].items():
            rows.append({"model": column, "metric": metric, "estimate": point[metric], "ci95_low": float(np.percentile(samples, 2.5)), "ci95_high": float(np.percentile(samples, 97.5)), "resamples": resamples})
    return pd.DataFrame(rows)


def write_outputs(config: Config, comparison: pd.DataFrame, paired: dict, selected: str) -> dict:
    report_root = config.report_root
    comparison.to_csv(report_root / "d1_curve_verifier_comparison.csv", index=False)
    (report_root / "d1_curve_verifier_paired_tests.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_rows = comparison.sort_values("accuracy")
    figure, axis = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    colors = ["#1b9e77" if item.model == "D1 curve verifier" else "#7570b3" for _, item in chart_rows.iterrows()]
    bars = axis.barh(chart_rows.model, chart_rows.accuracy * 100, color=colors)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("D1 probability-curve verifier — fold-safe OOF", weight="bold")
    axis.grid(axis="x", alpha=0.25)
    for bar, (_, item) in zip(bars, chart_rows.iterrows()):
        axis.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{item.accuracy * 100:.1f}% | Recall {item.recall * 100:.1f}%", va="center")
    chart_path = report_root / "d1_curve_verifier_chart.png"
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    table_lines = [
        "| Model | Accuracy | Precision | Recall | F1 | Confusion matrix [[TN,FP],[FN,TP]] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison.iterrows():
        table_lines.append(f"| {row.model} | {row.accuracy:.2%} | {row.precision:.2%} | {row.recall:.2%} | {row.f1:.3f} | {row.confusion_matrix} |")
    report = f"""# D1 probability-curve verifier — final OOF result

This is a lightweight, no-new-video experiment. It uses only D1's ordered
window probabilities. For every outer fold, its logistic verifier and decision
threshold were fitted/selected only on the 48 inner-validation videos; the
outer-validation fold remained untouched.

## Result

Accuracy-first selected policy: **{selected}**.

{chr(10).join(table_lines)}

Paired exact McNemar (verifier vs D1 nested Accuracy policy):
`{json.dumps(paired, ensure_ascii=False)}`.

The verifier can replace D1 only if its OOF Accuracy gain is stable and its
Recall loss is acceptable. It is not automatically handed off as an arbitrary
MP4 model merely because one 600-video experiment has a higher point estimate.
"""
    report_path = report_root / "d1_curve_verifier_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "chart": str(chart_path)}


def run() -> dict:
    config = Config()
    context = build_context(config)
    base_windows = pd.read_csv(config.oof_windows_path)
    base_windows["video_id"] = base_windows.video_id.astype(str)
    outer_d1_windows = base_windows.loc[base_windows.model.eq("d1")].copy()
    if outer_d1_windows.video_id.nunique() != 600:
        raise RuntimeError("The saved D1 OOF window table is incomplete")

    inner_windows = []
    fold_records = []
    outer_features_by_fold = []
    for fold in range(5):
        print(f"D1 curve verifier: scoring inner fold {fold}/4", flush=True)
        inner = score_inner_windows(context, fold)
        inner_windows.append(inner)
        inner_features = curve_feature_table(inner)
        outer_features = curve_feature_table(outer_d1_windows.loc[outer_d1_windows.outer_fold.eq(fold)].copy())
        if len(outer_features) != 120 or outer_features.video_label.value_counts().to_dict() != {0: 60, 1: 60}:
            raise RuntimeError(f"Invalid outer feature table for fold {fold}")

        verifier = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.1, solver="liblinear", max_iter=1000, random_state=42 + fold)),
        ])
        verifier.fit(inner_features[FEATURE_COLUMNS], inner_features.video_label)
        inner_probability = verifier.predict_proba(inner_features[FEATURE_COLUMNS])[:, 1]
        choice = select_accuracy_threshold(inner_features.video_label.to_numpy(dtype=int), inner_probability)
        outer_probability = verifier.predict_proba(outer_features[FEATURE_COLUMNS])[:, 1]
        outer_features["outer_fold"] = fold
        outer_features["verifier_probability"] = outer_probability
        outer_features["verifier_threshold"] = choice["threshold"]
        outer_features["d1_curve_verifier_prediction"] = (outer_probability >= choice["threshold"]).astype(int)
        outer_features_by_fold.append(outer_features)
        fold_records.append({
            "outer_fold": fold,
            "inner_videos": len(inner_features),
            "threshold": choice["threshold"],
            "inner_accuracy": choice["accuracy"],
            "inner_recall": choice["recall"],
            "inner_f1": choice["f1"],
        })

    pd.concat(inner_windows, ignore_index=True).to_csv(config.prediction_root / "d1_curve_verifier_inner_window_predictions.csv", index=False)
    folds = pd.DataFrame(fold_records).sort_values("outer_fold").reset_index(drop=True)
    folds.to_csv(config.report_root / "d1_curve_verifier_fold_thresholds.csv", index=False)
    verifier_oof = pd.concat(outer_features_by_fold, ignore_index=True)

    baseline = pd.read_csv(config.prediction_root / "e1_final_oof_ensemble_predictions.csv")
    baseline["video_id"] = baseline.video_id.astype(str)
    result = verifier_oof.merge(
        baseline[["video_id", "video_label", "outer_fold", "d1_nested_accuracy_prediction", "d1_current_deployed_prediction"]],
        on=["video_id", "video_label", "outer_fold"],
        validate="one_to_one",
    ).sort_values(["outer_fold", "video_id"]).reset_index(drop=True)
    if len(result) != 600 or result.video_label.value_counts().to_dict() != {0: 300, 1: 300}:
        raise RuntimeError("Verifier OOF merge is invalid")
    result.to_csv(config.prediction_root / "d1_curve_verifier_oof_predictions.csv", index=False)

    labels = result.video_label.to_numpy(dtype=int)
    models = [
        ("D1 curve verifier", "d1_curve_verifier_prediction"),
        ("D1 nested Accuracy policy", "d1_nested_accuracy_prediction"),
        ("D1 current deployed policy", "d1_current_deployed_prediction"),
    ]
    comparison = pd.DataFrame([
        {"model": name, **decision_metrics(labels, result[column].to_numpy(dtype=int))}
        for name, column in models
    ]).sort_values("accuracy", ascending=False).reset_index(drop=True)
    paired = {
        "comparison": "D1 curve verifier versus D1 nested Accuracy policy",
        **exact_mcnemar(result.d1_nested_accuracy_prediction.to_numpy(dtype=int), result.d1_curve_verifier_prediction.to_numpy(dtype=int), labels),
    }
    bootstrap_table = bootstrap(result, [column for _, column in models])
    bootstrap_table.to_csv(config.report_root / "d1_curve_verifier_bootstrap.csv", index=False)

    verifier_accuracy = float(comparison.loc[comparison.model.eq("D1 curve verifier"), "accuracy"].iloc[0])
    d1_nested_accuracy = float(comparison.loc[comparison.model.eq("D1 nested Accuracy policy"), "accuracy"].iloc[0])
    verifier_recall = float(comparison.loc[comparison.model.eq("D1 curve verifier"), "recall"].iloc[0])
    selected = "D1 curve verifier" if verifier_accuracy > d1_nested_accuracy and verifier_recall >= 0.85 else "D1 current deployed policy"
    output_paths = write_outputs(config, comparison, paired, selected)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "scope": "five_fold_outer_oof_with_48_video_fold_local_verifier_fit",
        "primary_metric": "accuracy",
        "selected_policy": selected,
        "verifier": "StandardScaler + LogisticRegression(C=0.1, L2) over eight D1 probability-curve features",
        "no_new_videos": True,
        "no_mp4_decode": True,
        "no_resnet_retraining": True,
        "comparison": comparison.to_dict(orient="records"),
        "paired_tests": paired,
        "paths": output_paths,
        "gate": "Promote only if Accuracy improves stably and Recall remains at least 0.85; otherwise retain deployed D1.",
    }
    (config.report_root / "d1_curve_verifier_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
