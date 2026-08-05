"""Fold-local A2-MP/D1 OOF training, statistics, and model selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score
from sklearn.model_selection import train_test_split

from v3_a2mp_hn1 import MeanMaxHead, aggregate_videos, metrics_at_threshold, round_robin_negative_selection, select_threshold
from v3_d1_motion import RGBMotionFusionHead
from v3_final_oof_cache import Config as CacheConfig, build_context as build_cache_context, sha256_file


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    feature_dim: int = 512
    batch_size: int = 64
    epochs: int = 30
    mining_epochs: int = 18
    early_stopping_patience: int = 6
    mining_early_stopping_patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    recall_floor: float = 0.85
    primary_aggregation: str = "top3_mean"
    inner_validation_fraction: float = 0.10
    hard_negative_target: int = 240
    hard_negative_score_threshold: float = 0.60
    hard_negative_max_per_video: int = 3
    hard_negative_multiplier: float = 1.5
    bootstrap_resamples: int = 2000

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests_v3"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed_v3"

    @property
    def prediction_root(self) -> Path:
        return self.data_root / "predictions_v3"

    @property
    def report_root(self) -> Path:
        return self.data_root / "reports_v3"

    @property
    def model_root(self) -> Path:
        return self.data_root / "models_v3"

    @property
    def union_path(self) -> Path:
        return self.manifest_root / "final_oof_union_sequences.csv"

    @property
    def rgb_features_path(self) -> Path:
        return self.processed_root / "final_oof_rgb_features.pt"

    @property
    def motion_features_path(self) -> Path:
        return self.processed_root / "final_oof_motion_features.pt"

    @property
    def oof_windows_path(self) -> Path:
        return self.prediction_root / "final_oof_base_windows.csv"

    @property
    def oof_videos_path(self) -> Path:
        return self.prediction_root / "final_oof_base_videos.csv"

    @property
    def calibrated_videos_path(self) -> Path:
        return self.prediction_root / "final_oof_calibrated_videos.csv"

    @property
    def comparison_path(self) -> Path:
        return self.report_root / "final_oof_model_comparison.csv"

    @property
    def bootstrap_path(self) -> Path:
        return self.report_root / "final_oof_bootstrap_statistics.csv"

    @property
    def paired_tests_path(self) -> Path:
        return self.report_root / "final_oof_paired_tests.json"

    @property
    def calibration_path(self) -> Path:
        return self.report_root / "final_oof_calibration_report.json"

    @property
    def selection_path(self) -> Path:
        return self.report_root / "final_model_selection.md"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "final_oof_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"

    def fold_train_path(self, fold: int) -> Path:
        return self.manifest_root / f"final_oof_fold_{fold}_train_windows.csv"

    def fold_inner_path(self, fold: int) -> Path:
        return self.manifest_root / f"final_oof_fold_{fold}_inner_validation_videos.csv"

    def fold_hard_path(self, fold: int) -> Path:
        return self.manifest_root / f"final_oof_fold_{fold}_hard_negatives.csv"

    def fold_model_path(self, kind: str, fold: int) -> Path:
        return self.model_root / f"final_oof_{kind}_fold_{fold}.pt"

    def fold_history_path(self, kind: str, fold: int) -> Path:
        return self.model_root / f"final_oof_{kind}_fold_{fold}_history.csv"


class CachedSequenceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, rgb: dict[str, torch.Tensor], motion: dict[str, torch.Tensor]) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.rgb = rgb
        self.motion = motion

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        sequence_id = str(row.sequence_id)
        return (
            self.rgb[sequence_id],
            self.motion[sequence_id],
            torch.tensor(float(row.hard_label), dtype=torch.float32),
            torch.tensor(float(row.loss_weight), dtype=torch.float32),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_directories(config: Config) -> None:
    for directory in [config.manifest_root, config.prediction_root, config.report_root, config.model_root]:
        directory.mkdir(parents=True, exist_ok=True)


def load_complete_features(path: Path, cache_context: dict, kind: str, config: Config) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Final OOF {kind} cache is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_signature") != cache_context["run_signature"] or payload.get("feature_kind") != kind:
        raise RuntimeError(f"Final OOF {kind} cache belongs to a different union manifest")
    sequence_ids = [str(item) for item in payload.get("sequence_ids", [])]
    features = payload.get("features")
    if len(sequence_ids) != len(features):
        raise RuntimeError(f"Final OOF {kind} cache IDs/features are inconsistent")
    by_sequence = {
        sequence_id: feature.float().cpu()
        for sequence_id, feature in zip(sequence_ids, features)
        if tuple(feature.shape) == (16, config.feature_dim)
    }
    required = cache_context["union"].sequence_id.tolist()
    missing = [item for item in required if item not in by_sequence]
    if missing:
        raise RuntimeError(f"Final OOF {kind} cache misses {len(missing)} sequences")
    return {item: by_sequence[item] for item in required}


def build_context(config: Config = Config()) -> dict:
    prepare_directories(config)
    cache_context = build_cache_context(CacheConfig())
    for path in [config.rgb_features_path, config.motion_features_path, config.registry_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Final OOF prerequisite is missing: {path}")
    union = pd.read_csv(config.union_path).copy()
    union["video_id"] = union.video_id.astype(str)
    union["sequence_id"] = union.sequence_id.astype(str)
    if len(union) != 8907 or union.video_id.nunique() != 600 or set(union.outer_fold.unique()) != {0, 1, 2, 3, 4}:
        raise RuntimeError("Final OOF union manifest is invalid")
    videos = union[["video_id", "video_label", "outer_fold"]].drop_duplicates("video_id").copy()
    if len(videos) != 600:
        raise RuntimeError("Final OOF video table is invalid")
    rgb = load_complete_features(config.rgb_features_path, cache_context, "rgb", config)
    motion = load_complete_features(config.motion_features_path, cache_context, "motion", config)
    signature = {
        "cache_run_signature": cache_context["run_signature"],
        "union_sha256": sha256_file(config.union_path),
        "rgb_sha256": sha256_file(config.rgb_features_path),
        "motion_sha256": sha256_file(config.motion_features_path),
        "config": {key: value for key, value in config.__dict__.items() if key not in {"data_root"}},
    }
    return {"config": config, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "union": union, "videos": videos, "rgb": rgb, "motion": motion, "cache_context": cache_context, "run_signature": json.dumps(signature, sort_keys=True, default=str)}


def build_head(kind: str, config: Config) -> nn.Module:
    if kind in {"a2mp", "hn_miner"}:
        return MeanMaxHead(config.feature_dim)
    if kind == "d1":
        return RGBMotionFusionHead(config.feature_dim)
    raise ValueError(f"Unknown head type: {kind}")


def forward_head(model: nn.Module, kind: str, rgb: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    return model(rgb) if kind in {"a2mp", "hn_miner"} else model(rgb, motion)


def score_rows(model: nn.Module, kind: str, rows: pd.DataFrame, context: dict, batch_size: int = 256) -> pd.DataFrame:
    sequence_ids = rows.sequence_id.astype(str).tolist()
    rgb = torch.stack([context["rgb"][item] for item in sequence_ids])
    motion = torch.stack([context["motion"][item] for item in sequence_ids])
    logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            logits.append(forward_head(model, kind, rgb[start:start + batch_size].to(context["device"]), motion[start:start + batch_size].to(context["device"])).cpu())
    result = rows.copy()
    result["window_logit"] = torch.cat(logits).numpy()
    result["positive_probability"] = torch.sigmoid(torch.cat(logits)).numpy()
    return result


def assign_loss_weights(positives: pd.DataFrame, normal_negatives: pd.DataFrame, hard_negatives: pd.DataFrame, config: Config) -> pd.DataFrame:
    positives, normal_negatives, hard_negatives = positives.copy(), normal_negatives.copy(), hard_negatives.copy()
    positives["training_role"] = "positive_core"
    normal_negatives["training_role"] = "normal_negative"
    hard_negatives["training_role"] = "hard_negative_fold_local"
    positives["hard_negative_multiplier"] = 1.0
    normal_negatives["hard_negative_multiplier"] = 1.0
    hard_negatives["hard_negative_multiplier"] = config.hard_negative_multiplier
    negative_total = float(len(normal_negatives) + config.hard_negative_multiplier * len(hard_negatives))
    positives["loss_weight"] = negative_total / len(positives)
    normal_negatives["loss_weight"] = 1.0
    hard_negatives["loss_weight"] = config.hard_negative_multiplier
    result = pd.concat([positives, normal_negatives, hard_negatives], ignore_index=True)
    if result.sequence_id.duplicated().any():
        raise RuntimeError("Fold training rows contain duplicate sequences")
    return result.sort_values(["training_role", "video_id", "window_index"]).reset_index(drop=True)


def split_fold_videos(context: dict, fold: int) -> tuple[set[str], set[str], set[str]]:
    config: Config = context["config"]
    videos = context["videos"]
    outer_validation = videos.loc[videos.outer_fold.eq(fold)].copy()
    outer_train = videos.loc[~videos.outer_fold.eq(fold)].copy()
    train_ids, inner_ids = train_test_split(
        outer_train.video_id.to_numpy(),
        test_size=config.inner_validation_fraction,
        random_state=config.seed + fold,
        stratify=outer_train.video_label.to_numpy(dtype=int),
    )
    if len(outer_validation) != 120 or len(train_ids) != 432 or len(inner_ids) != 48:
        raise RuntimeError("Final OOF fold split sizes are invalid")
    return set(train_ids), set(inner_ids), set(outer_validation.video_id)


def build_pre_mining_rows(context: dict, training_video_ids: set[str], fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config: Config = context["config"]
    union = context["union"]
    positives = union.loc[union.video_id.isin(training_video_ids) & union.video_label.eq(1) & union.window_role.eq("positive_core") & union.hard_label.eq(1)].copy()
    negative_pool = union.loc[union.video_id.isin(training_video_ids) & union.video_label.eq(0) & union.window_role.eq("negative_video") & union.hard_label.eq(0)].copy()
    if positives.empty or negative_pool.empty:
        raise RuntimeError("Fold training candidates are unexpectedly empty")
    normal = round_robin_negative_selection(negative_pool, len(positives), config.seed + fold).copy()
    empty_hard = negative_pool.head(0).copy()
    pre_mining = assign_loss_weights(positives, normal, empty_hard, config)
    return positives, negative_pool, pre_mining


def select_hard_negatives(scored_negative_pool: pd.DataFrame, config: Config) -> pd.DataFrame:
    ranked = scored_negative_pool.sort_values("positive_probability", ascending=False).copy()
    selected_indices: list[int] = []
    counts: dict[str, int] = {}
    for restrict_to_hard in [True, False]:
        for index, row in ranked.iterrows():
            if restrict_to_hard and float(row.positive_probability) < config.hard_negative_score_threshold:
                continue
            if index in selected_indices or counts.get(str(row.video_id), 0) >= config.hard_negative_max_per_video:
                continue
            selected_indices.append(index)
            counts[str(row.video_id)] = counts.get(str(row.video_id), 0) + 1
            if len(selected_indices) >= config.hard_negative_target:
                break
        if len(selected_indices) >= config.hard_negative_target:
            break
    selected = ranked.loc[selected_indices].copy()
    if len(selected) != config.hard_negative_target:
        raise RuntimeError("Unable to select the configured number of fold-local hard negatives")
    selected["hard_negative_score"] = selected.positive_probability
    selected["hard_negative_selection"] = np.where(selected.positive_probability >= config.hard_negative_score_threshold, "score_at_least_threshold", "fallback_ranked")
    return selected.sort_values(["video_id", "window_index"]).reset_index(drop=True)


def train_head(context: dict, kind: str, fold: int, train_rows: pd.DataFrame, inner_rows: pd.DataFrame, epochs: int, patience: int, model_path: Path, history_path: Path) -> dict:
    config: Config = context["config"]
    seed_everything(config.seed + fold + (0 if kind in {"a2mp", "hn_miner"} else 100))
    dataset = CachedSequenceDataset(train_rows, context["rgb"], context["motion"])
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed + fold + (0 if kind in {"a2mp", "hn_miner"} else 100)), num_workers=0)
    model = build_head(kind, config).to(context["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_pr_auc, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss, total_weight = 0.0, 0.0
        for rgb, motion, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_head(model, kind, rgb.to(context["device"]), motion.to(context["device"]))
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels.to(context["device"]), reduction="none")
            device_weights = weights.to(context["device"])
            loss = (raw_loss * device_weights).sum() / device_weights.sum()
            loss.backward()
            optimizer.step()
            weighted_loss += float((raw_loss.detach().cpu() * weights).sum())
            total_weight += float(weights.sum())
        inner_windows = score_rows(model, kind, inner_rows, context)
        inner_videos = aggregate_videos(inner_windows, config.primary_aggregation)
        pr_auc = float(average_precision_score(inner_videos.video_label, inner_videos.video_probability))
        record = {"fold": fold, "model": kind, "epoch": epoch, "weighted_train_loss": weighted_loss / total_weight, "inner_validation_pr_auc_top3_mean": pr_auc}
        history.append(record)
        print(record, flush=True)
        if pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale = pr_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "kind": kind, "fold": fold, "epoch": epoch, "inner_validation_pr_auc_top3_mean": pr_auc, "run_signature": context["run_signature"]}, model_path)
        else:
            stale += 1
        if stale >= patience:
            print(f"{kind} fold {fold}: early stopping at epoch {epoch}; best epoch {best_epoch}.", flush=True)
            break
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {"kind": kind, "fold": fold, "model_path": str(model_path), "best_epoch": best_epoch, "best_inner_pr_auc": best_pr_auc, "epochs_completed": len(history)}


def load_trained_head(context: dict, kind: str, fold: int, model_path: Path) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=context["device"], weights_only=False)
    if checkpoint.get("kind") != kind or int(checkpoint.get("fold")) != fold or checkpoint.get("run_signature") != context["run_signature"]:
        raise RuntimeError(f"Checkpoint {model_path} is incompatible with final OOF context")
    model = build_head(kind, context["config"]).to(context["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_fold(context: dict, fold: int) -> dict:
    config: Config = context["config"]
    training_ids, inner_ids, outer_validation_ids = split_fold_videos(context, fold)
    inner_rows = context["union"].loc[context["union"].video_id.isin(inner_ids)].copy()
    outer_validation_rows = context["union"].loc[context["union"].video_id.isin(outer_validation_ids)].copy()
    positives, negative_pool, pre_mining_rows = build_pre_mining_rows(context, training_ids, fold)
    miner_path = config.fold_model_path("hn_miner", fold)
    miner_history_path = config.fold_history_path("hn_miner", fold)
    miner = train_head(context, "hn_miner", fold, pre_mining_rows, inner_rows, config.mining_epochs, config.mining_early_stopping_patience, miner_path, miner_history_path)
    mining_model = load_trained_head(context, "hn_miner", fold, miner_path)
    scored_negatives = score_rows(mining_model, "hn_miner", negative_pool, context)
    hard_negatives = select_hard_negatives(scored_negatives, config)
    hard_negatives.to_csv(config.fold_hard_path(fold), index=False)
    normal_pool = negative_pool.loc[~negative_pool.sequence_id.isin(set(hard_negatives.sequence_id))].copy()
    normal_negatives = round_robin_negative_selection(normal_pool, len(positives), config.seed + fold).copy()
    final_train = assign_loss_weights(positives, normal_negatives, hard_negatives, config)
    final_train.to_csv(config.fold_train_path(fold), index=False)
    inner_videos = context["videos"].loc[context["videos"].video_id.isin(inner_ids)].sort_values("video_id")
    inner_videos.to_csv(config.fold_inner_path(fold), index=False)
    if set(final_train.video_id) & set(inner_ids) or set(final_train.video_id) & set(outer_validation_ids):
        raise RuntimeError("Fold final training rows leak an inner/outer validation video")
    outputs = {"fold": fold, "miner": miner, "hard_negatives": len(hard_negatives), "train_rows": len(final_train), "inner_videos": len(inner_ids), "outer_validation_videos": len(outer_validation_ids), "models": {}, "windows": [], "videos": []}
    for kind in ["a2mp", "d1"]:
        model_path = config.fold_model_path(kind, fold)
        history_path = config.fold_history_path(kind, fold)
        training = train_head(context, kind, fold, final_train, inner_rows, config.epochs, config.early_stopping_patience, model_path, history_path)
        model = load_trained_head(context, kind, fold, model_path)
        windows = score_rows(model, kind, outer_validation_rows, context)
        videos = aggregate_videos(windows, config.primary_aggregation)
        if len(videos) != 120 or videos.video_label.value_counts().to_dict() != {0: 60, 1: 60}:
            raise RuntimeError(f"Fold {fold} {kind} does not have balanced outer validation predictions")
        windows["model"] = kind
        windows["outer_fold"] = fold
        windows["checkpoint_epoch"] = training["best_epoch"]
        videos["model"] = kind
        videos["outer_fold"] = fold
        videos["checkpoint_epoch"] = training["best_epoch"]
        outputs["models"][kind] = training
        outputs["windows"].append(windows)
        outputs["videos"].append(videos)
    return outputs


def bootstrap_statistics(videos: pd.DataFrame, selected_thresholds: dict[str, float], config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []
    for kind, group in videos.groupby("model"):
        group = group.reset_index(drop=True)
        labels = group.video_label.to_numpy(dtype=int)
        probabilities = group.video_probability.to_numpy(dtype=float)
        negative = np.flatnonzero(labels == 0)
        positive = np.flatnonzero(labels == 1)
        values = {"f1": [], "recall": [], "pr_auc": []}
        threshold = selected_thresholds[kind]
        for _ in range(config.bootstrap_resamples):
            indices = np.concatenate([rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)])
            sampled_labels = labels[indices]
            sampled_probs = probabilities[indices]
            values["f1"].append(float(f1_score(sampled_labels, sampled_probs >= threshold, zero_division=0)))
            values["recall"].append(float(recall_score(sampled_labels, sampled_probs >= threshold, zero_division=0)))
            values["pr_auc"].append(float(average_precision_score(sampled_labels, sampled_probs)))
        for metric, samples in values.items():
            rows.append({"model": kind, "metric": metric, "estimate": float(np.mean(samples)), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "resamples": config.bootstrap_resamples})
    result = pd.DataFrame(rows)
    result.to_csv(config.bootstrap_path, index=False)
    return result


def exact_mcnemar_and_paired_bootstrap(videos: pd.DataFrame, selected_thresholds: dict[str, float], config: Config) -> dict:
    table = videos.pivot(index="video_id", columns="model", values=["video_label", "video_probability"])
    labels = table[("video_label", "a2mp")].to_numpy(dtype=int)
    a2_probs = table[("video_probability", "a2mp")].to_numpy(dtype=float)
    d1_probs = table[("video_probability", "d1")].to_numpy(dtype=float)
    a2_correct = (a2_probs >= selected_thresholds["a2mp"]) == labels
    d1_correct = (d1_probs >= selected_thresholds["d1"]) == labels
    b = int((a2_correct & ~d1_correct).sum())
    c = int((~a2_correct & d1_correct).sum())
    n = b + c
    mcnemar_p = 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, item) for item in range(0, min(b, c) + 1)) / (2 ** n))
    rng = np.random.default_rng(config.seed + 999)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    differences = []
    for _ in range(config.bootstrap_resamples):
        indices = np.concatenate([rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)])
        a2_f1 = f1_score(labels[indices], a2_probs[indices] >= selected_thresholds["a2mp"], zero_division=0)
        d1_f1 = f1_score(labels[indices], d1_probs[indices] >= selected_thresholds["d1"], zero_division=0)
        differences.append(float(d1_f1 - a2_f1))
    result = {
        "models": ["a2mp", "d1"],
        "mcnemar": {"a2_correct_d1_wrong": b, "a2_wrong_d1_correct": c, "discordant": n, "exact_two_sided_p": mcnemar_p},
        "paired_bootstrap_f1_d1_minus_a2": {"estimate": float(np.mean(differences)), "ci95_low": float(np.quantile(differences, 0.025)), "ci95_high": float(np.quantile(differences, 0.975)), "d1_better_fraction": float(np.mean(np.asarray(differences) > 0)), "resamples": config.bootstrap_resamples},
    }
    config.paired_tests_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def cross_fitted_platt(videos: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict]:
    output, report = [], {}
    for kind, group in videos.groupby("model"):
        group = group.copy().reset_index(drop=True)
        calibrated = np.zeros(len(group), dtype=float)
        for fold in range(5):
            train = group.loc[group.outer_fold.ne(fold)]
            heldout = group.loc[group.outer_fold.eq(fold)]
            calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=config.seed + fold)
            calibrator.fit(train[["video_probability"]], train.video_label.astype(int))
            calibrated[heldout.index.to_numpy()] = calibrator.predict_proba(heldout[["video_probability"]])[:, 1]
        group["cross_fitted_platt_probability"] = calibrated
        raw_brier = float(brier_score_loss(group.video_label, group.video_probability))
        calibrated_brier = float(brier_score_loss(group.video_label, calibrated))
        report[kind] = {"raw_brier": raw_brier, "cross_fitted_platt_brier": calibrated_brier}
        output.append(group)
    calibrated_videos = pd.concat(output, ignore_index=True)
    calibrated_videos.to_csv(config.calibrated_videos_path, index=False)
    config.calibration_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return calibrated_videos, report


def write_selection(config: Config, comparison: pd.DataFrame, paired: dict) -> str:
    a2 = comparison.loc[comparison.model.eq("a2mp")].iloc[0]
    d1 = comparison.loc[comparison.model.eq("d1")].iloc[0]
    if d1.recall >= config.recall_floor and d1.f1 > a2.f1:
        chosen, reason = "D1", "Higher OOF F1 while meeting the frozen Recall floor."
    else:
        chosen, reason = "A2-MP", "D1 did not exceed A2-MP OOF F1 under the frozen selection rule."
    text = "\n".join([
        "# Final OOF model selection",
        "",
        f"Selected video-only model: **{chosen}**.",
        "",
        reason,
        "",
        "| model | threshold | accuracy | precision | recall | F1 | PR-AUC | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {row.model} | {row.threshold:.2f} | {row.accuracy:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.pr_auc:.3f} | {row.roc_auc:.3f} |" for _, row in comparison.iterrows()],
        "",
        f"Paired bootstrap D1-A2 F1 CI: [{paired['paired_bootstrap_f1_d1_minus_a2']['ci95_low']:.4f}, {paired['paired_bootstrap_f1_d1_minus_a2']['ci95_high']:.4f}].",
        f"Exact McNemar p-value: {paired['mcnemar']['exact_two_sided_p']:.6f}.",
        "",
        "This selection is based on OOF probabilities. Final all-data retraining and reusable arbitrary-MP4 inference are a separate hand-off step.",
    ])
    config.selection_path.write_text(text, encoding="utf-8")
    return chosen


def run_final_oof_cv() -> dict:
    context = build_context()
    config: Config = context["config"]
    all_windows, all_videos, fold_reports = [], [], []
    for fold in range(5):
        print(f"===== Final OOF fold {fold}/4 =====", flush=True)
        result = run_fold(context, fold)
        all_windows.extend(result["windows"])
        all_videos.extend(result["videos"])
        fold_reports.append({key: value for key, value in result.items() if key not in {"windows", "videos"}})
    windows = pd.concat(all_windows, ignore_index=True)
    videos = pd.concat(all_videos, ignore_index=True)
    windows.to_csv(config.oof_windows_path, index=False)
    videos.to_csv(config.oof_videos_path, index=False)
    if len(videos) != 1200 or videos.groupby("model").size().to_dict() != {"a2mp": 600, "d1": 600}:
        raise RuntimeError("Final OOF must save exactly 600 video predictions for each base model")
    if videos.duplicated(["model", "video_id"]).any():
        raise RuntimeError("Final OOF contains repeated base-model video predictions")

    rows, selected_thresholds = [], {}
    for kind, group in videos.groupby("model"):
        selected = select_threshold(group.video_label.to_numpy(dtype=int), group.video_probability.to_numpy(dtype=float), config.recall_floor)
        fixed_half = metrics_at_threshold(group.video_label.to_numpy(dtype=int), group.video_probability.to_numpy(dtype=float), 0.5)
        selected_thresholds[kind] = float(selected["threshold"])
        rows.append({"model": kind, "evaluation": "selected_threshold_oof", **selected})
        rows.append({"model": kind, "evaluation": "fixed_threshold_0.50", **fixed_half})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.comparison_path, index=False)
    primary = comparison.loc[comparison.evaluation.eq("selected_threshold_oof")].copy().sort_values("model").reset_index(drop=True)
    bootstrap = bootstrap_statistics(videos, selected_thresholds, config)
    paired = exact_mcnemar_and_paired_bootstrap(videos, selected_thresholds, config)
    calibrated_videos, calibration = cross_fitted_platt(videos, config)
    chosen = write_selection(config, primary, paired)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed", "scope": "five_fold_outer_oof_full_mp4", "models": ["a2mp", "d1"], "selected_thresholds": selected_thresholds, "primary_comparison": primary.to_dict(orient="records"), "fold_reports": fold_reports, "bootstrap_path": str(config.bootstrap_path), "paired_tests": paired, "calibration": calibration, "selected_model": chosen, "not_final_all_data_retrain": True}
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = pd.read_csv(config.registry_path)
    row = {"run_id": "V3_10_FINAL_OOF_A2MP_D1", "stage": "V3-10 final_five_fold_oof", "model_id": "A2-MP vs D1 fold-local OOF", "dataset_version": "v3_all_600_sliding", "split_version": "cv_folds_v3_5fold", "window_version": "fold_local_core_hardnegative_top3mean", "feature_version": "final_oof_rgb_motion_features", "augmentation_version": "none", "checkpoint_path": "models_v3/final_oof_{a2mp,d1}_fold_*.pt", "config_path": "notebooks/40_v3_final_oof_cv.ipynb", "git_commit": "not_available", "status": "completed", "primary_metric": "OOF_accident_f1_recall_floor", "primary_value": float(primary.loc[primary.model.eq("d1"), "f1"].iloc[0]), "notes": f"Selected={chosen}; no ensemble after E1-D gate."}
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"summary": summary, "comparison": comparison, "bootstrap": bootstrap, "calibrated_videos": calibrated_videos}


def context_report(context: dict) -> dict:
    return {"device": str(context["device"]), "union_sequences": len(context["union"]), "videos": len(context["videos"]), "rgb_features": len(context["rgb"]), "motion_features": len(context["motion"]), "outer_fold_videos": context["videos"].groupby("outer_fold").size().to_dict(), "models": ["a2mp", "d1"], "primary_aggregation": context["config"].primary_aggregation}
