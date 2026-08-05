"""Resumable V3-4A A2-MP-HN1 development experiment.

The module deliberately keeps the ResNet18 encoder frozen.  It creates a
feature cache for a fixed, documented V3 training subset and all V3
validation sliding windows, then trains only the mean-max pooling head.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    num_frames: int = 16
    target_height: int = 224
    target_width: int = 320
    feature_dim: int = 512
    sequence_batch_size: int = 2
    frame_batch_size: int = 32
    save_every_batches: int = 25
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    primary_aggregation: str = "top3_mean"
    recall_floor: float = 0.85
    normal_negative_target: int | None = None
    preprocessing_version: str = "v2_multipos_rgb_letterbox_replicate_224x320"

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
    def v2_model_root(self) -> Path:
        return self.data_root / "models_v2"

    @property
    def sequence_manifest_path(self) -> Path:
        return self.manifest_root / "sequence_manifest_v3_sliding.csv"

    @property
    def hard_negatives_path(self) -> Path:
        return self.manifest_root / "hard_negatives_round1.csv"

    @property
    def train_windows_path(self) -> Path:
        return self.manifest_root / "a2mp_hn1_train_windows.csv"

    @property
    def source_hn_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn_train_negative_features.pt"

    @property
    def partial_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features_partial.pt"

    @property
    def features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.v2_model_root / "resnet18_meanmax_pooling_frozen_multipos_best.pt"

    @property
    def best_model_path(self) -> Path:
        return self.model_root / "a2mp_hn1_frozen_best.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "a2mp_hn1_training_history.csv"

    @property
    def validation_windows_csv_path(self) -> Path:
        return self.prediction_root / "a2mp_hn1_validation_window_predictions.csv"

    @property
    def validation_windows_parquet_path(self) -> Path:
        return self.prediction_root / "a2mp_hn1_validation_window_predictions.parquet"

    @property
    def validation_videos_path(self) -> Path:
        return self.prediction_root / "a2mp_hn1_validation_video_predictions.csv"

    @property
    def aggregation_path(self) -> Path:
        return self.report_root / "a2mp_hn1_aggregation_ablation.csv"

    @property
    def failures_path(self) -> Path:
        return self.report_root / "a2mp_hn1_feature_decode_failures.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "a2mp_hn1_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


class MeanMaxHead(nn.Module):
    """The frozen A2-MP pooling head architecture, without a second encoder."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(0.35),
            nn.Linear(feature_dim * 2, 1),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        mean_features = sequence_features.mean(dim=1)
        max_features = sequence_features.max(dim=1).values
        return self.classifier(torch.cat([mean_features, max_features], dim=1)).squeeze(1)


class FeatureDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, features_by_sequence: dict[str, torch.Tensor]) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.features_by_sequence = features_by_sequence

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        return (
            self.features_by_sequence[row.sequence_id],
            torch.tensor(float(row.hard_label), dtype=torch.float32),
            torch.tensor(float(row.loss_weight), dtype=torch.float32),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_directories(config: Config) -> None:
    for directory in [config.processed_root, config.prediction_root, config.report_root, config.model_root]:
        directory.mkdir(parents=True, exist_ok=True)


def round_robin_negative_selection(pool: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Choose windows across videos before taking another window from one video."""
    if target > len(pool):
        raise ValueError("normal-negative target exceeds the available pool")
    rng = np.random.default_rng(seed)
    queues: dict[str, list[int]] = {}
    for video_id, group in pool.groupby("video_id", sort=True):
        positions = rng.permutation(len(group)).tolist()
        queues[str(video_id)] = group.iloc[positions].index.tolist()
    chosen_indices: list[int] = []
    while len(chosen_indices) < target:
        progressed = False
        for video_id in sorted(queues, key=lambda value: int(value)):
            if queues[video_id] and len(chosen_indices) < target:
                chosen_indices.append(queues[video_id].pop(0))
                progressed = True
        if not progressed:
            break
    if len(chosen_indices) != target:
        raise RuntimeError("round-robin selection ended before its requested target")
    return pool.loc[chosen_indices].copy()


def build_context(config: Config) -> dict:
    """Build and freeze the exact training and validation sequence sets."""
    prepare_directories(config)
    for required in [
        config.sequence_manifest_path,
        config.hard_negatives_path,
        config.source_hn_features_path,
        config.checkpoint_path,
        config.registry_path,
    ]:
        if not required.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {required}")

    manifest = pd.read_csv(config.sequence_manifest_path).copy()
    manifest["video_id"] = manifest["video_id"].astype(str)
    manifest["sequence_id"] = manifest["sequence_id"].astype(str)
    hard_negative_rows = pd.read_csv(config.hard_negatives_path).copy()
    hard_negative_rows["video_id"] = hard_negative_rows["video_id"].astype(str)
    hard_negative_ids = set(hard_negative_rows["sequence_id"].astype(str))

    positives = manifest.loc[
        manifest["split"].eq("train")
        & manifest["video_label"].eq(1)
        & manifest["window_role"].eq("positive_core")
        & manifest["hard_label"].eq(1)
    ].copy()
    negative_pool = manifest.loc[
        manifest["split"].eq("train")
        & manifest["video_label"].eq(0)
        & manifest["window_role"].eq("negative_video")
        & manifest["hard_label"].eq(0)
    ].copy()
    hard_negatives = negative_pool.loc[negative_pool["sequence_id"].isin(hard_negative_ids)].copy()
    normal_pool = negative_pool.loc[~negative_pool["sequence_id"].isin(hard_negative_ids)].copy()
    validation = manifest.loc[manifest["split"].eq("validation")].copy()

    if positives.empty or hard_negatives.empty or validation.empty:
        raise RuntimeError("V3-4 input selection is unexpectedly empty")
    if len(hard_negatives) != len(hard_negative_ids):
        raise RuntimeError("A hard-negative id is not a valid train negative sequence")
    if not hard_negatives["video_label"].eq(0).all() or not hard_negatives["split"].eq("train").all():
        raise RuntimeError("hard-negative file contains a non-train-negative row")
    if not validation["split"].eq("validation").all():
        raise RuntimeError("validation selection includes a train row")

    normal_target = config.normal_negative_target or len(positives)
    normal_negatives = round_robin_negative_selection(normal_pool, int(normal_target), config.seed)
    if set(normal_negatives.sequence_id) & set(hard_negatives.sequence_id):
        raise RuntimeError("normal and hard-negative samples overlap")

    positives = positives.copy()
    normal_negatives = normal_negatives.copy()
    hard_negatives = hard_negatives.copy()
    positives["training_role"] = "positive_core"
    normal_negatives["training_role"] = "normal_negative"
    hard_negatives["training_role"] = "hard_negative_r1"
    positives["hard_negative_multiplier"] = 1.0
    normal_negatives["hard_negative_multiplier"] = 1.0
    hard_negatives["hard_negative_multiplier"] = 1.5
    negative_loss_total = float(normal_negatives["hard_negative_multiplier"].sum() + hard_negatives["hard_negative_multiplier"].sum())
    positive_balance_weight = negative_loss_total / len(positives)
    positives["loss_weight"] = positive_balance_weight
    normal_negatives["loss_weight"] = 1.0
    hard_negatives["loss_weight"] = 1.5

    train_rows = pd.concat([positives, normal_negatives, hard_negatives], ignore_index=True)
    train_rows = train_rows.sort_values(["training_role", "video_id", "window_index"]).reset_index(drop=True)
    if train_rows["sequence_id"].duplicated().any():
        raise RuntimeError("training sequence ids must be unique")
    if not train_rows["split"].eq("train").all():
        raise RuntimeError("training manifest contains a validation row")

    expected_rows = pd.concat([train_rows, validation], ignore_index=True)
    expected_rows = expected_rows.drop_duplicates("sequence_id", keep="first").copy()
    expected_rows["cache_scope"] = np.where(expected_rows["split"].eq("train"), "train_selected", "validation_full_mp4")
    expected_rows = expected_rows.sort_values(["split", "video_id", "window_index"]).reset_index(drop=True)
    if expected_rows["sequence_id"].duplicated().any():
        raise RuntimeError("feature-cache sequence ids must be unique")

    train_rows.to_csv(config.train_windows_path, index=False)
    selection_payload = {
        "source_manifest_sha256": sha256_file(config.sequence_manifest_path),
        "hard_negative_sha256": sha256_file(config.hard_negatives_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "preprocessing_version": config.preprocessing_version,
        "seed": config.seed,
        "num_frames": config.num_frames,
        "target_height": config.target_height,
        "target_width": config.target_width,
        "normal_negative_target": int(normal_target),
        "train_sequence_ids": train_rows["sequence_id"].tolist(),
        "validation_sequence_ids": validation["sequence_id"].tolist(),
    }
    run_signature = sha256_text(json.dumps(selection_payload, sort_keys=True))
    return {
        "config": config,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "manifest": manifest,
        "train_rows": train_rows,
        "validation_rows": validation,
        "expected_rows": expected_rows,
        "selection_payload": selection_payload,
        "run_signature": run_signature,
        "positive_balance_weight": positive_balance_weight,
    }


def build_encoder(config: Config, device: torch.device) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def build_head_from_a2mp(config: Config, device: torch.device) -> tuple[MeanMaxHead, dict, str]:
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    head = MeanMaxHead(config.feature_dim).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])
    signature = "epoch={}|pr_auc={:.12f}|sha256={}".format(
        checkpoint["epoch"], checkpoint["validation_pr_auc"], sha256_file(config.checkpoint_path)
    )
    return head, checkpoint, signature


def resize_letterbox_rgb(image_rgb: np.ndarray, config: Config) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(config.target_width / width, config.target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=interpolation)
    pad_x, pad_y = config.target_width - new_width, config.target_height - new_height
    left, right = pad_x // 2, pad_x - pad_x // 2
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)


def normalized_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float().div_(255.0)
    return (tensor - mean) / std


def decode_rgb_at_timestamp(cap: cv2.VideoCapture, timestamp: float, fps: float):
    step = 1.0 / fps if np.isfinite(fps) and fps > 0 else 1.0 / 30.0
    for attempt, offset in enumerate((0.0, step, -step, 2 * step)):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp) + offset) * 1000.0)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "exact" if attempt == 0 else f"seek_fallback_{attempt}"
    return None, "decode_failed"


def decode_sequence(row: pd.Series, config: Config):
    cap = cv2.VideoCapture(str(row.video_path))
    if not cap.isOpened():
        return None, {"sequence_id": row.sequence_id, "video_id": row.video_id, "video_path": row.video_path, "window_start": row.window_start, "window_end": row.window_end, "error_reason": "cannot_open"}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    timestamps = np.linspace(float(row.window_start), float(row.window_end), num=config.num_frames, endpoint=False)
    frames, statuses, previous_frame = [], [], None
    try:
        for timestamp in timestamps:
            frame_rgb, status = decode_rgb_at_timestamp(cap, timestamp, fps)
            if frame_rgb is None and previous_frame is not None:
                frame_rgb, status = previous_frame.copy(), "repeated_previous_after_decode_failure"
            if frame_rgb is None:
                return None, {"sequence_id": row.sequence_id, "video_id": row.video_id, "video_path": row.video_path, "window_start": row.window_start, "window_end": row.window_end, "error_reason": status}
            previous_frame = frame_rgb
            frames.append(normalized_tensor(resize_letterbox_rgb(frame_rgb, config)))
            statuses.append(status)
    finally:
        cap.release()
    return torch.stack(frames), {"sequence_id": row.sequence_id, "decode_status": ";".join(sorted(set(statuses))), "valid_frames": config.num_frames}


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_feature_payload(path: Path, run_signature: str, config: Config) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not path.is_file():
        return {}, {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_signature") != run_signature or payload.get("preprocessing_version") != config.preprocessing_version:
        return {}, {}
    features = {
        str(sequence_id): feature.float().cpu()
        for sequence_id, feature in payload.get("features_by_sequence", {}).items()
        if tuple(feature.shape) == (config.num_frames, config.feature_dim)
    }
    sources = {str(key): str(value) for key, value in payload.get("feature_source_by_sequence", {}).items() if str(key) in features}
    return features, sources


def seed_hard_negative_features(context: dict) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    config: Config = context["config"]
    payload = torch.load(config.source_hn_features_path, map_location="cpu", weights_only=False)
    expected_manifest_sha = context["selection_payload"]["source_manifest_sha256"]
    if payload.get("manifest_sha256") != expected_manifest_sha:
        raise RuntimeError("V3-3 hard-negative feature cache has a different source manifest")
    if payload.get("checkpoint_sha256") != context["selection_payload"]["checkpoint_sha256"]:
        raise RuntimeError("V3-3 hard-negative feature cache has a different A2-MP checkpoint")
    if payload.get("preprocessing_version") != config.preprocessing_version:
        raise RuntimeError("V3-3 hard-negative feature cache has a different preprocessing contract")
    target_ids = set(context["expected_rows"]["sequence_id"])
    features, sources = {}, {}
    for sequence_id, feature in zip(payload["sequence_ids"], payload["features"]):
        sequence_id = str(sequence_id)
        if sequence_id in target_ids and tuple(feature.shape) == (config.num_frames, config.feature_dim):
            features[sequence_id] = feature.float().cpu()
            sources[sequence_id] = "reused_v3_3_hard_negative_cache"
    return features, sources


def cache_preflight(context: dict) -> dict:
    """Decode two non-reused rows and verify the exact encoder contract."""
    config: Config = context["config"]
    device: torch.device = context["device"]
    seed_features, _ = seed_hard_negative_features(context)
    candidates = context["expected_rows"].loc[~context["expected_rows"]["sequence_id"].isin(seed_features)].head(2)
    if len(candidates) < 2:
        candidates = context["expected_rows"].head(2)
    encoder = build_encoder(config, device)
    start = time.perf_counter()
    tensors = []
    details = []
    for _, row in candidates.iterrows():
        tensor, row_details = decode_sequence(row, config)
        if tensor is None:
            raise RuntimeError(row_details)
        tensors.append(tensor)
        details.append(row_details)
    with torch.inference_mode():
        features = encoder(torch.cat(tensors, dim=0).to(device)).flatten(1).cpu()
    elapsed = time.perf_counter() - start
    if tuple(features.shape) != (len(candidates) * config.num_frames, config.feature_dim):
        raise RuntimeError("unexpected ResNet18 feature shape in preflight")
    return {
        "preflight_rows": len(candidates),
        "preflight_seconds": elapsed,
        "estimated_minutes_missing_only": elapsed * (len(context["expected_rows"]) - len(seed_features)) / len(candidates) / 60.0,
        "feature_shape": tuple(features.shape),
        "reused_hard_negative_features": len(seed_features),
        "decode_statuses": [item["decode_status"] for item in details],
    }


def ensure_feature_cache(context: dict, run_full_cache: bool, max_sequences: int | None = None) -> dict:
    """Return the feature dictionary; cache writing is safe to resume, not parallel."""
    config: Config = context["config"]
    expected_rows = context["expected_rows"]
    expected_ids = expected_rows["sequence_id"].tolist()
    expected_set = set(expected_ids)
    features, sources = load_feature_payload(config.partial_features_path, context["run_signature"], config)
    features = {key: value for key, value in features.items() if key in expected_set}
    sources = {key: value for key, value in sources.items() if key in features}
    seed_features, seed_sources = seed_hard_negative_features(context)
    for sequence_id, feature in seed_features.items():
        if sequence_id not in features:
            features[sequence_id] = feature
            sources[sequence_id] = seed_sources[sequence_id]
    failures: list[dict] = []
    missing = expected_rows.loc[~expected_rows["sequence_id"].isin(features)].copy()
    if max_sequences is not None:
        missing = missing.head(int(max_sequences))
    if run_full_cache and len(missing):
        encoder = build_encoder(config, context["device"])
        batches_since_save = 0
        with torch.inference_mode():
            for batch_start in tqdm(range(0, len(missing), config.sequence_batch_size), desc="A2-MP-HN1 frozen features"):
                batch_rows = [row for _, row in missing.iloc[batch_start:batch_start + config.sequence_batch_size].iterrows()]
                valid_tensors, valid_rows = [], []
                for row in batch_rows:
                    tensor, details = decode_sequence(row, config)
                    if tensor is None:
                        failures.append(details)
                    else:
                        valid_tensors.append(tensor)
                        valid_rows.append(row)
                if valid_tensors:
                    image_batch = torch.cat(valid_tensors, dim=0)
                    chunks = []
                    for frame_start in range(0, len(image_batch), config.frame_batch_size):
                        chunks.append(encoder(image_batch[frame_start:frame_start + config.frame_batch_size].to(context["device"])).flatten(1).cpu())
                    batch_features = torch.cat(chunks, dim=0).reshape(len(valid_tensors), config.num_frames, config.feature_dim)
                    for row, feature in zip(valid_rows, batch_features):
                        features[str(row.sequence_id)] = feature
                        sources[str(row.sequence_id)] = "new_v3_4_decode"
                batches_since_save += 1
                if batches_since_save >= config.save_every_batches:
                    atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "features_by_sequence": features, "feature_source_by_sequence": sources}, config.partial_features_path)
                    batches_since_save = 0
        atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "features_by_sequence": features, "feature_source_by_sequence": sources}, config.partial_features_path)

    failure_table = pd.DataFrame(failures, columns=["sequence_id", "video_id", "video_path", "window_start", "window_end", "error_reason"])
    failure_table.to_csv(config.failures_path, index=False)
    missing_after = [sequence_id for sequence_id in expected_ids if sequence_id not in features]
    complete = not missing_after
    if complete:
        atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "sequence_ids": expected_ids, "features": torch.stack([features[key] for key in expected_ids]), "feature_source_by_sequence": sources}, config.features_path)
    return {
        "features_by_sequence": features,
        "feature_source_by_sequence": sources,
        "features_available": len(features),
        "features_expected": len(expected_ids),
        "missing_after_run": len(missing_after),
        "failures_this_run": len(failure_table),
        "complete": complete,
        "new_decode_requested": bool(run_full_cache),
        "new_decode_rows_this_run": len(missing),
    }


def score_window_rows(head: nn.Module, rows: pd.DataFrame, features_by_sequence: dict[str, torch.Tensor], device: torch.device, batch_size: int = 256) -> pd.DataFrame:
    features = torch.stack([features_by_sequence[str(sequence_id)] for sequence_id in rows["sequence_id"]])
    logits, probabilities = [], []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch_logits = head(features[start:start + batch_size].to(device)).cpu()
            logits.append(batch_logits)
            probabilities.append(torch.sigmoid(batch_logits))
    scored = rows.copy()
    scored["window_logit"] = torch.cat(logits).numpy()
    scored["positive_probability"] = torch.cat(probabilities).numpy()
    return scored


def aggregate_videos(window_predictions: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    records = []
    for video_id, group in window_predictions.groupby("video_id", sort=True):
        probabilities = group["positive_probability"].to_numpy(dtype=float)
        logits = group["window_logit"].to_numpy(dtype=float)
        if aggregation == "max":
            probability = float(probabilities.max())
        elif aggregation == "mean":
            probability = float(probabilities.mean())
        elif aggregation.startswith("top") and aggregation.endswith("_mean"):
            k = int(aggregation.removeprefix("top").removesuffix("_mean"))
            probability = float(np.sort(probabilities)[-min(k, len(probabilities)):].mean())
        elif aggregation == "noisy_or":
            probability = float(1.0 - np.exp(np.log(np.clip(1.0 - probabilities, 1e-7, 1.0)).sum()))
        elif aggregation == "logsumexp":
            shifted = logits - logits.max()
            aggregate_logit = logits.max() + np.log(np.exp(shifted).sum()) - np.log(len(logits))
            probability = float(1.0 / (1.0 + np.exp(-aggregate_logit)))
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
        records.append({"video_id": str(video_id), "video_label": int(group["video_label"].iloc[0]), "window_count": len(group), "video_probability": probability, "aggregation": aggregation})
    return pd.DataFrame(records)


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, recall_floor: float) -> dict:
    candidates = [metrics_at_threshold(labels, probabilities, float(value)) for value in np.arange(0.05, 0.951, 0.01)]
    eligible = [item for item in candidates if item["recall"] >= recall_floor]
    pool = eligible if eligible else candidates
    return max(pool, key=lambda item: (item["f1"], item["precision"], item["threshold"]))


def aggregation_ablation(window_predictions: pd.DataFrame, recall_floor: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, video_tables = [], {}
    for aggregation in ["max", "mean", "top2_mean", "top3_mean", "top5_mean", "noisy_or", "logsumexp"]:
        videos = aggregate_videos(window_predictions, aggregation)
        labels = videos["video_label"].to_numpy(dtype=int)
        probabilities = videos["video_probability"].to_numpy(dtype=float)
        selected = select_threshold(labels, probabilities, recall_floor)
        rows.append({"aggregation": aggregation, **selected})
        video_tables[aggregation] = videos
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True), video_tables


def train_head(context: dict, cache_result: dict) -> dict:
    if not cache_result["complete"]:
        raise RuntimeError("Feature cache is incomplete; resume extraction before training")
    config: Config = context["config"]
    device: torch.device = context["device"]
    seed_everything(config.seed)
    train_rows = context["train_rows"]
    dataset = FeatureDataset(train_rows, cache_result["features_by_sequence"])
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    head, initial_checkpoint, initial_signature = build_head_from_a2mp(config, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    best_pr_auc, best_epoch, stale_epochs = -float("inf"), 0, 0

    for epoch in range(1, config.epochs + 1):
        head.train()
        total_weighted_loss, total_weight = 0.0, 0.0
        for features, labels, loss_weights in loader:
            features, labels, loss_weights = features.to(device), labels.to(device), loss_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (raw_loss * loss_weights).sum() / loss_weights.sum()
            loss.backward()
            optimizer.step()
            total_weighted_loss += float((raw_loss.detach() * loss_weights).sum().cpu())
            total_weight += float(loss_weights.sum().cpu())

        validation_windows = score_window_rows(head, context["validation_rows"], cache_result["features_by_sequence"], device)
        validation_videos = aggregate_videos(validation_windows, config.primary_aggregation)
        validation_pr_auc = float(average_precision_score(validation_videos.video_label, validation_videos.video_probability))
        history.append({"epoch": epoch, "weighted_train_loss": total_weighted_loss / total_weight, "validation_pr_auc_top3_mean": validation_pr_auc})
        print(history[-1])
        if validation_pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale_epochs = validation_pr_auc, epoch, 0
            torch.save({
                "model_state_dict": head.state_dict(),
                "epoch": epoch,
                "validation_pr_auc_top3_mean": validation_pr_auc,
                "initial_checkpoint_signature": initial_signature,
                "run_signature": context["run_signature"],
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
            }, config.best_model_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    pd.DataFrame(history).to_csv(config.history_path, index=False)
    return {"best_epoch": best_epoch, "best_validation_pr_auc_top3_mean": best_pr_auc, "initial_checkpoint": initial_checkpoint}


def write_parquet_or_csv(frame: pd.DataFrame, parquet_path: Path, csv_path: Path) -> str:
    frame.to_csv(csv_path, index=False)
    try:
        frame.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except (ImportError, ModuleNotFoundError, ValueError):
        return str(csv_path)


def evaluate_best_model(context: dict, cache_result: dict) -> dict:
    if not cache_result["complete"] or not context["config"].best_model_path.is_file():
        raise RuntimeError("complete cache and best model are required for evaluation")
    config: Config = context["config"]
    head, _, initial_signature = build_head_from_a2mp(config, context["device"])
    best = torch.load(config.best_model_path, map_location=context["device"], weights_only=False)
    if best.get("run_signature") != context["run_signature"]:
        raise RuntimeError("best checkpoint belongs to a different V3-4 feature selection")
    head.load_state_dict(best["model_state_dict"])
    validation_windows = score_window_rows(head, context["validation_rows"], cache_result["features_by_sequence"], context["device"])
    validation_windows["checkpoint_epoch"] = int(best["epoch"])
    validation_windows["run_signature"] = context["run_signature"]
    saved_window_path = write_parquet_or_csv(validation_windows, config.validation_windows_parquet_path, config.validation_windows_csv_path)
    ablation, video_tables = aggregation_ablation(validation_windows, config.recall_floor)
    ablation.to_csv(config.aggregation_path, index=False)
    primary_videos = video_tables[config.primary_aggregation].copy()
    primary_metrics = select_threshold(primary_videos.video_label.to_numpy(dtype=int), primary_videos.video_probability.to_numpy(dtype=float), config.recall_floor)
    primary_videos["threshold"] = primary_metrics["threshold"]
    primary_videos["prediction"] = (primary_videos["video_probability"] >= primary_metrics["threshold"]).astype(int)
    primary_videos.to_csv(config.validation_videos_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_id": "A2-MP-HN1 frozen ResNet18 + mean-max pooling",
        "development_split": "fixed_480_train_120_validation",
        "primary_aggregation_frozen": config.primary_aggregation,
        "primary_metrics_validation_selected_threshold": primary_metrics,
        "best_epoch": int(best["epoch"]),
        "best_validation_pr_auc_top3_mean": float(best["validation_pr_auc_top3_mean"]),
        "initial_checkpoint_signature": initial_signature,
        "run_signature": context["run_signature"],
        "train_rows": int(len(context["train_rows"])),
        "validation_windows": int(len(context["validation_rows"])),
        "hard_negative_train_rows": int(context["train_rows"]["training_role"].eq("hard_negative_r1").sum()),
        "positive_loss_weight": float(context["positive_balance_weight"]),
        "window_predictions_path": saved_window_path,
        "video_predictions_path": str(config.validation_videos_path),
        "aggregation_ablation_path": str(config.aggregation_path),
        "not_final_cv_result": True,
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_registry(context, summary)
    return {"summary": summary, "aggregation_ablation": ablation, "primary_videos": primary_videos, "validation_windows": validation_windows}


def update_registry(context: dict, summary: dict) -> None:
    config: Config = context["config"]
    registry = pd.read_csv(config.registry_path)
    row = {
        "run_id": "V3_04_A2MP_HN1",
        "stage": "V3-4 strong_rgb_baseline",
        "model_id": "A2-MP-HN1 frozen ResNet18 mean-max",
        "dataset_version": "v3_sliding_core_context_v1",
        "split_version": "fixed_480_120_development",
        "window_version": "5s_stride2.5s",
        "feature_version": "a2mp_hn1_frozen_features_v1",
        "augmentation_version": "none",
        "checkpoint_path": str(config.best_model_path),
        "config_path": "notebooks/34_v3_a2mp_hn1_baseline.ipynb",
        "git_commit": "not_available",
        "status": "completed",
        "primary_metric": "validation_f1_top3_mean_recall_floor",
        "primary_value": summary["primary_metrics_validation_selected_threshold"]["f1"],
        "notes": "Development only; hard-negative mining must be fold-local during final CV.",
    }
    registry = registry.loc[~registry["run_id"].eq(row["run_id"])]
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry.to_csv(config.registry_path, index=False)


def context_report(context: dict) -> dict:
    train_rows = context["train_rows"]
    return {
        "device": str(context["device"]),
        "run_signature": context["run_signature"],
        "train_rows": int(len(train_rows)),
        "train_role_counts": train_rows["training_role"].value_counts().to_dict(),
        "train_video_label_counts": train_rows["video_label"].value_counts().to_dict(),
        "validation_windows_all": int(len(context["validation_rows"])),
        "validation_videos": int(context["validation_rows"]["video_id"].nunique()),
        "positive_loss_weight": float(context["positive_balance_weight"]),
        "train_manifest_path": str(context["config"].train_windows_path),
    }
