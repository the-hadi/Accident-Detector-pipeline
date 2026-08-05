"""V3-6 D1: frozen RGB + frame-difference ResNet18 feature fusion."""

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
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from v3_a2mp_hn1 import (
    Config as EncoderConfig,
    aggregate_videos,
    atomic_torch_save,
    build_encoder,
    decode_rgb_at_timestamp,
    metrics_at_threshold,
    normalized_tensor,
    resize_letterbox_rgb,
    select_threshold,
    sha256_file,
)


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    num_frames: int = 16
    feature_dim: int = 512
    sequence_batch_size: int = 2
    frame_batch_size: int = 32
    save_every_batches: int = 25
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    recall_floor: float = 0.85
    primary_aggregation: str = "top3_mean"
    preprocessing_version: str = "v2_multipos_rgb_letterbox_replicate_224x320"
    motion_version: str = "absolute_rgb_frame_difference_uint8_before_imagenet_normalization_v1"

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
    def sequence_manifest_path(self) -> Path:
        return self.manifest_root / "sequence_manifest_v3_sliding.csv"

    @property
    def train_manifest_path(self) -> Path:
        return self.manifest_root / "a2mp_hn1_train_windows.csv"

    @property
    def rgb_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def motion_partial_path(self) -> Path:
        return self.processed_root / "d1_motion_features_partial.pt"

    @property
    def motion_features_path(self) -> Path:
        return self.processed_root / "d1_motion_features.pt"

    @property
    def failure_path(self) -> Path:
        return self.report_root / "d1_motion_decode_failures.csv"

    @property
    def model_path(self) -> Path:
        return self.model_root / "d1_rgb_motion_fusion_frozen_best.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "d1_rgb_motion_fusion_training_history.csv"

    @property
    def window_predictions_path(self) -> Path:
        return self.prediction_root / "d1_validation_window_predictions.csv"

    @property
    def video_predictions_path(self) -> Path:
        return self.prediction_root / "d1_validation_video_predictions.csv"

    @property
    def aggregation_path(self) -> Path:
        return self.report_root / "d1_aggregation_ablation.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "d1_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


class RGBMotionFusionHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        input_dim = feature_dim * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.35),
            nn.Linear(input_dim, 1),
        )

    def forward(self, rgb_features: torch.Tensor, motion_features: torch.Tensor) -> torch.Tensor:
        rgb_summary = torch.cat([rgb_features.mean(dim=1), rgb_features.max(dim=1).values], dim=1)
        motion_summary = torch.cat([motion_features.mean(dim=1), motion_features.max(dim=1).values], dim=1)
        return self.classifier(torch.cat([rgb_summary, motion_summary], dim=1)).squeeze(1)


class DualFeatureDataset(Dataset):
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_context(config: Config) -> dict:
    for directory in [config.processed_root, config.prediction_root, config.report_root, config.model_root]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in [config.sequence_manifest_path, config.train_manifest_path, config.rgb_features_path, config.registry_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")
    sequence = pd.read_csv(config.sequence_manifest_path).copy()
    train = pd.read_csv(config.train_manifest_path).copy()
    for frame in [sequence, train]:
        frame["sequence_id"] = frame["sequence_id"].astype(str)
        frame["video_id"] = frame["video_id"].astype(str)
    validation = sequence.loc[sequence.split.eq("validation")].copy()
    expected = pd.concat([train, validation], ignore_index=True).drop_duplicates("sequence_id", keep="first")
    expected = expected.sort_values(["split", "video_id", "window_index"]).reset_index(drop=True)
    if len(train) != 1446 or len(validation) != 1768 or len(expected) != 3214:
        raise RuntimeError("D1 must use the frozen V3-4A train/validation scope")
    if not train.split.eq("train").all() or not validation.split.eq("validation").all():
        raise RuntimeError("D1 split contract is invalid")
    signature = {
        "sequence_manifest_sha256": sha256_file(config.sequence_manifest_path),
        "train_manifest_sha256": sha256_file(config.train_manifest_path),
        "rgb_features_sha256": sha256_file(config.rgb_features_path),
        "preprocessing_version": config.preprocessing_version,
        "motion_version": config.motion_version,
        "sequence_ids": expected.sequence_id.tolist(),
    }
    return {"config": config, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "train": train, "validation": validation, "expected": expected, "run_signature": sha256_text(json.dumps(signature, sort_keys=True)), "signature": signature}


def load_rgb_features(context: dict) -> dict[str, torch.Tensor]:
    config: Config = context["config"]
    payload = torch.load(config.rgb_features_path, map_location="cpu", weights_only=False)
    if payload.get("preprocessing_version") != config.preprocessing_version:
        raise RuntimeError("RGB feature cache preprocessing is incompatible with D1")
    rgb = {str(sequence_id): feature.float().cpu() for sequence_id, feature in zip(payload["sequence_ids"], payload["features"])}
    required = set(context["expected"].sequence_id)
    if not required.issubset(rgb):
        raise RuntimeError("D1 expected sequence is missing from the A2-MP-HN1 RGB cache")
    return {sequence_id: rgb[sequence_id] for sequence_id in context["expected"].sequence_id}


def decode_motion_sequence(row: pd.Series, config: Config):
    encoder_config = EncoderConfig()
    cap = cv2.VideoCapture(str(row.video_path))
    if not cap.isOpened():
        return None, {"sequence_id": row.sequence_id, "video_id": row.video_id, "video_path": row.video_path, "window_start": row.window_start, "window_end": row.window_end, "error_reason": "cannot_open"}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    timestamps = np.linspace(float(row.window_start), float(row.window_end), num=config.num_frames, endpoint=False)
    frames, statuses, previous = [], [], None
    try:
        for timestamp in timestamps:
            rgb, status = decode_rgb_at_timestamp(cap, timestamp, fps)
            if rgb is None and previous is not None:
                rgb, status = previous.copy(), "repeated_previous_after_decode_failure"
            if rgb is None:
                return None, {"sequence_id": row.sequence_id, "video_id": row.video_id, "video_path": row.video_path, "window_start": row.window_start, "window_end": row.window_end, "error_reason": status}
            processed = resize_letterbox_rgb(rgb, encoder_config)
            previous = rgb
            frames.append(processed)
            statuses.append(status)
    finally:
        cap.release()
    differences = [np.zeros_like(frames[0])]
    differences.extend([cv2.absdiff(current, previous) for previous, current in zip(frames[:-1], frames[1:])])
    return torch.stack([normalized_tensor(image) for image in differences]), {"sequence_id": row.sequence_id, "decode_status": ";".join(sorted(set(statuses))), "valid_frames": config.num_frames}


def load_motion_partial(context: dict) -> dict[str, torch.Tensor]:
    config: Config = context["config"]
    if not config.motion_partial_path.is_file():
        return {}
    payload = torch.load(config.motion_partial_path, map_location="cpu", weights_only=False)
    if payload.get("run_signature") != context["run_signature"] or payload.get("motion_version") != config.motion_version:
        return {}
    return {str(sequence_id): feature.float().cpu() for sequence_id, feature in payload.get("features_by_sequence", {}).items() if tuple(feature.shape) == (config.num_frames, config.feature_dim)}


def cache_preflight(context: dict) -> dict:
    config: Config = context["config"]
    encoder = build_encoder(EncoderConfig(), context["device"])
    rows = context["expected"].head(2)
    tensors, statuses = [], []
    started = time.perf_counter()
    for _, row in rows.iterrows():
        tensor, detail = decode_motion_sequence(row, config)
        if tensor is None:
            raise RuntimeError(detail)
        tensors.append(tensor)
        statuses.append(detail["decode_status"])
    with torch.inference_mode():
        features = encoder(torch.cat(tensors).to(context["device"])).flatten(1).cpu()
    elapsed = time.perf_counter() - started
    if tuple(features.shape) != (len(rows) * config.num_frames, config.feature_dim):
        raise RuntimeError("D1 preflight feature shape is invalid")
    return {"preflight_rows": len(rows), "preflight_seconds": elapsed, "expected_motion_sequences": len(context["expected"]), "estimated_minutes": elapsed * len(context["expected"]) / len(rows) / 60.0, "decode_statuses": statuses}


def ensure_motion_cache(context: dict, run_full_cache: bool, max_sequences: int | None = None) -> dict:
    config: Config = context["config"]
    expected_ids = context["expected"].sequence_id.tolist()
    expected_set = set(expected_ids)
    features = {key: value for key, value in load_motion_partial(context).items() if key in expected_set}
    missing = context["expected"].loc[~context["expected"].sequence_id.isin(features)].copy()
    if max_sequences is not None:
        missing = missing.head(int(max_sequences))
    failures = []
    if run_full_cache and len(missing):
        encoder = build_encoder(EncoderConfig(), context["device"])
        batches_since_save = 0
        with torch.inference_mode():
            for batch_start in tqdm(range(0, len(missing), config.sequence_batch_size), desc="D1 motion features"):
                rows = [row for _, row in missing.iloc[batch_start:batch_start + config.sequence_batch_size].iterrows()]
                tensors, valid_rows = [], []
                for row in rows:
                    tensor, detail = decode_motion_sequence(row, config)
                    if tensor is None:
                        failures.append(detail)
                    else:
                        tensors.append(tensor)
                        valid_rows.append(row)
                if tensors:
                    image_batch = torch.cat(tensors)
                    chunks = [encoder(image_batch[start:start + config.frame_batch_size].to(context["device"])).flatten(1).cpu() for start in range(0, len(image_batch), config.frame_batch_size)]
                    batch_features = torch.cat(chunks).reshape(len(tensors), config.num_frames, config.feature_dim)
                    for row, feature in zip(valid_rows, batch_features):
                        features[str(row.sequence_id)] = feature
                batches_since_save += 1
                if batches_since_save >= config.save_every_batches:
                    atomic_torch_save({"run_signature": context["run_signature"], "motion_version": config.motion_version, "features_by_sequence": features}, config.motion_partial_path)
                    batches_since_save = 0
        atomic_torch_save({"run_signature": context["run_signature"], "motion_version": config.motion_version, "features_by_sequence": features}, config.motion_partial_path)
    pd.DataFrame(failures, columns=["sequence_id", "video_id", "video_path", "window_start", "window_end", "error_reason"]).to_csv(config.failure_path, index=False)
    missing_after = [sequence_id for sequence_id in expected_ids if sequence_id not in features]
    complete = not missing_after
    if complete:
        atomic_torch_save({"run_signature": context["run_signature"], "motion_version": config.motion_version, "sequence_ids": expected_ids, "features": torch.stack([features[sequence_id] for sequence_id in expected_ids])}, config.motion_features_path)
    return {"features_by_sequence": features, "features_available": len(features), "features_expected": len(expected_ids), "missing_after_run": len(missing_after), "failures_this_run": len(failures), "complete": complete, "new_decode_rows_this_run": len(missing)}


def score_windows(head: nn.Module, rows: pd.DataFrame, rgb: dict[str, torch.Tensor], motion: dict[str, torch.Tensor], device: torch.device) -> pd.DataFrame:
    sequence_ids = rows.sequence_id.tolist()
    rgb_tensor = torch.stack([rgb[item] for item in sequence_ids])
    motion_tensor = torch.stack([motion[item] for item in sequence_ids])
    logits = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            logits.append(head(rgb_tensor[start:start + 256].to(device), motion_tensor[start:start + 256].to(device)).cpu())
    scored = rows.copy()
    scored["window_logit"] = torch.cat(logits).numpy()
    scored["positive_probability"] = torch.sigmoid(torch.cat(logits)).numpy()
    return scored


def aggregation_table(window_predictions: pd.DataFrame, recall_floor: float):
    rows, tables = [], {}
    for aggregation in ["max", "mean", "top2_mean", "top3_mean", "top5_mean", "noisy_or", "logsumexp"]:
        videos = aggregate_videos(window_predictions, aggregation)
        metrics = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), recall_floor)
        rows.append({"aggregation": aggregation, **metrics})
        tables[aggregation] = videos
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True), tables


def train_model(context: dict, motion_cache: dict) -> dict:
    if not motion_cache["complete"]:
        raise RuntimeError("D1 motion cache is incomplete")
    config: Config = context["config"]
    seed_everything(config.seed)
    rgb = load_rgb_features(context)
    dataset = DualFeatureDataset(context["train"], rgb, motion_cache["features_by_sequence"])
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    head = RGBMotionFusionHead(config.feature_dim).to(context["device"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_pr_auc, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        head.train()
        weighted_loss, total_weight = 0.0, 0.0
        for rgb_batch, motion_batch, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(rgb_batch.to(context["device"]), motion_batch.to(context["device"]))
            raw = F.binary_cross_entropy_with_logits(logits, labels.to(context["device"]), reduction="none")
            loss = (raw * weights.to(context["device"])).sum() / weights.sum().to(context["device"])
            loss.backward()
            optimizer.step()
            weighted_loss += float((raw.detach().cpu() * weights).sum())
            total_weight += float(weights.sum())
        validation_windows = score_windows(head, context["validation"], rgb, motion_cache["features_by_sequence"], context["device"])
        videos = aggregate_videos(validation_windows, config.primary_aggregation)
        pr_auc = float(average_precision_score(videos.video_label, videos.video_probability))
        record = {"epoch": epoch, "weighted_train_loss": weighted_loss / total_weight, "validation_pr_auc_top3_mean": pr_auc}
        history.append(record)
        print(record)
        if pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale = pr_auc, epoch, 0
            torch.save({"model_state_dict": head.state_dict(), "epoch": epoch, "validation_pr_auc_top3_mean": pr_auc, "run_signature": context["run_signature"], "config": config.__dict__}, config.model_path)
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break
    pd.DataFrame(history).to_csv(config.history_path, index=False)
    return {"best_epoch": best_epoch, "best_validation_pr_auc_top3_mean": best_pr_auc, "epochs_completed": len(history)}


def evaluate_best(context: dict, motion_cache: dict) -> dict:
    config: Config = context["config"]
    if not motion_cache["complete"] or not config.model_path.is_file():
        raise RuntimeError("D1 cache or model is missing")
    checkpoint = torch.load(config.model_path, map_location=context["device"], weights_only=False)
    if checkpoint.get("run_signature") != context["run_signature"]:
        raise RuntimeError("D1 model belongs to another feature cache")
    rgb = load_rgb_features(context)
    head = RGBMotionFusionHead(config.feature_dim).to(context["device"])
    head.load_state_dict(checkpoint["model_state_dict"])
    windows = score_windows(head, context["validation"], rgb, motion_cache["features_by_sequence"], context["device"])
    windows["checkpoint_epoch"] = int(checkpoint["epoch"])
    windows.to_csv(config.window_predictions_path, index=False)
    ablation, video_tables = aggregation_table(windows, config.recall_floor)
    ablation.to_csv(config.aggregation_path, index=False)
    videos = video_tables[config.primary_aggregation].copy()
    metrics = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), config.recall_floor)
    videos["threshold"] = metrics["threshold"]
    videos["prediction"] = (videos.video_probability >= metrics["threshold"]).astype(int)
    videos.to_csv(config.video_predictions_path, index=False)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed", "model_id": "D1 frozen ResNet18 RGB + frame-difference feature fusion", "development_split": "fixed_480_train_120_validation", "primary_aggregation_frozen": config.primary_aggregation, "primary_metrics_validation_selected_threshold": metrics, "best_epoch": int(checkpoint["epoch"]), "best_validation_pr_auc_top3_mean": float(checkpoint["validation_pr_auc_top3_mean"]), "run_signature": context["run_signature"], "motion_sequences": len(context["expected"]), "window_predictions_path": str(config.window_predictions_path), "video_predictions_path": str(config.video_predictions_path), "aggregation_ablation_path": str(config.aggregation_path), "not_final_cv_result": True}
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = pd.read_csv(config.registry_path)
    row = {"run_id": "V3_06_D1_RGB_MOTION", "stage": "V3-6 motion_frame_difference", "model_id": "D1 frozen ResNet18 RGB+motion fusion", "dataset_version": "v3_sliding_core_context_v1", "split_version": "fixed_480_120_development", "window_version": "5s_stride2.5s", "feature_version": "d1_motion_features_v1", "augmentation_version": "none", "checkpoint_path": str(config.model_path), "config_path": "notebooks/36_v3_d1_motion.ipynb", "git_commit": "not_available", "status": "completed", "primary_metric": "validation_f1_top3_mean_recall_floor", "primary_value": metrics["f1"], "notes": "RGB cache reused; motion is absolute RGB frame difference."}
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"summary": summary, "aggregation_ablation": ablation, "videos": videos}


def context_report(context: dict) -> dict:
    return {"device": str(context["device"]), "run_signature": context["run_signature"], "train_windows": len(context["train"]), "train_roles": context["train"].training_role.value_counts().to_dict(), "validation_windows": len(context["validation"]), "validation_videos": context["validation"].video_id.nunique(), "motion_sequences": len(context["expected"]), "rgb_cache_reused": str(context["config"].rgb_features_path)}
