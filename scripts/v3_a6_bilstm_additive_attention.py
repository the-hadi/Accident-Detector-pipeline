"""V3-7 A6: frozen ResNet18 features + BiLSTM + additive attention + FFN.

This module intentionally trains only a small temporal head.  It reuses the
already validated A2-MP-HN1 RGB feature cache and therefore never decodes an
MP4 file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

from v3_a2mp_hn1 import aggregate_videos, metrics_at_threshold, select_threshold, sha256_file


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    num_frames: int = 16
    feature_dim: int = 512
    hidden_size: int = 128
    attention_dim: int = 128
    dropout: float = 0.35
    batch_size: int = 64
    epochs: int = 45
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    recall_floor: float = 0.85
    primary_aggregation: str = "top3_mean"
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
    def sequence_manifest_path(self) -> Path:
        return self.manifest_root / "sequence_manifest_v3_sliding.csv"

    @property
    def train_manifest_path(self) -> Path:
        return self.manifest_root / "a2mp_hn1_train_windows.csv"

    @property
    def rgb_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def model_path(self) -> Path:
        return self.model_root / "a6_bilstm_additive_attention_frozen_best.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "a6_bilstm_additive_attention_training_history.csv"

    @property
    def window_predictions_path(self) -> Path:
        return self.prediction_root / "a6_bilstm_additive_attention_validation_window_predictions.csv"

    @property
    def video_predictions_path(self) -> Path:
        return self.prediction_root / "a6_bilstm_additive_attention_validation_video_predictions.csv"

    @property
    def attention_path(self) -> Path:
        return self.prediction_root / "a6_bilstm_additive_attention_temporal_attention.csv"

    @property
    def aggregation_path(self) -> Path:
        return self.report_root / "a6_bilstm_additive_attention_aggregation_ablation.csv"

    @property
    def attention_summary_path(self) -> Path:
        return self.report_root / "a6_bilstm_additive_attention_attention_summary.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "a6_bilstm_additive_attention_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


class BiLSTMAdditiveAttentionHead(nn.Module):
    """Ordered-window classifier; attention is over frame positions only."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(config.feature_dim)
        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = config.hidden_size * 2
        self.attention_projection = nn.Linear(temporal_dim, config.attention_dim)
        self.attention_score = nn.Linear(config.attention_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor, return_attention: bool = False):
        encoded, _ = self.lstm(self.input_normalization(features))
        scores = self.attention_score(torch.tanh(self.attention_projection(encoded))).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return (logits, attention) if return_attention else logits


class FeatureDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, features_by_sequence: dict[str, torch.Tensor]) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.features_by_sequence = features_by_sequence

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        return (
            self.features_by_sequence[str(row.sequence_id)],
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


def prepare_directories(config: Config) -> None:
    for directory in [config.prediction_root, config.report_root, config.model_root]:
        directory.mkdir(parents=True, exist_ok=True)


def build_context(config: Config) -> dict:
    prepare_directories(config)
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
        raise RuntimeError("A6 must use the frozen V3 A2-MP-HN1 train and full validation scope")
    if not train.split.eq("train").all() or not validation.split.eq("validation").all():
        raise RuntimeError("A6 split contract is invalid")
    if train.video_id.nunique() != 480 or validation.video_id.nunique() != 120:
        raise RuntimeError("A6 video-level split contract is invalid")

    signature = {
        "sequence_manifest_sha256": sha256_file(config.sequence_manifest_path),
        "train_manifest_sha256": sha256_file(config.train_manifest_path),
        "rgb_features_sha256": sha256_file(config.rgb_features_path),
        "preprocessing_version": config.preprocessing_version,
        "architecture": "bilstm128x2_additive_attention128_ffn128_v1",
        "sequence_ids": expected.sequence_id.tolist(),
    }
    return {
        "config": config,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train": train,
        "validation": validation,
        "expected": expected,
        "signature": signature,
        "run_signature": sha256_text(json.dumps(signature, sort_keys=True)),
    }


def load_rgb_features(context: dict) -> dict[str, torch.Tensor]:
    config: Config = context["config"]
    payload = torch.load(config.rgb_features_path, map_location="cpu", weights_only=False)
    if payload.get("preprocessing_version") != config.preprocessing_version:
        raise RuntimeError("A6 RGB feature cache preprocessing is incompatible")
    sequence_ids = [str(value) for value in payload.get("sequence_ids", [])]
    features = payload.get("features")
    if len(sequence_ids) != len(features):
        raise RuntimeError("A6 RGB feature cache has mismatched IDs and features")
    by_sequence = {
        sequence_id: feature.float().cpu()
        for sequence_id, feature in zip(sequence_ids, features)
        if tuple(feature.shape) == (config.num_frames, config.feature_dim)
    }
    required_ids = context["expected"].sequence_id.tolist()
    missing = [sequence_id for sequence_id in required_ids if sequence_id not in by_sequence]
    if missing:
        raise RuntimeError(f"A6 RGB cache is missing {len(missing)} required valid [16, 512] sequences")
    return {sequence_id: by_sequence[sequence_id] for sequence_id in required_ids}


def cache_preflight(context: dict) -> dict:
    config: Config = context["config"]
    features = load_rgb_features(context)
    probe_ids = context["expected"].head(2).sequence_id.tolist()
    probe = torch.stack([features[sequence_id] for sequence_id in probe_ids])
    model = BiLSTMAdditiveAttentionHead(config).to(context["device"])
    model.eval()
    with torch.inference_mode():
        logits, attention = model(probe.to(context["device"]), return_attention=True)
    if tuple(logits.shape) != (2,) or tuple(attention.shape) != (2, config.num_frames):
        raise RuntimeError("A6 preflight output shapes are invalid")
    if not torch.allclose(attention.sum(dim=1), torch.ones(2, device=attention.device), atol=1e-6):
        raise RuntimeError("A6 additive attention does not sum to one")
    return {
        "device": str(context["device"]),
        "feature_cache": str(config.rgb_features_path),
        "features_available": len(features),
        "features_expected": len(context["expected"]),
        "feature_shape": list(probe.shape[1:]),
        "attention_shape": list(attention.shape[1:]),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "mp4_decoding_required": False,
    }


def score_windows(
    model: nn.Module,
    rows: pd.DataFrame,
    features_by_sequence: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_ids = rows.sequence_id.tolist()
    features = torch.stack([features_by_sequence[sequence_id] for sequence_id in sequence_ids])
    logits, attention = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_logits, batch_attention = model(features[start:start + batch_size].to(device), return_attention=True)
            logits.append(batch_logits.cpu())
            attention.append(batch_attention.cpu())
    all_logits = torch.cat(logits)
    all_attention = torch.cat(attention).numpy()
    scored = rows.copy()
    scored["window_logit"] = all_logits.numpy()
    scored["positive_probability"] = torch.sigmoid(all_logits).numpy()
    attention_table = scored[["sequence_id", "video_id", "video_label", "window_start", "window_end", "window_index"]].copy()
    for position in range(all_attention.shape[1]):
        attention_table[f"attention_frame_{position:02d}"] = all_attention[:, position]
    attention_table["attention_peak_frame"] = all_attention.argmax(axis=1)
    attention_table["attention_entropy"] = -(all_attention * np.log(np.clip(all_attention, 1e-12, 1.0))).sum(axis=1)
    return scored, attention_table


def aggregation_table(window_predictions: pd.DataFrame, recall_floor: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, tables = [], {}
    for aggregation in ["max", "mean", "top2_mean", "top3_mean", "top5_mean", "noisy_or", "logsumexp"]:
        videos = aggregate_videos(window_predictions, aggregation)
        metrics = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), recall_floor)
        rows.append({"aggregation": aggregation, **metrics})
        tables[aggregation] = videos
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True), tables


def train_model(context: dict, features_by_sequence: dict[str, torch.Tensor]) -> dict:
    config: Config = context["config"]
    seed_everything(config.seed)
    dataset = FeatureDataset(context["train"], features_by_sequence)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    model = BiLSTMAdditiveAttentionHead(config).to(context["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_pr_auc, best_epoch, stale = [], -float("inf"), 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss, total_weight = 0.0, 0.0
        for batch_features, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(context["device"]))
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels.to(context["device"]), reduction="none")
            device_weights = weights.to(context["device"])
            loss = (raw_loss * device_weights).sum() / device_weights.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            weighted_loss += float((raw_loss.detach().cpu() * weights).sum())
            total_weight += float(weights.sum())

        validation_windows, _ = score_windows(model, context["validation"], features_by_sequence, context["device"])
        validation_videos = aggregate_videos(validation_windows, config.primary_aggregation)
        pr_auc = float(average_precision_score(validation_videos.video_label, validation_videos.video_probability))
        record = {
            "epoch": epoch,
            "weighted_train_loss": weighted_loss / total_weight,
            "validation_pr_auc_top3_mean": pr_auc,
        }
        history.append(record)
        print(record, flush=True)
        if pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale = pr_auc, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_pr_auc_top3_mean": pr_auc,
                    "run_signature": context["run_signature"],
                    "config": config.__dict__,
                },
                config.model_path,
            )
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.", flush=True)
            break

    pd.DataFrame(history).to_csv(config.history_path, index=False)
    return {
        "best_epoch": best_epoch,
        "best_validation_pr_auc_top3_mean": best_pr_auc,
        "epochs_completed": len(history),
    }


def evaluate_best(context: dict, features_by_sequence: dict[str, torch.Tensor]) -> dict:
    config: Config = context["config"]
    if not config.model_path.is_file():
        raise FileNotFoundError(f"A6 model checkpoint is missing: {config.model_path}")
    checkpoint = torch.load(config.model_path, map_location=context["device"], weights_only=False)
    if checkpoint.get("run_signature") != context["run_signature"]:
        raise RuntimeError("A6 checkpoint does not belong to this frozen feature cache and manifest")
    model = BiLSTMAdditiveAttentionHead(config).to(context["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    windows, attention = score_windows(model, context["validation"], features_by_sequence, context["device"])
    windows["checkpoint_epoch"] = int(checkpoint["epoch"])
    windows.to_csv(config.window_predictions_path, index=False)
    attention.to_csv(config.attention_path, index=False)

    aggregation, video_tables = aggregation_table(windows, config.recall_floor)
    aggregation.to_csv(config.aggregation_path, index=False)
    primary_videos = video_tables[config.primary_aggregation].copy()
    metrics = select_threshold(
        primary_videos.video_label.to_numpy(dtype=int),
        primary_videos.video_probability.to_numpy(dtype=float),
        config.recall_floor,
    )
    primary_videos["threshold"] = metrics["threshold"]
    primary_videos["prediction"] = (primary_videos.video_probability >= metrics["threshold"]).astype(int)
    primary_videos.to_csv(config.video_predictions_path, index=False)

    attention_summary = attention.groupby("video_label", as_index=False).agg(
        windows=("sequence_id", "size"),
        mean_entropy=("attention_entropy", "mean"),
        median_entropy=("attention_entropy", "median"),
        mean_peak_frame=("attention_peak_frame", "mean"),
    )
    attention_summary.to_csv(config.attention_summary_path, index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_id": "A6 frozen ResNet18 + BiLSTM + additive attention + FFN",
        "development_split": "fixed_480_train_120_validation",
        "primary_aggregation_frozen": config.primary_aggregation,
        "primary_metrics_validation_selected_threshold": metrics,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_pr_auc_top3_mean": float(checkpoint["validation_pr_auc_top3_mean"]),
        "run_signature": context["run_signature"],
        "reused_feature_sequences": len(features_by_sequence),
        "mp4_decoding_required": False,
        "window_predictions_path": str(config.window_predictions_path),
        "video_predictions_path": str(config.video_predictions_path),
        "attention_path": str(config.attention_path),
        "aggregation_ablation_path": str(config.aggregation_path),
        "not_final_cv_result": True,
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    registry = pd.read_csv(config.registry_path)
    row = {
        "run_id": "V3_07_A6_BILSTM_ADDITIVE_ATTENTION",
        "stage": "V3-7 ordered_temporal_model",
        "model_id": "A6 frozen ResNet18 + BiLSTM + additive attention + FFN",
        "dataset_version": "v3_sliding_core_context_v1",
        "split_version": "fixed_480_120_development",
        "window_version": "5s_stride2.5s",
        "feature_version": "a2mp_hn1_frozen_rgb_features_v1",
        "augmentation_version": "none_cached_features",
        "checkpoint_path": str(config.model_path),
        "config_path": "notebooks/37_v3_a6_bilstm_additive_attention.ipynb",
        "git_commit": "not_available",
        "status": "completed",
        "primary_metric": "validation_f1_top3_mean_recall_floor",
        "primary_value": metrics["f1"],
        "notes": "BiLSTM models ordered RGB feature sequences; additive attention is interpretive only; no MP4 decoding.",
    }
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"summary": summary, "aggregation_ablation": aggregation, "videos": primary_videos, "attention_summary": attention_summary}


def context_report(context: dict) -> dict:
    config: Config = context["config"]
    return {
        "device": str(context["device"]),
        "run_signature": context["run_signature"],
        "train_windows": len(context["train"]),
        "train_roles": context["train"].training_role.value_counts().to_dict(),
        "validation_windows": len(context["validation"]),
        "validation_videos": context["validation"].video_id.nunique(),
        "expected_feature_sequences": len(context["expected"]),
        "rgb_cache_reused": str(config.rgb_features_path),
        "mp4_decoding_required": False,
    }
