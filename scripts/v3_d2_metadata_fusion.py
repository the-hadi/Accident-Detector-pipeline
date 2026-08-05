"""V3-8 D2: conservative metadata ablation and RGB+motion metadata fusion.

The experiment reuses frozen feature caches.  It never opens an MP4 file and
explicitly excludes event time, identifiers, and technical video metadata.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from v3_a2mp_hn1 import aggregate_videos, select_threshold, sha256_file


METADATA_FIELDS = ("weather", "light_conditions", "scene")


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    num_frames: int = 16
    feature_dim: int = 512
    metadata_dropout: float = 0.20
    dropout: float = 0.35
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
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
    def video_manifest_path(self) -> Path:
        return self.data_root / "video_manifest_v2.csv"

    @property
    def rgb_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def motion_features_path(self) -> Path:
        return self.processed_root / "d1_motion_features.pt"

    @property
    def metadata_vocab_path(self) -> Path:
        return self.report_root / "d2_metadata_vocab.json"

    @property
    def metadata_only_prediction_path(self) -> Path:
        return self.prediction_root / "d2_metadata_only_validation_video_predictions.csv"

    @property
    def metadata_only_metrics_path(self) -> Path:
        return self.report_root / "d2_metadata_only_metrics.json"

    @property
    def video_only_model_path(self) -> Path:
        return self.model_root / "d2_video_only_frozen_best.pt"

    @property
    def fusion_model_path(self) -> Path:
        return self.model_root / "d2_video_metadata_fusion_frozen_best.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "d2_training_history.csv"

    @property
    def video_only_window_path(self) -> Path:
        return self.prediction_root / "d2_video_only_validation_window_predictions.csv"

    @property
    def video_only_video_path(self) -> Path:
        return self.prediction_root / "d2_video_only_validation_video_predictions.csv"

    @property
    def video_only_aggregation_path(self) -> Path:
        return self.report_root / "d2_video_only_aggregation_ablation.csv"

    @property
    def fusion_window_path(self) -> Path:
        return self.prediction_root / "d2_video_metadata_fusion_validation_window_predictions.csv"

    @property
    def fusion_video_path(self) -> Path:
        return self.prediction_root / "d2_video_metadata_fusion_validation_video_predictions.csv"

    @property
    def fusion_aggregation_path(self) -> Path:
        return self.report_root / "d2_video_metadata_aggregation_ablation.csv"

    @property
    def bias_report_path(self) -> Path:
        return self.report_root / "d2_metadata_bias_report.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "d2_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


class MetadataDropout(nn.Module):
    """Drop all metadata fields for selected training examples, not individual categories."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0:
            return metadata
        keep = (torch.rand((len(metadata), 1), device=metadata.device) >= self.probability).to(metadata.dtype)
        return metadata * keep


class VideoOnlyHead(nn.Module):
    def __init__(self, video_dim: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(video_dim), nn.Dropout(dropout), nn.Linear(video_dim, 1))

    def forward(self, video_features: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(video_features).squeeze(1)


class VideoMetadataFusionHead(nn.Module):
    def __init__(self, video_dim: int, metadata_dim: int, dropout: float, metadata_dropout: float) -> None:
        super().__init__()
        self.metadata_dropout = MetadataDropout(metadata_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(video_dim + metadata_dim),
            nn.Dropout(dropout),
            nn.Linear(video_dim + metadata_dim, 1),
        )

    def forward(self, video_features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([video_features, self.metadata_dropout(metadata)], dim=1)).squeeze(1)


class FusionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, video_features: dict[str, torch.Tensor], metadata_by_video: dict[str, torch.Tensor]) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.video_features = video_features
        self.metadata_by_video = metadata_by_video

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        return (
            self.video_features[str(row.sequence_id)],
            self.metadata_by_video[str(row.video_id)],
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
    required_paths = [
        config.sequence_manifest_path,
        config.train_manifest_path,
        config.video_manifest_path,
        config.rgb_features_path,
        config.motion_features_path,
        config.registry_path,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")

    sequence = pd.read_csv(config.sequence_manifest_path).copy()
    train = pd.read_csv(config.train_manifest_path).copy()
    manifest = pd.read_csv(config.video_manifest_path).copy()
    for frame in [sequence, train, manifest]:
        frame["video_id"] = frame["video_id"].astype(str)
    for frame in [sequence, train]:
        frame["sequence_id"] = frame["sequence_id"].astype(str)

    validation = sequence.loc[sequence.split.eq("validation")].copy()
    expected = pd.concat([train, validation], ignore_index=True).drop_duplicates("sequence_id", keep="first")
    expected = expected.sort_values(["split", "video_id", "window_index"]).reset_index(drop=True)
    if len(train) != 1446 or len(validation) != 1768 or len(expected) != 3214:
        raise RuntimeError("D2 must use the frozen V3 A2-MP-HN1 train and full validation scope")
    if train.video_id.nunique() != 480 or validation.video_id.nunique() != 120:
        raise RuntimeError("D2 video-level split contract is invalid")
    if not train.split.eq("train").all() or not validation.split.eq("validation").all():
        raise RuntimeError("D2 split contract is invalid")

    metadata = manifest[["video_id", "split", "label", *METADATA_FIELDS]].copy()
    if metadata.video_id.duplicated().any():
        raise RuntimeError("Video metadata manifest must have one row per video")
    metadata = metadata.rename(columns={"label": "video_label"})
    metadata["video_label"] = metadata.video_label.astype(int)
    train_ids, validation_ids = set(train.video_id), set(validation.video_id)
    if train_ids & validation_ids:
        raise RuntimeError("D2 detected a video crossing the frozen split")
    metadata = metadata.loc[metadata.video_id.isin(train_ids | validation_ids)].copy()
    if len(metadata) != 600 or set(metadata.video_id) != train_ids | validation_ids:
        raise RuntimeError("D2 metadata does not cover the selected 600 videos exactly")
    for field in METADATA_FIELDS:
        metadata[field] = metadata[field].fillna("__MISSING__").astype(str)
    expected_split = {video_id: "train" for video_id in train_ids} | {video_id: "validation" for video_id in validation_ids}
    if any(expected_split[video_id] != split for video_id, split in zip(metadata.video_id, metadata.split)):
        raise RuntimeError("Metadata split disagrees with frozen V3 split")

    train_metadata = metadata.loc[metadata.video_id.isin(train_ids)].sort_values("video_id").reset_index(drop=True)
    validation_metadata = metadata.loc[metadata.video_id.isin(validation_ids)].sort_values("video_id").reset_index(drop=True)
    vocabularies = {field: sorted(set(train_metadata[field])) + ["__UNK__"] for field in METADATA_FIELDS}
    metadata_dim = sum(len(values) for values in vocabularies.values())
    vocabulary_payload = {
        "source": str(config.video_manifest_path),
        "fit_split": "train_480_only",
        "fields": list(METADATA_FIELDS),
        "vocabularies": vocabularies,
        "metadata_dim": metadata_dim,
        "excluded_fields": ["time_of_event", "video_id", "video_path", "duration", "fps", "frame_count", "width", "height", "split", "sha256"],
    }
    config.metadata_vocab_path.write_text(json.dumps(vocabulary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signature = {
        "sequence_manifest_sha256": sha256_file(config.sequence_manifest_path),
        "train_manifest_sha256": sha256_file(config.train_manifest_path),
        "video_manifest_sha256": sha256_file(config.video_manifest_path),
        "rgb_features_sha256": sha256_file(config.rgb_features_path),
        "motion_features_sha256": sha256_file(config.motion_features_path),
        "metadata_vocabularies": vocabularies,
        "preprocessing_version": config.preprocessing_version,
        "sequence_ids": expected.sequence_id.tolist(),
    }
    return {
        "config": config,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train": train,
        "validation": validation,
        "expected": expected,
        "metadata": metadata,
        "train_metadata": train_metadata,
        "validation_metadata": validation_metadata,
        "vocabularies": vocabularies,
        "metadata_dim": metadata_dim,
        "signature": signature,
        "run_signature": sha256_text(json.dumps(signature, sort_keys=True)),
    }


def encode_metadata(metadata: pd.DataFrame, vocabularies: dict[str, list[str]]) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    encoded_rows = []
    for _, row in metadata.iterrows():
        values = []
        for field in METADATA_FIELDS:
            vocabulary = vocabularies[field]
            value = row[field] if row[field] in vocabulary else "__UNK__"
            values.extend(float(value == candidate) for candidate in vocabulary)
        encoded_rows.append(values)
    matrix = np.asarray(encoded_rows, dtype=np.float32)
    return matrix, {str(video_id): torch.tensor(vector, dtype=torch.float32) for video_id, vector in zip(metadata.video_id, matrix)}


def load_sequence_features(path: Path, expected_ids: list[str], config: Config, name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if name == "rgb" and payload.get("preprocessing_version") != config.preprocessing_version:
        raise RuntimeError("D2 RGB feature cache preprocessing is incompatible")
    sequence_ids = [str(value) for value in payload.get("sequence_ids", [])]
    features = payload.get("features")
    if len(sequence_ids) != len(features):
        raise RuntimeError(f"D2 {name} cache has mismatched IDs and features")
    by_sequence = {
        sequence_id: feature.float().cpu()
        for sequence_id, feature in zip(sequence_ids, features)
        if tuple(feature.shape) == (config.num_frames, config.feature_dim)
    }
    missing = [sequence_id for sequence_id in expected_ids if sequence_id not in by_sequence]
    if missing:
        raise RuntimeError(f"D2 {name} cache misses {len(missing)} required [16, 512] sequences")
    return {sequence_id: by_sequence[sequence_id] for sequence_id in expected_ids}


def load_video_features(context: dict) -> dict[str, torch.Tensor]:
    config: Config = context["config"]
    expected_ids = context["expected"].sequence_id.tolist()
    rgb = load_sequence_features(config.rgb_features_path, expected_ids, config, "rgb")
    motion = load_sequence_features(config.motion_features_path, expected_ids, config, "motion")
    return {
        sequence_id: torch.cat(
            [rgb[sequence_id].mean(dim=0), rgb[sequence_id].max(dim=0).values, motion[sequence_id].mean(dim=0), motion[sequence_id].max(dim=0).values]
        )
        for sequence_id in expected_ids
    }


def cache_preflight(context: dict) -> dict:
    video_features = load_video_features(context)
    metadata_matrix, metadata_by_video = encode_metadata(context["metadata"], context["vocabularies"])
    config: Config = context["config"]
    probe = context["expected"].head(2)
    video_probe = torch.stack([video_features[sequence_id] for sequence_id in probe.sequence_id])
    metadata_probe = torch.stack([metadata_by_video[video_id] for video_id in probe.video_id])
    fusion = VideoMetadataFusionHead(video_probe.shape[1], metadata_probe.shape[1], config.dropout, config.metadata_dropout).to(context["device"])
    fusion.eval()
    with torch.inference_mode():
        logits = fusion(video_probe.to(context["device"]), metadata_probe.to(context["device"]))
    if tuple(logits.shape) != (2,):
        raise RuntimeError("D2 fusion preflight output shape is invalid")
    return {
        "device": str(context["device"]),
        "expected_sequences": len(context["expected"]),
        "video_feature_shape": list(video_probe.shape[1:]),
        "metadata_dim": int(metadata_probe.shape[1]),
        "metadata_fields": list(METADATA_FIELDS),
        "metadata_rows": int(metadata_matrix.shape[0]),
        "mp4_decoding_required": False,
        "video_only_trainable_parameters": int(sum(item.numel() for item in VideoOnlyHead(video_probe.shape[1], config.dropout).parameters())),
        "fusion_trainable_parameters": int(sum(item.numel() for item in fusion.parameters())),
    }


def score_windows(model: nn.Module, rows: pd.DataFrame, video_features: dict[str, torch.Tensor], metadata_by_video: dict[str, torch.Tensor], device: torch.device) -> pd.DataFrame:
    sequence_ids = rows.sequence_id.tolist()
    features = torch.stack([video_features[sequence_id] for sequence_id in sequence_ids])
    metadata = torch.stack([metadata_by_video[video_id] for video_id in rows.video_id])
    logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            logits.append(model(features[start:start + 256].to(device), metadata[start:start + 256].to(device)).cpu())
    scored = rows.copy()
    all_logits = torch.cat(logits)
    scored["window_logit"] = all_logits.numpy()
    scored["positive_probability"] = torch.sigmoid(all_logits).numpy()
    return scored


def aggregation_table(window_predictions: pd.DataFrame, recall_floor: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, tables = [], {}
    for aggregation in ["max", "mean", "top2_mean", "top3_mean", "top5_mean", "noisy_or", "logsumexp"]:
        videos = aggregate_videos(window_predictions, aggregation)
        metrics = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), recall_floor)
        rows.append({"aggregation": aggregation, **metrics})
        tables[aggregation] = videos
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True), tables


def train_window_head(
    context: dict,
    video_features: dict[str, torch.Tensor],
    metadata_by_video: dict[str, torch.Tensor],
    kind: str,
) -> dict:
    config: Config = context["config"]
    if kind not in {"video_only", "fusion"}:
        raise ValueError(f"Unknown D2 head kind: {kind}")
    seed_everything(config.seed + (0 if kind == "video_only" else 1))
    dataset = FusionDataset(context["train"], video_features, metadata_by_video)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed + (0 if kind == "video_only" else 1)), num_workers=0)
    video_dim = next(iter(video_features.values())).numel()
    metadata_dim = next(iter(metadata_by_video.values())).numel()
    model = VideoOnlyHead(video_dim, config.dropout) if kind == "video_only" else VideoMetadataFusionHead(video_dim, metadata_dim, config.dropout, config.metadata_dropout)
    model = model.to(context["device"])
    model_path = config.video_only_model_path if kind == "video_only" else config.fusion_model_path
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_pr_auc, best_epoch, stale = [], -float("inf"), 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss, total_weight = 0.0, 0.0
        for batch_video, batch_metadata, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_video.to(context["device"]), batch_metadata.to(context["device"]))
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels.to(context["device"]), reduction="none")
            device_weights = weights.to(context["device"])
            loss = (raw_loss * device_weights).sum() / device_weights.sum()
            loss.backward()
            optimizer.step()
            weighted_loss += float((raw_loss.detach().cpu() * weights).sum())
            total_weight += float(weights.sum())

        validation_windows = score_windows(model, context["validation"], video_features, metadata_by_video, context["device"])
        validation_videos = aggregate_videos(validation_windows, config.primary_aggregation)
        pr_auc = float(average_precision_score(validation_videos.video_label, validation_videos.video_probability))
        record = {"model": kind, "epoch": epoch, "weighted_train_loss": weighted_loss / total_weight, "validation_pr_auc_top3_mean": pr_auc}
        history.append(record)
        print(record, flush=True)
        if pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale = pr_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_pr_auc_top3_mean": pr_auc, "run_signature": context["run_signature"], "kind": kind, "config": config.__dict__}, model_path)
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            print(f"{kind}: early stopping at epoch {epoch}; best epoch {best_epoch}.", flush=True)
            break

    return {"kind": kind, "history": history, "best_epoch": best_epoch, "best_pr_auc": best_pr_auc, "epochs_completed": len(history)}


def evaluate_window_head(context: dict, video_features: dict[str, torch.Tensor], metadata_by_video: dict[str, torch.Tensor], kind: str) -> dict:
    config: Config = context["config"]
    model_path = config.video_only_model_path if kind == "video_only" else config.fusion_model_path
    checkpoint = torch.load(model_path, map_location=context["device"], weights_only=False)
    if checkpoint.get("run_signature") != context["run_signature"] or checkpoint.get("kind") != kind:
        raise RuntimeError(f"D2 {kind} checkpoint is incompatible with the frozen inputs")
    video_dim = next(iter(video_features.values())).numel()
    metadata_dim = next(iter(metadata_by_video.values())).numel()
    model = VideoOnlyHead(video_dim, config.dropout) if kind == "video_only" else VideoMetadataFusionHead(video_dim, metadata_dim, config.dropout, config.metadata_dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(context["device"])
    windows = score_windows(model, context["validation"], video_features, metadata_by_video, context["device"])
    windows["checkpoint_epoch"] = int(checkpoint["epoch"])
    window_path = config.video_only_window_path if kind == "video_only" else config.fusion_window_path
    video_path = config.video_only_video_path if kind == "video_only" else config.fusion_video_path
    aggregation_path = config.video_only_aggregation_path if kind == "video_only" else config.fusion_aggregation_path
    windows.to_csv(window_path, index=False)
    aggregation, tables = aggregation_table(windows, config.recall_floor)
    aggregation.to_csv(aggregation_path, index=False)
    videos = tables[config.primary_aggregation].copy()
    metrics = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), config.recall_floor)
    videos["threshold"] = metrics["threshold"]
    videos["prediction"] = (videos.video_probability >= metrics["threshold"]).astype(int)
    videos.to_csv(video_path, index=False)
    return {"kind": kind, "best_epoch": int(checkpoint["epoch"]), "best_pr_auc": float(checkpoint["validation_pr_auc_top3_mean"]), "metrics": metrics, "aggregation": aggregation, "videos": videos}


def run_metadata_only(context: dict) -> dict:
    config: Config = context["config"]
    train_matrix, _ = encode_metadata(context["train_metadata"], context["vocabularies"])
    validation_matrix, _ = encode_metadata(context["validation_metadata"], context["vocabularies"])
    classifier = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=config.seed)
    classifier.fit(train_matrix, context["train_metadata"].video_label.to_numpy(dtype=int))
    probabilities = classifier.predict_proba(validation_matrix)[:, 1]
    labels = context["validation_metadata"].video_label.to_numpy(dtype=int)
    metrics = select_threshold(labels, probabilities, config.recall_floor)
    predictions = context["validation_metadata"][["video_id", "video_label", *METADATA_FIELDS]].copy()
    predictions["video_probability"] = probabilities
    predictions["threshold"] = metrics["threshold"]
    predictions["prediction"] = (probabilities >= metrics["threshold"]).astype(int)
    predictions.to_csv(config.metadata_only_prediction_path, index=False)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "D2-1 regularized logistic regression metadata-only",
        "fit_split": "train_480_only",
        "validation_videos": len(predictions),
        "metadata_fields": list(METADATA_FIELDS),
        "metrics_validation_selected_threshold": metrics,
        "warning": "Metadata-only is a shortcut diagnostic, not a general MP4 classifier.",
    }
    config.metadata_only_metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "payload": payload}


def make_bias_report(context: dict, fusion_videos: pd.DataFrame) -> pd.DataFrame:
    config: Config = context["config"]
    metadata = context["validation_metadata"][["video_id", *METADATA_FIELDS]].copy()
    report = fusion_videos.merge(metadata, on="video_id", how="left", validate="one_to_one")
    rows = []
    for field in METADATA_FIELDS:
        for value, group in report.groupby(field, dropna=False):
            rows.append({
                "metadata_field": field,
                "metadata_value": value,
                "videos": len(group),
                "accidents": int(group.video_label.sum()),
                "false_positive": int(((group.video_label == 0) & (group.prediction == 1)).sum()),
                "false_negative": int(((group.video_label == 1) & (group.prediction == 0)).sum()),
                "error_rate": float((group.video_label != group.prediction).mean()),
                "descriptive_only_small_group": bool(len(group) < 10),
            })
    result = pd.DataFrame(rows).sort_values(["metadata_field", "videos"], ascending=[True, False]).reset_index(drop=True)
    result.to_csv(config.bias_report_path, index=False)
    return result


def run_experiment() -> dict:
    config = Config()
    context = build_context(config)
    video_features = load_video_features(context)
    _, metadata_by_video = encode_metadata(context["metadata"], context["vocabularies"])
    metadata_only = run_metadata_only(context)
    video_training = train_window_head(context, video_features, metadata_by_video, "video_only")
    fusion_training = train_window_head(context, video_features, metadata_by_video, "fusion")
    history = pd.DataFrame(video_training["history"] + fusion_training["history"])
    history.to_csv(config.history_path, index=False)
    video_only = evaluate_window_head(context, video_features, metadata_by_video, "video_only")
    fusion = evaluate_window_head(context, video_features, metadata_by_video, "fusion")
    bias_report = make_bias_report(context, fusion["videos"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "development_split": "fixed_480_train_120_validation",
        "primary_aggregation_frozen": config.primary_aggregation,
        "metadata_only": metadata_only["metrics"],
        "video_only_control": video_only["metrics"],
        "video_metadata_fusion": fusion["metrics"],
        "video_only_best_epoch": video_only["best_epoch"],
        "fusion_best_epoch": fusion["best_epoch"],
        "metadata_fields": list(METADATA_FIELDS),
        "metadata_dim": context["metadata_dim"],
        "metadata_dropout": config.metadata_dropout,
        "reused_feature_sequences": len(video_features),
        "mp4_decoding_required": False,
        "not_final_cv_result": True,
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = pd.read_csv(config.registry_path)
    row = {
        "run_id": "V3_08_D2_METADATA_FUSION",
        "stage": "V3-8 metadata_fusion",
        "model_id": "D2 frozen RGB+motion + weather/light/scene fusion",
        "dataset_version": "v3_sliding_core_context_v1",
        "split_version": "fixed_480_120_development",
        "window_version": "5s_stride2.5s",
        "feature_version": "a2mp_hn1_rgb_and_d1_motion_features_v1",
        "augmentation_version": "metadata_dropout_0.20",
        "checkpoint_path": str(config.fusion_model_path),
        "config_path": "notebooks/38_v3_d2_metadata_fusion.ipynb",
        "git_commit": "not_available",
        "status": "completed",
        "primary_metric": "validation_f1_top3_mean_recall_floor",
        "primary_value": fusion["metrics"]["f1"],
        "notes": "Dataset-only benchmark; metadata is not required by the default arbitrary-MP4 video-only inference pipeline.",
    }
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"context": context, "preflight": cache_preflight(context), "metadata_only": metadata_only, "video_only": video_only, "fusion": fusion, "bias_report": bias_report, "summary": summary}


def context_report(context: dict) -> dict:
    return {
        "device": str(context["device"]),
        "run_signature": context["run_signature"],
        "train_windows": len(context["train"]),
        "validation_windows": len(context["validation"]),
        "validation_videos": context["validation"].video_id.nunique(),
        "metadata_fields": list(METADATA_FIELDS),
        "metadata_dim": context["metadata_dim"],
        "rgb_motion_feature_sequences": len(context["expected"]),
        "mp4_decoding_required": False,
    }
