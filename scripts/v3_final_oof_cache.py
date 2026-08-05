"""Resumable RGB and motion feature cache for final five-fold OOF evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from v3_a2mp_hn1 import (
    Config as EncoderConfig,
    atomic_torch_save,
    build_encoder,
    decode_rgb_at_timestamp,
    normalized_tensor,
    resize_letterbox_rgb,
    sha256_file,
)
from v3_d1_motion import decode_motion_sequence


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    num_frames: int = 16
    feature_dim: int = 512
    sequence_batch_size: int = 2
    frame_batch_size: int = 32
    save_every_batches: int = 25
    preprocessing_version: str = "v2_multipos_rgb_letterbox_replicate_224x320"
    motion_version: str = "absolute_rgb_frame_difference_uint8_before_imagenet_normalization_v1"

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests_v3"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed_v3"

    @property
    def report_root(self) -> Path:
        return self.data_root / "reports_v3"

    @property
    def sequence_manifest_path(self) -> Path:
        return self.manifest_root / "sequence_manifest_v3_sliding.csv"

    @property
    def folds_path(self) -> Path:
        return self.manifest_root / "cv_folds_v3.csv"

    @property
    def union_path(self) -> Path:
        return self.manifest_root / "final_oof_union_sequences.csv"

    @property
    def source_rgb_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def source_motion_path(self) -> Path:
        return self.processed_root / "d1_motion_features.pt"

    @property
    def rgb_partial_path(self) -> Path:
        return self.processed_root / "final_oof_rgb_features_partial.pt"

    @property
    def rgb_features_path(self) -> Path:
        return self.processed_root / "final_oof_rgb_features.pt"

    @property
    def motion_partial_path(self) -> Path:
        return self.processed_root / "final_oof_motion_features_partial.pt"

    @property
    def motion_features_path(self) -> Path:
        return self.processed_root / "final_oof_motion_features.pt"

    @property
    def rgb_failures_path(self) -> Path:
        return self.report_root / "final_oof_rgb_decode_failures.csv"

    @property
    def motion_failures_path(self) -> Path:
        return self.report_root / "final_oof_motion_decode_failures.csv"

    @property
    def cache_summary_path(self) -> Path:
        return self.report_root / "final_oof_feature_cache_summary.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_context(config: Config = Config()) -> dict:
    for directory in [config.manifest_root, config.processed_root, config.report_root]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in [config.sequence_manifest_path, config.folds_path, config.source_rgb_path, config.source_motion_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing final OOF cache prerequisite: {path}")
    sequence = pd.read_csv(config.sequence_manifest_path).copy()
    folds = pd.read_csv(config.folds_path).copy()
    for frame in [sequence, folds]:
        frame["video_id"] = frame["video_id"].astype(str)
    sequence["sequence_id"] = sequence["sequence_id"].astype(str)
    if len(sequence) != 8907 or sequence.sequence_id.duplicated().any() or sequence.video_id.nunique() != 600:
        raise RuntimeError("Final OOF requires exactly the frozen 8,907 sequence / 600 video manifest")
    if len(folds) != 600 or folds.video_id.duplicated().any() or set(folds.outer_fold.unique()) != {0, 1, 2, 3, 4}:
        raise RuntimeError("Final OOF fold manifest is invalid")
    fold_map = folds[["video_id", "outer_fold", "label"]].rename(columns={"label": "fold_label"})
    if "outer_fold" in sequence.columns:
        sequence_fold_map = sequence[["video_id", "outer_fold"]].drop_duplicates("video_id")
        checked_folds = sequence_fold_map.merge(fold_map[["video_id", "outer_fold"]], on="video_id", suffixes=("_sequence", "_frozen"), validate="one_to_one")
        if not checked_folds.outer_fold_sequence.astype(int).eq(checked_folds.outer_fold_frozen.astype(int)).all():
            raise RuntimeError("Sequence manifest fold assignment disagrees with frozen CV folds")
        sequence = sequence.drop(columns="outer_fold")
    union = sequence.merge(fold_map, on="video_id", how="left", validate="many_to_one")
    if union.outer_fold.isna().any() or not union.video_label.astype(int).eq(union.fold_label.astype(int)).all():
        raise RuntimeError("Sequence labels/folds are incompatible")
    union = union.drop(columns="fold_label").sort_values(["outer_fold", "video_id", "window_index"]).reset_index(drop=True)
    union.to_csv(config.union_path, index=False)
    signature = {
        "union_manifest_sha256": sha256_file(config.union_path),
        "num_frames": config.num_frames,
        "feature_dim": config.feature_dim,
        "preprocessing_version": config.preprocessing_version,
        "motion_version": config.motion_version,
    }
    return {"config": config, "union": union, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "signature": signature, "run_signature": sha256_text(json.dumps(signature, sort_keys=True))}


def decode_rgb_sequence(row: pd.Series, config: Config):
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
            previous = rgb
            frames.append(resize_letterbox_rgb(rgb, EncoderConfig()))
            statuses.append(status)
    finally:
        cap.release()
    return torch.stack([normalized_tensor(frame) for frame in frames]), {"sequence_id": row.sequence_id, "decode_status": ";".join(sorted(set(statuses))), "valid_frames": config.num_frames}


def _source_features(path: Path, required_ids: set[str], config: Config, kind: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "rgb" and payload.get("preprocessing_version") != config.preprocessing_version:
        raise RuntimeError("Existing RGB cache has an incompatible preprocessing version")
    sequence_ids = [str(item) for item in payload.get("sequence_ids", [])]
    features = payload.get("features")
    if len(sequence_ids) != len(features):
        raise RuntimeError(f"Existing {kind} cache has mismatched IDs/features")
    return {
        sequence_id: feature.float().cpu()
        for sequence_id, feature in zip(sequence_ids, features)
        if sequence_id in required_ids and tuple(feature.shape) == (config.num_frames, config.feature_dim)
    }


def _load_partial(path: Path, context: dict, kind: str) -> dict[str, torch.Tensor]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_signature") != context["run_signature"] or payload.get("feature_kind") != kind:
        return {}
    config: Config = context["config"]
    return {
        str(sequence_id): feature.float().cpu()
        for sequence_id, feature in payload.get("features_by_sequence", {}).items()
        if tuple(feature.shape) == (config.num_frames, config.feature_dim)
    }


def cache_status(context: dict) -> dict:
    config: Config = context["config"]
    required_ids = context["union"].sequence_id.tolist()
    required_set = set(required_ids)
    rgb_source = _source_features(config.source_rgb_path, required_set, config, "rgb")
    motion_source = _source_features(config.source_motion_path, required_set, config, "motion")
    rgb_partial = _load_partial(config.rgb_partial_path, context, "rgb")
    motion_partial = _load_partial(config.motion_partial_path, context, "motion")
    rgb_available = set(rgb_source) | (set(rgb_partial) & required_set)
    motion_available = set(motion_source) | (set(motion_partial) & required_set)
    return {
        "union_sequences": len(required_ids),
        "existing_rgb_reusable": len(rgb_source),
        "existing_motion_reusable": len(motion_source),
        "rgb_partial_resumable": len(set(rgb_partial) & required_set),
        "motion_partial_resumable": len(set(motion_partial) & required_set),
        "rgb_missing": len(required_set - rgb_available),
        "motion_missing": len(required_set - motion_available),
    }


def cache_preflight(context: dict) -> dict:
    config: Config = context["config"]
    status = cache_status(context)
    output = {"device": str(context["device"]), **status, "rgb": None, "motion": None}
    encoder = build_encoder(EncoderConfig(), context["device"])
    for kind in ["rgb", "motion"]:
        source_path = config.source_rgb_path if kind == "rgb" else config.source_motion_path
        partial_path = config.rgb_partial_path if kind == "rgb" else config.motion_partial_path
        required_ids = set(context["union"].sequence_id)
        available = set(_source_features(source_path, required_ids, config, kind)) | set(_load_partial(partial_path, context, kind))
        candidates = context["union"].loc[~context["union"].sequence_id.isin(available)].head(2)
        if candidates.empty:
            output[kind] = {"missing": 0, "estimated_minutes": 0.0, "feature_shape": [config.num_frames, config.feature_dim]}
            continue
        started = time.perf_counter()
        tensors, details = [], []
        for _, row in candidates.iterrows():
            tensor, detail = (decode_rgb_sequence(row, config) if kind == "rgb" else decode_motion_sequence(row, config))
            if tensor is None:
                raise RuntimeError(f"{kind} preflight decode failure: {detail}")
            tensors.append(tensor)
            details.append(detail)
        with torch.inference_mode():
            features = encoder(torch.cat(tensors).to(context["device"])).flatten(1).cpu()
        elapsed = time.perf_counter() - started
        if tuple(features.shape) != (len(candidates) * config.num_frames, config.feature_dim):
            raise RuntimeError(f"{kind} preflight feature shape is invalid")
        output[kind] = {
            "missing": int(status[f"{kind}_missing"]),
            "preflight_sequences": len(candidates),
            "preflight_seconds": elapsed,
            "estimated_minutes": elapsed * status[f"{kind}_missing"] / len(candidates) / 60.0,
            "feature_shape": [config.num_frames, config.feature_dim],
            "decode_statuses": [detail.get("decode_status", "exact") for detail in details],
        }
    return output


def ensure_feature_cache(context: dict, kind: str, run_full: bool, max_sequences: int | None = None) -> dict:
    if kind not in {"rgb", "motion"}:
        raise ValueError("kind must be rgb or motion")
    config: Config = context["config"]
    source_path = config.source_rgb_path if kind == "rgb" else config.source_motion_path
    partial_path = config.rgb_partial_path if kind == "rgb" else config.motion_partial_path
    complete_path = config.rgb_features_path if kind == "rgb" else config.motion_features_path
    failures_path = config.rgb_failures_path if kind == "rgb" else config.motion_failures_path
    required_ids = context["union"].sequence_id.tolist()
    required_set = set(required_ids)
    features = {key: value for key, value in _load_partial(partial_path, context, kind).items() if key in required_set}
    reused = _source_features(source_path, required_set, config, kind)
    for sequence_id, feature in reused.items():
        features.setdefault(sequence_id, feature)
    missing = context["union"].loc[~context["union"].sequence_id.isin(features)].copy()
    if max_sequences is not None:
        missing = missing.head(int(max_sequences))
    failures = []
    if run_full and len(missing):
        encoder = build_encoder(EncoderConfig(), context["device"])
        batches_since_save = 0
        with torch.inference_mode():
            for start in tqdm(range(0, len(missing), config.sequence_batch_size), desc=f"Final OOF {kind} features"):
                batch_rows = [row for _, row in missing.iloc[start:start + config.sequence_batch_size].iterrows()]
                tensors, valid_rows = [], []
                for row in batch_rows:
                    tensor, detail = (decode_rgb_sequence(row, config) if kind == "rgb" else decode_motion_sequence(row, config))
                    if tensor is None:
                        failures.append(detail)
                    else:
                        tensors.append(tensor)
                        valid_rows.append(row)
                if tensors:
                    image_batch = torch.cat(tensors)
                    chunks = [encoder(image_batch[item:item + config.frame_batch_size].to(context["device"])).flatten(1).cpu() for item in range(0, len(image_batch), config.frame_batch_size)]
                    batch_features = torch.cat(chunks).reshape(len(tensors), config.num_frames, config.feature_dim)
                    for row, feature in zip(valid_rows, batch_features):
                        features[str(row.sequence_id)] = feature
                batches_since_save += 1
                if batches_since_save >= config.save_every_batches:
                    atomic_torch_save({"run_signature": context["run_signature"], "feature_kind": kind, "features_by_sequence": features}, partial_path)
                    batches_since_save = 0
        atomic_torch_save({"run_signature": context["run_signature"], "feature_kind": kind, "features_by_sequence": features}, partial_path)
    pd.DataFrame(failures, columns=["sequence_id", "video_id", "video_path", "window_start", "window_end", "error_reason"]).to_csv(failures_path, index=False)
    missing_after = [sequence_id for sequence_id in required_ids if sequence_id not in features]
    complete = not missing_after
    if complete:
        atomic_torch_save({"run_signature": context["run_signature"], "feature_kind": kind, "sequence_ids": required_ids, "features": torch.stack([features[sequence_id] for sequence_id in required_ids])}, complete_path)
    return {"kind": kind, "features_available": len(features), "features_expected": len(required_ids), "existing_reused": len(reused), "new_sequences_requested_this_run": len(missing), "missing_after_run": len(missing_after), "failures_this_run": len(failures), "complete": complete, "partial_path": str(partial_path), "complete_path": str(complete_path)}


def write_summary(context: dict, rgb: dict, motion: dict, preflight: dict | None = None) -> dict:
    config: Config = context["config"]
    summary = {"run_signature": context["run_signature"], "union_manifest": str(config.union_path), "preflight": preflight, "rgb": rgb, "motion": motion, "cache_complete": bool(rgb["complete"] and motion["complete"])}
    config.cache_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def context_report(context: dict) -> dict:
    return {"device": str(context["device"]), "run_signature": context["run_signature"], "union_sequences": len(context["union"]), "videos": context["union"].video_id.nunique(), "outer_fold_counts": context["union"].groupby("outer_fold").video_id.nunique().to_dict(), **cache_status(context)}
