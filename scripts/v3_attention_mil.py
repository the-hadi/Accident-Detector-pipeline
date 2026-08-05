"""V3-4B gated attention MIL over frozen ResNet18 video-window features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random
import time

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
    atomic_torch_save,
    build_encoder,
    decode_sequence,
    metrics_at_threshold,
    select_threshold,
    sha256_file,
)


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")
    seed: int = 42
    num_frames: int = 16
    feature_dim: int = 512
    max_train_windows: int = 8
    max_special_windows: int = 3
    attention_dim: int = 128
    dropout: float = 0.30
    sequence_batch_size: int = 2
    frame_batch_size: int = 32
    save_every_batches: int = 25
    batch_size_bags: int = 16
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    recall_floor: float = 0.85
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
    def bag_manifest_path(self) -> Path:
        return self.manifest_root / "bag_manifest_v3_mil.csv"

    @property
    def hard_negatives_path(self) -> Path:
        return self.manifest_root / "hard_negatives_round1.csv"

    @property
    def selected_bags_path(self) -> Path:
        return self.manifest_root / "amil_bag_manifest_v3.csv"

    @property
    def hn_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn_train_negative_features.pt"

    @property
    def a2mp_hn1_features_path(self) -> Path:
        return self.processed_root / "a2mp_hn1_features.pt"

    @property
    def partial_features_path(self) -> Path:
        return self.processed_root / "amil_features_partial.pt"

    @property
    def features_path(self) -> Path:
        return self.processed_root / "amil_features.pt"

    @property
    def failures_path(self) -> Path:
        return self.report_root / "amil_feature_decode_failures.csv"

    @property
    def best_model_path(self) -> Path:
        return self.model_root / "amil_frozen_resnet18_best.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "amil_training_history.csv"

    @property
    def video_predictions_path(self) -> Path:
        return self.prediction_root / "amil_validation_video_predictions.csv"

    @property
    def attention_weights_path(self) -> Path:
        return self.prediction_root / "amil_validation_attention_weights.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "amil_summary.json"

    @property
    def registry_path(self) -> Path:
        return self.report_root / "experiments_v3_registry.csv"


class GatedAttentionMIL(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_v = nn.Linear(feature_dim, attention_dim)
        self.attention_u = nn.Linear(feature_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, bag_features: torch.Tensor, mask: torch.Tensor):
        gated = torch.tanh(self.attention_v(bag_features)) * torch.sigmoid(self.attention_u(bag_features))
        raw_attention = self.attention_w(gated).squeeze(-1)
        raw_attention = raw_attention.masked_fill(~mask, -torch.inf)
        attention = torch.softmax(raw_attention, dim=1)
        pooled = torch.bmm(attention.unsqueeze(1), bag_features).squeeze(1)
        return self.classifier(pooled).squeeze(1), attention


class BagDataset(Dataset):
    def __init__(self, bag_rows: pd.DataFrame, features_by_sequence: dict[str, torch.Tensor]) -> None:
        self.rows = bag_rows.reset_index(drop=True).copy()
        self.bag_features: list[torch.Tensor] = []
        for ids_json in self.rows["sequence_ids_json"]:
            ids = json.loads(ids_json)
            # Only RGB features enter the model; no timestamps or window labels.
            self.bag_features.append(torch.stack([features_by_sequence[str(sequence_id)].mean(dim=0) for sequence_id in ids]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        return self.bag_features[index], int(row.video_label), str(row.video_id), row.sequence_ids_json


def collate_bags(batch):
    features, labels, video_ids, ids_json = zip(*batch)
    maximum = max(item.shape[0] for item in features)
    feature_dim = features[0].shape[1]
    padded = torch.zeros((len(features), maximum, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(features), maximum), dtype=torch.bool)
    for index, item in enumerate(features):
        padded[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return padded, mask, torch.tensor(labels, dtype=torch.float32), list(video_ids), list(ids_json)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def take_evenly(ids: list[str], limit: int) -> list[str]:
    if len(ids) <= limit:
        return list(ids)
    positions = np.linspace(0, len(ids) - 1, limit).round().astype(int)
    return [ids[int(position)] for position in positions]


def create_selected_bags(config: Config) -> dict:
    for directory in [config.processed_root, config.prediction_root, config.report_root, config.model_root]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in [config.sequence_manifest_path, config.bag_manifest_path, config.hard_negatives_path, config.hn_features_path, config.a2mp_hn1_features_path, config.registry_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")
    sequence = pd.read_csv(config.sequence_manifest_path).copy()
    sequence["sequence_id"] = sequence["sequence_id"].astype(str)
    sequence["video_id"] = sequence["video_id"].astype(str)
    bags = pd.read_csv(config.bag_manifest_path).copy()
    bags["video_id"] = bags["video_id"].astype(str)
    hard = pd.read_csv(config.hard_negatives_path).copy()
    hard["video_id"] = hard["video_id"].astype(str)
    hard_rank = dict(zip(hard["sequence_id"].astype(str), hard["rank_in_video"].astype(int)))
    core_ids = set(sequence.loc[(sequence["split"].eq("train")) & (sequence["window_role"].eq("positive_core")), "sequence_id"])

    selected_records = []
    for _, bag in bags.iterrows():
        ids = [str(item) for item in json.loads(bag.sequence_ids_json)]
        if bag.split == "validation":
            selected = ids
            special_core, special_hn, policy = 0, 0, "validation_all_sliding_windows"
        else:
            if int(bag.video_label) == 1:
                special = take_evenly([item for item in ids if item in core_ids], config.max_special_windows)
                special_core, special_hn = len(special), 0
                policy = "train_core_preserving_plus_uniform_fill"
            else:
                special = sorted([item for item in ids if item in hard_rank], key=lambda item: hard_rank[item])[: config.max_special_windows]
                special_core, special_hn = 0, len(special)
                policy = "train_hard_negative_preserving_plus_uniform_fill"
            remaining = [item for item in ids if item not in set(special)]
            selected = special + take_evenly(remaining, max(0, config.max_train_windows - len(special)))
            # Keep temporal order in each bag; attention is order-agnostic, but provenance is clearer this way.
            selected = [item for item in ids if item in set(selected)]
        selected_records.append({
            **bag.to_dict(),
            "sequence_ids_json": json.dumps(selected),
            "selected_window_count": len(selected),
            "selected_positive_core_count": special_core,
            "selected_hard_negative_count": special_hn,
            "bag_sampling_policy": policy,
            "sampling_seed": config.seed,
        })
    selected_bags = pd.DataFrame(selected_records)
    if len(selected_bags) != 600 or not selected_bags.video_id.is_unique:
        raise RuntimeError("A-MIL must create exactly one bag for each video")
    if not selected_bags.loc[selected_bags.split.eq("validation"), "selected_window_count"].eq(selected_bags.loc[selected_bags.split.eq("validation"), "bag_window_count"]).all():
        raise RuntimeError("validation bags must retain all sliding windows")
    train_positive = selected_bags.loc[(selected_bags.split.eq("train")) & (selected_bags.video_label.eq(1))]
    if not train_positive.selected_positive_core_count.ge(1).all():
        raise RuntimeError("each positive train bag must retain core evidence")
    selected_bags.to_csv(config.selected_bags_path, index=False)
    train_ids = list(dict.fromkeys(item for ids_json in selected_bags.loc[selected_bags.split.eq("train"), "sequence_ids_json"] for item in json.loads(ids_json)))
    validation_ids = list(dict.fromkeys(item for ids_json in selected_bags.loc[selected_bags.split.eq("validation"), "sequence_ids_json"] for item in json.loads(ids_json)))
    sequence_lookup = sequence.set_index("sequence_id", drop=False)
    expected_ids = list(dict.fromkeys(train_ids + validation_ids))
    expected_rows = sequence_lookup.loc[expected_ids].reset_index(drop=True)
    signature_payload = {
        "sequence_manifest_sha256": sha256_file(config.sequence_manifest_path),
        "bag_manifest_sha256": sha256_file(config.bag_manifest_path),
        "hard_negative_sha256": sha256_file(config.hard_negatives_path),
        "hn_features_sha256": sha256_file(config.hn_features_path),
        "a2mp_hn1_features_sha256": sha256_file(config.a2mp_hn1_features_path),
        "preprocessing_version": config.preprocessing_version,
        "seed": config.seed,
        "max_train_windows": config.max_train_windows,
        "max_special_windows": config.max_special_windows,
        "expected_sequence_ids": expected_ids,
    }
    return {
        "config": config,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "sequence": sequence,
        "selected_bags": selected_bags,
        "expected_rows": expected_rows,
        "expected_ids": expected_ids,
        "run_signature": sha256_text(json.dumps(signature_payload, sort_keys=True)),
        "signature_payload": signature_payload,
    }


def source_feature_map(path: Path, required_preprocessing: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("preprocessing_version") != required_preprocessing:
        raise RuntimeError(f"Incompatible preprocessing in {path}")
    return {str(sequence_id): feature.float().cpu() for sequence_id, feature in zip(payload["sequence_ids"], payload["features"])}


def initial_feature_cache(context: dict) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    config: Config = context["config"]
    expected = set(context["expected_ids"])
    features, source = {}, {}
    for name, path in [("reused_v3_3_negative", config.hn_features_path), ("reused_v3_4_a2mp_hn1", config.a2mp_hn1_features_path)]:
        for sequence_id, feature in source_feature_map(path, config.preprocessing_version).items():
            if sequence_id in expected and sequence_id not in features:
                if tuple(feature.shape) != (config.num_frames, config.feature_dim):
                    raise RuntimeError("incompatible frozen feature shape")
                features[sequence_id] = feature
                source[sequence_id] = name
    return features, source


def load_partial(context: dict) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    config: Config = context["config"]
    if not config.partial_features_path.is_file():
        return {}, {}
    payload = torch.load(config.partial_features_path, map_location="cpu", weights_only=False)
    if payload.get("run_signature") != context["run_signature"] or payload.get("preprocessing_version") != config.preprocessing_version:
        return {}, {}
    features = {str(key): value.float().cpu() for key, value in payload.get("features_by_sequence", {}).items() if tuple(value.shape) == (config.num_frames, config.feature_dim)}
    sources = {str(key): str(value) for key, value in payload.get("feature_source_by_sequence", {}).items() if str(key) in features}
    return features, sources


def cache_preflight(context: dict) -> dict:
    config: Config = context["config"]
    seeded, _ = initial_feature_cache(context)
    candidates = context["expected_rows"].loc[~context["expected_rows"].sequence_id.isin(seeded)].head(2)
    if len(candidates) < 2:
        candidates = context["expected_rows"].head(2)
    encoder_config = EncoderConfig()
    encoder = build_encoder(encoder_config, context["device"])
    tensors, statuses = [], []
    start = time.perf_counter()
    for _, row in candidates.iterrows():
        tensor, detail = decode_sequence(row, encoder_config)
        if tensor is None:
            raise RuntimeError(detail)
        tensors.append(tensor)
        statuses.append(detail["decode_status"])
    with torch.inference_mode():
        features = encoder(torch.cat(tensors).to(context["device"])).flatten(1).cpu()
    elapsed = time.perf_counter() - start
    if tuple(features.shape) != (len(candidates) * config.num_frames, config.feature_dim):
        raise RuntimeError("A-MIL preflight feature shape is invalid")
    return {
        "preflight_rows": len(candidates),
        "preflight_seconds": elapsed,
        "expected_sequences": len(context["expected_ids"]),
        "reused_sequences_before_decode": len(seeded),
        "new_sequences_to_decode": len(context["expected_ids"]) - len(seeded),
        "estimated_minutes_for_new_sequences": elapsed * (len(context["expected_ids"]) - len(seeded)) / len(candidates) / 60.0,
        "decode_statuses": statuses,
    }


def ensure_feature_cache(context: dict, run_full_cache: bool, max_sequences: int | None = None) -> dict:
    config: Config = context["config"]
    expected_set = set(context["expected_ids"])
    features, sources = load_partial(context)
    features = {key: value for key, value in features.items() if key in expected_set}
    sources = {key: value for key, value in sources.items() if key in features}
    seeded, seeded_sources = initial_feature_cache(context)
    for sequence_id, feature in seeded.items():
        if sequence_id not in features:
            features[sequence_id] = feature
            sources[sequence_id] = seeded_sources[sequence_id]
    missing = context["expected_rows"].loc[~context["expected_rows"].sequence_id.isin(features)].copy()
    if max_sequences is not None:
        missing = missing.head(int(max_sequences))
    failures = []
    if run_full_cache and len(missing):
        encoder_config = EncoderConfig()
        encoder = build_encoder(encoder_config, context["device"])
        batches_since_save = 0
        with torch.inference_mode():
            for batch_start in tqdm(range(0, len(missing), config.sequence_batch_size), desc="A-MIL frozen features"):
                rows = [row for _, row in missing.iloc[batch_start:batch_start + config.sequence_batch_size].iterrows()]
                tensors, valid_rows = [], []
                for row in rows:
                    tensor, detail = decode_sequence(row, encoder_config)
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
                        sources[str(row.sequence_id)] = "new_amil_decode"
                batches_since_save += 1
                if batches_since_save >= config.save_every_batches:
                    atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "features_by_sequence": features, "feature_source_by_sequence": sources}, config.partial_features_path)
                    batches_since_save = 0
        atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "features_by_sequence": features, "feature_source_by_sequence": sources}, config.partial_features_path)
    failure_table = pd.DataFrame(failures, columns=["sequence_id", "video_id", "video_path", "window_start", "window_end", "error_reason"])
    failure_table.to_csv(config.failures_path, index=False)
    missing_after = [item for item in context["expected_ids"] if item not in features]
    complete = not missing_after
    if complete:
        atomic_torch_save({"run_signature": context["run_signature"], "preprocessing_version": config.preprocessing_version, "sequence_ids": context["expected_ids"], "features": torch.stack([features[item] for item in context["expected_ids"]]), "feature_source_by_sequence": sources}, config.features_path)
    return {"features_by_sequence": features, "feature_source_by_sequence": sources, "features_available": len(features), "features_expected": len(context["expected_ids"]), "missing_after_run": len(missing_after), "failures_this_run": len(failure_table), "complete": complete, "new_decode_rows_this_run": len(missing)}


def validation_predictions(model: nn.Module, bag_rows: pd.DataFrame, features_by_sequence: dict[str, torch.Tensor], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = BagDataset(bag_rows, features_by_sequence)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_bags)
    video_records, attention_records = [], []
    model.eval()
    with torch.inference_mode():
        for features, mask, labels, video_ids, ids_json in loader:
            logits, attention = model(features.to(device), mask.to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            attention = attention.cpu().numpy()
            for batch_index, (video_id, ids_encoded) in enumerate(zip(video_ids, ids_json)):
                ids = json.loads(ids_encoded)
                label = int(labels[batch_index].item())
                video_records.append({"video_id": video_id, "video_label": label, "bag_window_count": len(ids), "video_logit": float(logits[batch_index].cpu()), "video_probability": float(probabilities[batch_index])})
                for sequence_id, weight in zip(ids, attention[batch_index, : len(ids)]):
                    attention_records.append({"video_id": video_id, "video_label": label, "sequence_id": sequence_id, "attention_weight": float(weight)})
    return pd.DataFrame(video_records), pd.DataFrame(attention_records)


def train_model(context: dict, cache_result: dict) -> dict:
    if not cache_result["complete"]:
        raise RuntimeError("A-MIL cache is incomplete")
    config: Config = context["config"]
    seed_everything(config.seed)
    train_rows = context["selected_bags"].loc[context["selected_bags"].split.eq("train")].copy()
    validation_rows = context["selected_bags"].loc[context["selected_bags"].split.eq("validation")].copy()
    if train_rows.video_label.value_counts().to_dict() != {1: 240, 0: 240}:
        raise RuntimeError("A-MIL training bags must be balanced at video level")
    dataset = BagDataset(train_rows, cache_result["features_by_sequence"])
    loader = DataLoader(dataset, batch_size=config.batch_size_bags, shuffle=True, generator=torch.Generator().manual_seed(config.seed), num_workers=0, collate_fn=collate_bags)
    model = GatedAttentionMIL(config.feature_dim, config.attention_dim, config.dropout).to(context["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_pr_auc, best_epoch, stale = [], -float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, total_items = 0.0, 0
        for features, mask, labels, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features.to(context["device"]), mask.to(context["device"]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(context["device"]))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            total_items += len(labels)
        validation_videos, _ = validation_predictions(model, validation_rows, cache_result["features_by_sequence"], context["device"])
        pr_auc = float(average_precision_score(validation_videos.video_label, validation_videos.video_probability))
        record = {"epoch": epoch, "train_loss": total_loss / total_items, "validation_pr_auc": pr_auc}
        history.append(record)
        print(record)
        if pr_auc > best_pr_auc + 1e-12:
            best_pr_auc, best_epoch, stale = pr_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_pr_auc": pr_auc, "run_signature": context["run_signature"], "config": config.__dict__}, config.best_model_path)
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break
    pd.DataFrame(history).to_csv(config.history_path, index=False)
    return {"best_epoch": best_epoch, "best_validation_pr_auc": best_pr_auc, "epochs_completed": len(history)}


def evaluate_best(context: dict, cache_result: dict) -> dict:
    config: Config = context["config"]
    if not cache_result["complete"] or not config.best_model_path.is_file():
        raise RuntimeError("A-MIL model or cache is missing")
    checkpoint = torch.load(config.best_model_path, map_location=context["device"], weights_only=False)
    if checkpoint.get("run_signature") != context["run_signature"]:
        raise RuntimeError("A-MIL checkpoint belongs to another bag manifest")
    model = GatedAttentionMIL(config.feature_dim, config.attention_dim, config.dropout).to(context["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_rows = context["selected_bags"].loc[context["selected_bags"].split.eq("validation")].copy()
    videos, attention = validation_predictions(model, validation_rows, cache_result["features_by_sequence"], context["device"])
    selected = select_threshold(videos.video_label.to_numpy(dtype=int), videos.video_probability.to_numpy(dtype=float), config.recall_floor)
    videos["threshold"] = selected["threshold"]
    videos["prediction"] = (videos.video_probability >= selected["threshold"]).astype(int)
    lookup = context["sequence"].set_index("sequence_id")[["window_start", "window_end", "window_role"]]
    attention = attention.join(lookup, on="sequence_id", how="left")
    attention["attention_rank_in_video"] = attention.groupby("video_id")["attention_weight"].rank(method="first", ascending=False).astype(int)
    videos.to_csv(config.video_predictions_path, index=False)
    attention.to_csv(config.attention_weights_path, index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_id": "A-MIL frozen ResNet18 mean-frame features + gated attention",
        "development_split": "fixed_480_train_120_validation",
        "metrics_validation_selected_threshold": selected,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_pr_auc": float(checkpoint["validation_pr_auc"]),
        "run_signature": context["run_signature"],
        "train_bags": 480,
        "validation_bags": 120,
        "train_selected_sequences": int(sum(context["selected_bags"].loc[context["selected_bags"].split.eq("train"), "selected_window_count"])),
        "validation_full_mp4_sequences": int(sum(context["selected_bags"].loc[context["selected_bags"].split.eq("validation"), "selected_window_count"])),
        "video_predictions_path": str(config.video_predictions_path),
        "attention_weights_path": str(config.attention_weights_path),
        "not_final_cv_result": True,
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = pd.read_csv(config.registry_path)
    row = {"run_id": "V3_04B_ATTENTION_MIL", "stage": "V3-4 attention_MIL", "model_id": "A-MIL frozen ResNet18 gated attention", "dataset_version": "v3_sliding_core_context_v1", "split_version": "fixed_480_120_development", "window_version": "5s_stride2.5s", "feature_version": "amil_frozen_features_v1", "augmentation_version": "none", "checkpoint_path": str(config.best_model_path), "config_path": "notebooks/35_v3_attention_mil.ipynb", "git_commit": "not_available", "status": "completed", "primary_metric": "validation_f1_recall_floor", "primary_value": selected["f1"], "notes": "Training uses video labels only; event/hard-negative aware bag construction is documented."}
    registry = registry.loc[~registry.run_id.eq(row["run_id"])]
    pd.concat([registry, pd.DataFrame([row])], ignore_index=True).to_csv(config.registry_path, index=False)
    return {"summary": summary, "videos": videos, "attention": attention}


def context_report(context: dict) -> dict:
    bags = context["selected_bags"]
    train = bags.loc[bags.split.eq("train")]
    validation = bags.loc[bags.split.eq("validation")]
    return {
        "device": str(context["device"]),
        "run_signature": context["run_signature"],
        "train_bags": len(train),
        "train_video_labels": train.video_label.value_counts().to_dict(),
        "train_selected_windows": int(train.selected_window_count.sum()),
        "positive_bags_with_core": int((train.loc[train.video_label.eq(1), "selected_positive_core_count"] >= 1).sum()),
        "negative_bags_with_hard_negative": int((train.loc[train.video_label.eq(0), "selected_hard_negative_count"] >= 1).sum()),
        "validation_bags": len(validation),
        "validation_full_mp4_windows": int(validation.selected_window_count.sum()),
        "selected_bag_manifest": str(context["config"].selected_bags_path),
    }
