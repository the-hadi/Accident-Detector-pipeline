"""Train the selected D1 recipe on all 600 videos and save a deployable artifact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
from statistics import median

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from v3_a2mp_hn1 import MeanMaxHead, round_robin_negative_selection
from v3_final_oof_cv import (
    CachedSequenceDataset,
    Config as CVConfig,
    assign_loss_weights,
    build_context as build_cv_context,
    forward_head,
    score_rows,
    seed_everything,
    select_hard_negatives,
)


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(r"P:\NexarCollisionData")

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests_v3"

    @property
    def model_root(self) -> Path:
        return self.data_root / "models_v3"

    @property
    def report_root(self) -> Path:
        return self.data_root / "reports_v3"

    @property
    def final_oof_summary_path(self) -> Path:
        return self.report_root / "final_oof_summary.json"

    @property
    def train_windows_path(self) -> Path:
        return self.manifest_root / "final_d1_all600_train_windows.csv"

    @property
    def hard_negatives_path(self) -> Path:
        return self.manifest_root / "final_d1_all600_hard_negatives.csv"

    @property
    def miner_path(self) -> Path:
        return self.model_root / "final_d1_all600_hn_miner.pt"

    @property
    def model_path(self) -> Path:
        return self.model_root / "final_d1_all600.pt"

    @property
    def history_path(self) -> Path:
        return self.model_root / "final_d1_all600_training_history.csv"

    @property
    def recipe_path(self) -> Path:
        return self.report_root / "final_d1_inference_recipe.json"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "final_d1_handoff_summary.json"


def selected_epochs(summary: dict) -> tuple[int, int]:
    miner_epochs = [int(item["miner"]["best_epoch"]) for item in summary["fold_reports"]]
    d1_epochs = [int(item["models"]["d1"]["best_epoch"]) for item in summary["fold_reports"]]
    return int(median(miner_epochs)), int(median(d1_epochs))


def build_all_data_rows(context: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config: CVConfig = context["config"]
    union = context["union"]
    positives = union.loc[union.video_label.eq(1) & union.window_role.eq("positive_core") & union.hard_label.eq(1)].copy()
    negative_pool = union.loc[union.video_label.eq(0) & union.window_role.eq("negative_video") & union.hard_label.eq(0)].copy()
    if len(positives) != 757 or len(negative_pool) != 4363:
        raise RuntimeError("Unexpected all-data positive-core or negative-window count")
    preliminary_normal = round_robin_negative_selection(negative_pool, len(positives), config.seed)
    preliminary = assign_loss_weights(positives, preliminary_normal, negative_pool.head(0), config)
    return positives, negative_pool, preliminary


def train_fixed_epochs(context: dict, kind: str, rows: pd.DataFrame, epochs: int, output_path: Path, history_records: list[dict]) -> torch.nn.Module:
    config: CVConfig = context["config"]
    seed_offset = 1000 if kind == "hn_miner" else 2000
    seed_everything(config.seed + seed_offset)
    dataset = CachedSequenceDataset(rows, context["rgb"], context["motion"])
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed + seed_offset), num_workers=0)
    model = MeanMaxHead(config.feature_dim) if kind == "hn_miner" else __import__("v3_d1_motion", fromlist=["RGBMotionFusionHead"]).RGBMotionFusionHead(config.feature_dim)
    model = model.to(context["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
        record = {"model": kind, "epoch": epoch, "weighted_train_loss": weighted_loss / total_weight, "selection": "fixed_epoch_median_from_final_oof"}
        history_records.append(record)
        print(record, flush=True)
    torch.save({"model_state_dict": model.state_dict(), "kind": kind, "epochs": epochs, "run_signature": context["run_signature"]}, output_path)
    return model


def run_final_handoff() -> dict:
    config = Config()
    for directory in [config.manifest_root, config.model_root, config.report_root]:
        directory.mkdir(parents=True, exist_ok=True)
    if not config.final_oof_summary_path.is_file():
        raise FileNotFoundError("Final OOF summary is required before all-data retraining")
    oof_summary = json.loads(config.final_oof_summary_path.read_text(encoding="utf-8"))
    if oof_summary.get("selected_model") != "D1":
        raise RuntimeError("This handoff is specifically for the OOF-selected D1 model")
    miner_epochs, d1_epochs = selected_epochs(oof_summary)
    context = build_cv_context(CVConfig())
    positives, negative_pool, preliminary_rows = build_all_data_rows(context)
    history: list[dict] = []
    miner = train_fixed_epochs(context, "hn_miner", preliminary_rows, miner_epochs, config.miner_path, history)
    scored_negatives = score_rows(miner, "hn_miner", negative_pool, context)
    hard_negatives = select_hard_negatives(scored_negatives, context["config"])
    hard_negatives.to_csv(config.hard_negatives_path, index=False)
    normal_pool = negative_pool.loc[~negative_pool.sequence_id.isin(set(hard_negatives.sequence_id))].copy()
    normal_negatives = round_robin_negative_selection(normal_pool, len(positives), context["config"].seed)
    final_rows = assign_loss_weights(positives, normal_negatives, hard_negatives, context["config"])
    if final_rows.video_id.nunique() != 600 or len(final_rows) != len(positives) * 2 + len(hard_negatives):
        raise RuntimeError("Final D1 training rows do not cover the expected all-data scope")
    final_rows.to_csv(config.train_windows_path, index=False)
    d1 = train_fixed_epochs(context, "d1", final_rows, d1_epochs, config.model_path, history)
    pd.DataFrame(history).to_csv(config.history_path, index=False)
    checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=False)
    threshold = float(oof_summary["selected_thresholds"]["d1"])
    recipe = {
        "model_id": "final_d1_all600",
        "selected_from": "five_fold_outer_oof_full_mp4",
        "selected_model": "D1",
        "oof_threshold": threshold,
        "aggregation": "top3_mean",
        "window_seconds": 5.0,
        "stride_seconds": 2.5,
        "num_frames": 16,
        "preprocessing": "RGB letterbox replicated-edge 224x320, ImageNet normalization",
        "motion": "absolute RGB difference between adjacent sampled frames before normalization",
        "miner_epochs_median": miner_epochs,
        "d1_epochs_median": d1_epochs,
        "training_videos": 600,
        "training_rows": len(final_rows),
        "hard_negatives": len(hard_negatives),
        "model_path": str(config.model_path),
        "oof_metrics": next(row for row in oof_summary["primary_comparison"] if row["model"] == "d1"),
    }
    config.recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_path": str(config.model_path),
        "checkpoint_kind": checkpoint["kind"],
        "recipe_path": str(config.recipe_path),
        "train_windows_path": str(config.train_windows_path),
        "hard_negatives_path": str(config.hard_negatives_path),
        "training_videos": int(final_rows.video_id.nunique()),
        "training_rows": len(final_rows),
        "hard_negatives": len(hard_negatives),
        "fixed_epochs_from_oof_median": {"hn_miner": miner_epochs, "d1": d1_epochs},
        "oof_selected_threshold": threshold,
        "oof_metrics": recipe["oof_metrics"],
        "note": "All-data training creates a deployable artifact; it is not a new independent accuracy estimate.",
    }
    config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
