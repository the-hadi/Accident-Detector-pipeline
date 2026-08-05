"""Reusable arbitrary-MP4 inference for the final all-data D1 artifact."""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from v3_a2mp_hn1 import Config as EncoderConfig, build_encoder, decode_rgb_at_timestamp, normalized_tensor, resize_letterbox_rgb
from v3_d1_motion import RGBMotionFusionHead


DEFAULT_DATA_ROOT = Path(r"P:\NexarCollisionData")
DEFAULT_MODEL_PATH = DEFAULT_DATA_ROOT / "models_v3" / "final_d1_all600.pt"
DEFAULT_RECIPE_PATH = DEFAULT_DATA_ROOT / "reports_v3" / "final_d1_inference_recipe.json"


def window_starts(duration: float, window_seconds: float, stride_seconds: float) -> list[float]:
    maximum = duration - window_seconds
    if maximum < -1e-9:
        raise ValueError(f"Video must be at least {window_seconds:.1f} seconds long.")
    starts = list(np.arange(0.0, maximum + 1e-9, stride_seconds, dtype=float))
    if not starts or maximum - starts[-1] > 1e-6:
        starts.append(float(maximum))
    return [float(value) for value in starts]


def load_final_d1(model_path: str | Path = DEFAULT_MODEL_PATH, recipe_path: str | Path = DEFAULT_RECIPE_PATH, device: str | None = None) -> dict:
    model_path, recipe_path = Path(model_path), Path(recipe_path)
    if not model_path.is_file() or not recipe_path.is_file():
        raise FileNotFoundError("Final D1 model or inference recipe is missing; run the final handoff first.")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location=selected_device, weights_only=False)
    if checkpoint.get("kind") != "d1":
        raise RuntimeError("The supplied checkpoint is not the final D1 artifact")
    encoder = build_encoder(EncoderConfig(), selected_device)
    head = RGBMotionFusionHead(512).to(selected_device)
    head.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    head.eval()
    return {"device": selected_device, "encoder": encoder, "head": head, "recipe": recipe, "model_path": str(model_path)}


@torch.inference_mode()
def predict_full_mp4(video_path: str | Path, predictor: dict | None = None) -> dict:
    """Classify any valid complete MP4; label, event time, and metadata are not required."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"MP4 file does not exist: {video_path}")
    predictor = predictor or load_final_d1()
    recipe = predictor["recipe"]
    window_seconds = float(recipe["window_seconds"])
    stride_seconds = float(recipe["stride_seconds"])
    num_frames = int(recipe["num_frames"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open MP4: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else float("nan")
    if not np.isfinite(duration):
        cap.release()
        raise ValueError("Video has invalid FPS or frame count")
    starts = window_starts(float(duration), window_seconds, stride_seconds)
    records, started = [], time.perf_counter()
    try:
        for start in starts:
            rgb_images, statuses, previous = [], [], None
            for timestamp in np.linspace(start, start + window_seconds, num=num_frames, endpoint=False):
                rgb, status = decode_rgb_at_timestamp(cap, float(timestamp), fps)
                if rgb is None and previous is not None:
                    rgb, status = previous.copy(), "repeated_previous_after_decode_failure"
                if rgb is None:
                    raise RuntimeError(f"Could not decode frame near {timestamp:.3f}s in window starting {start:.3f}s")
                previous = rgb
                rgb_images.append(resize_letterbox_rgb(rgb, EncoderConfig()))
                statuses.append(status)
            motion_images = [np.zeros_like(rgb_images[0])]
            motion_images.extend([cv2.absdiff(current, prior) for prior, current in zip(rgb_images[:-1], rgb_images[1:])])
            rgb_tensor = torch.stack([normalized_tensor(image) for image in rgb_images])
            motion_tensor = torch.stack([normalized_tensor(image) for image in motion_images])
            encoded = predictor["encoder"](torch.cat([rgb_tensor, motion_tensor]).to(predictor["device"])).flatten(1)
            probability = float(torch.sigmoid(predictor["head"](encoded[:num_frames].unsqueeze(0), encoded[num_frames:].unsqueeze(0))).item())
            records.append({"window_start": float(start), "window_end": float(start + window_seconds), "positive_probability": probability, "decode_status": ";".join(sorted(set(statuses)))})
    finally:
        cap.release()
    probabilities = np.asarray([record["positive_probability"] for record in records], dtype=float)
    top_count = min(3, len(probabilities))
    video_probability = float(np.sort(probabilities)[-top_count:].mean())
    best = records[int(np.argmax(probabilities))]
    threshold = float(recipe["oof_threshold"])
    return {
        "video_path": str(video_path),
        "duration_seconds": float(duration),
        "num_windows": len(records),
        "video_probability": video_probability,
        "prediction": int(video_probability >= threshold),
        "prediction_label": "accident" if video_probability >= threshold else "no_accident",
        "threshold": threshold,
        "aggregation": recipe["aggregation"],
        "highest_probability_window": [best["window_start"], best["window_end"]],
        "inference_seconds": time.perf_counter() - started,
        "window_predictions": records,
    }
