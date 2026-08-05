# %% [markdown]
# # استخراج فریم برای مدل اولیهٔ تشخیص تصادف
#
# از هر یک از 600 ویدئوی منتخب، 8 فریم با فاصلهٔ یکنواخت استخراج می‌شود.
# خروجی روی درایو P ذخیره می‌شود و فایل `frame_index.csv` اتصال هر فریم به
# ویدئو و برچسب آن را نگه می‌دارد.

# %%
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

# این اسکریپت باید در ترمینال و VS Code بدون بازکردن پنجرهٔ گرافیکی اجرا شود.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# %% [markdown]
# ## پیکربندی

# %%
DATA_ROOT = Path(r"P:\NexarCollisionData")
SELECTED_VIDEOS_CSV = DATA_ROOT / "selected_train_600.csv"
FRAMES_ROOT = DATA_ROOT / "frames_8_224"
NUM_FRAMES = 8
IMAGE_SIZE = (224, 224)  # width, height
JPEG_QUALITY = 90
RANDOM_SEED = 42

videos = pd.read_csv(SELECTED_VIDEOS_CSV)
videos["label"] = videos["label"].astype(int)
assert len(videos) == 600, "Expected the balanced initial set of 600 videos."
assert videos["local_path"].map(lambda path: Path(path).exists()).all(), "Some selected videos are missing."

print(videos["label"].value_counts().sort_index())
print(f"Frame output: {FRAMES_ROOT}")


# %% [markdown]
# ## توابع کمکی

# %%
def frame_positions(total_frames: int, n_frames: int) -> np.ndarray:
    """اندیس n فریم یکنواخت در سراسر ویدئو."""
    if total_frames < 1:
        raise ValueError("Video has no readable frames.")
    return np.linspace(0, total_frames - 1, num=n_frames, dtype=int)


def read_selected_frames(video_path: Path, n_frames: int = NUM_FRAMES):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    positions = frame_positions(total_frames, n_frames)
    frames = []

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot read frame {position} from {video_path}")
        frames.append(frame)

    capture.release()
    return frames, positions, fps, total_frames


def output_paths(video_path: Path, label: int) -> list[Path]:
    label_name = "positive" if label == 1 else "negative"
    video_dir = FRAMES_ROOT / label_name / video_path.stem
    return [video_dir / f"frame_{index:02d}.jpg" for index in range(NUM_FRAMES)]


# %% [markdown]
# ## پیش‌نمایش: یک ویدئوی منفی و یک ویدئوی مثبت
#
# این تصویر قبل از استخراج کامل ایجاد می‌شود تا ترتیب فریم‌ها و کیفیت داده بررسی شود.

# %%
preview_rows = (
    videos.groupby("label", group_keys=False)
    .sample(n=1, random_state=RANDOM_SEED)
    .sort_values("label")
)

fig, axes = plt.subplots(2, NUM_FRAMES, figsize=(20, 5))
for row_number, (_, row) in enumerate(preview_rows.iterrows()):
    frames, positions, fps, _ = read_selected_frames(Path(row["local_path"]))
    for column, (frame, position) in enumerate(zip(frames, positions)):
        axes[row_number, column].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[row_number, column].set_title(f"frame {position}\n{position / fps:.1f}s")
        axes[row_number, column].axis("off")
    axes[row_number, 0].set_ylabel(
        "positive" if row["label"] == 1 else "negative", rotation=90, size=12
    )

fig.tight_layout()
preview_path = DATA_ROOT / "frame_preview.png"
fig.savefig(preview_path, dpi=150, bbox_inches="tight")
plt.close(fig)  # در اجرای اسکریپت، نمایش پنجره می‌تواند پردازش را متوقف کند.
print(f"Preview saved to: {preview_path}")


# %% [markdown]
# ## استخراج و ذخیرهٔ همهٔ فریم‌ها
#
# هر ویدئو در پوشهٔ برچسب خودش قرار می‌گیرد. اگر هر 8 فریم از قبل وجود داشته
# باشند، آن ویدئو رد می‌شود؛ بنابراین اجرای دوباره امن و قابل ادامه‌دادن است.

# %%
frame_records = []
processed = 0
skipped = 0
failed = []

for _, row in tqdm(videos.iterrows(), total=len(videos), desc="Extracting frames"):
    video_path = Path(row["local_path"])
    label = int(row["label"])
    paths = output_paths(video_path, label)

    try:
        frames, positions, fps, total_frames = read_selected_frames(video_path)
        frames_already_saved = all(path.exists() for path in paths)
        if frames_already_saved:
            skipped += 1
        else:
            for path, frame in zip(paths, frames):
                path.parent.mkdir(parents=True, exist_ok=True)
                resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                success = cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not success:
                    raise RuntimeError(f"Cannot write frame: {path}")
            processed += 1

        for sequence_index, (path, source_index) in enumerate(zip(paths, positions)):
            frame_records.append(
                {
                    "video_id": video_path.stem,
                    "video_path": str(video_path),
                    "label": label,
                    "frame_order": sequence_index,
                    "source_frame_index": int(source_index),
                    "timestamp_seconds": float(source_index / fps),
                    "total_video_frames": total_frames,
                    "fps": fps,
                    "frame_path": str(path),
                }
            )
    except Exception as error:
        failed.append({"video_path": str(video_path), "error": str(error)})

frame_index = pd.DataFrame(frame_records)
frame_index_path = DATA_ROOT / "frame_index.csv"
frame_index.to_csv(frame_index_path, index=False)

failed_path = DATA_ROOT / "frame_extraction_failures.csv"
pd.DataFrame(failed).to_csv(failed_path, index=False)

print(
    {
        "processed_videos": processed,
        "skipped_videos": skipped,
        "failed_videos": len(failed),
        "new_frames": len(frame_records),
    }
)
print(f"Frame index: {frame_index_path}")
