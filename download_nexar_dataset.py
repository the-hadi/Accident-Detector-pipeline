# %% [markdown]
# # دریافت محلی دیتاست Nexar Collision Prediction
#
# این فایل با سلول‌های `# %%` در VS Code مانند یک نوت‌بوک Jupyter اجرا می‌شود.
# ابتدا تمام متادیتا از API دریافت می‌شود؛ سپس همه‌ی ویدئوهای `train` و `test`
# مستقیماً از URLهای رسمی Hugging Face دانلود می‌شوند. دانلود قابل ادامه دادن است.

# %%
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm


# %% [markdown]
# ## پیکربندی
#
# حجم کامل داده حدود 31.4GB است. برای جلوگیری از پرشدن درایو C، داده‌ها روی P
# ذخیره می‌شوند. پیش از شروع دانلود، حداقل 35GB فضای خالی در درایو مقصد نگه دارید.

# %%
DATASET_ID = "nexar-ai/nexar_collision_prediction"
DATA_ROOT = Path(r"P:\NexarCollisionData")
API_ROWS = "https://datasets-server.huggingface.co/rows"
API_SPLITS = "https://datasets-server.huggingface.co/splits"
PAGE_SIZE = 100
MAX_WORKERS = 4  # بالاتر بردن این مقدار معمولاً اینترنت/دیسک را بی‌دلیل تحت فشار می‌گذارد.
DOWNLOAD_VIDEOS = True
METADATA_REQUEST_INTERVAL_SECONDS = 7
TRAIN_VIDEOS_PER_CLASS = 300
RANDOM_SEED = 42

DATA_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Data root: {DATA_ROOT}")


def get_with_retry(url: str, *, attempts: int = 6, **kwargs) -> requests.Response:
    """GET مقاوم در برابر خطاهای موقت 429/5xx در سرویس Hugging Face."""
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(f"HTTP {response.status_code}: {response.url}")
        except requests.RequestException as error:
            last_error = error
        if attempt < attempts - 1:
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}")


# %% [markdown]
# ## دریافت همه‌ی متادیتا و ساخت برچسب آموزشی
#
# API `rows` فقط متادیتا و URL ویدئو را می‌دهد، نه خود فایل ویدئو. بنابراین برای
# ساخت manifest کم‌حجم مناسب است. برچسب `label` فقط برای `train` ساخته می‌شود:
# وجود `time_of_event` یعنی نمونه‌ی مثبت.

# %%
def get_splits() -> list[str]:
    response = get_with_retry(API_SPLITS, params={"dataset": DATASET_ID}, timeout=60)
    return [item["split"] for item in response.json()["splits"]]


def get_split_rows(split: str) -> list[dict]:
    first = get_with_retry(
        API_ROWS,
        params={
            "dataset": DATASET_ID,
            "config": "default",
            "split": split,
            "offset": 0,
            "length": 1,
        },
        timeout=60,
    )
    first.raise_for_status()
    total = first.json()["num_rows_total"]

    cache_dir = DATA_ROOT / "metadata_page_cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for offset in tqdm(range(0, total, PAGE_SIZE), desc=f"Metadata: {split}"):
        cache_file = cache_dir / f"{offset:05d}.json"
        if cache_file.exists():
            page_rows = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            # درایوهای شبکه/خارجی ممکن است در میانه‌ی اجرا پوشه را حذف کنند.
            # ایجاد دوباره‌ی پوشه باعث می‌شود cache قابل ادامه بماند.
            cache_dir.mkdir(parents=True, exist_ok=True)
            response = get_with_retry(
                API_ROWS,
                params={
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": PAGE_SIZE,
                },
                timeout=60,
            )
            page_rows = response.json()["rows"]
            cache_file.write_text(json.dumps(page_rows), encoding="utf-8")
            time.sleep(METADATA_REQUEST_INTERVAL_SECONDS)
        rows.extend(page_rows)
    return rows


def build_manifest(split: str, rows: list[dict]) -> pd.DataFrame:
    records = []
    for item in rows:
        row = item["row"]
        video_url = row["video"]["src"]
        filename = Path(video_url.split("?")[0]).name
        is_positive = row.get("time_of_event") is not None
        records.append(
            {
                "row_idx": item["row_idx"],
                "split": split,
                "video_url": video_url,
                "local_path": str(DATA_ROOT / split / ("positive" if is_positive else "negative") / filename),
                "time_of_event": row.get("time_of_event"),
                "time_of_alert": row.get("time_of_alert"),
                "light_conditions": row.get("light_conditions"),
                "weather": row.get("weather"),
                "scene": row.get("scene"),
                "time_to_accident": row.get("time_to_accident"),
                "label": int(is_positive) if split == "train" else pd.NA,
            }
        )
    return pd.DataFrame.from_records(records)


splits = get_splits()
print("Available splits:", splits)

manifests = {}
for split_name in splits:
    manifest_path = DATA_ROOT / f"{split_name}_metadata.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        print(f"{split_name}: loaded existing manifest from {manifest_path}")
    else:
        manifest = build_manifest(split_name, get_split_rows(split_name))
        manifest.to_csv(manifest_path, index=False)
    manifests[split_name] = manifest
    print(f"{split_name}: {len(manifest):,} rows -> {manifest_path}")

train_df = manifests["train"]
print("Train class counts:")
print(train_df["label"].value_counts().sort_index())


# %% [markdown]
# ## دانلود ویدئوها با امکان Resume
#
# هر فایل ابتدا با پسوند `.part` نوشته می‌شود و فقط پس از کامل‌شدن به `.mp4`
# تغییر نام می‌دهد. اجرای دوباره‌ی این سلول فایل‌های سالم را رد می‌کند و دانلود
# ناقص را از همان نقطه ادامه می‌دهد.

# %%
def download_one(record: dict) -> tuple[str, str]:
    url = record["video_url"]
    destination = Path(record["local_path"])
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_size = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    with get_with_retry(url, stream=True, headers=headers, timeout=(30, 300)) as response:
        content_length = int(response.headers.get("Content-Length", 0))
        # اگر فایل کامل قبلی هم‌اندازه‌ی نسخه‌ی رسمی بود، دوباره دریافتش نکن.
        if destination.exists() and response.status_code == 200 and content_length:
            if destination.stat().st_size == content_length:
                return "skipped", str(destination)
            # فایل ناقصی که اشتباهاً .mp4 نام گرفته، دوباره از ابتدا دانلود می‌شود.
            destination.unlink()

        # اگر سرور Resume را نپذیرد، فایل ناقص را از نو می‌گیریم.
        if existing_size and response.status_code == 200:
            partial.unlink(missing_ok=True)
            existing_size = 0
        mode = "ab" if existing_size and response.status_code == 206 else "wb"
        with partial.open(mode) as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    expected_size = existing_size + content_length if response.status_code == 206 else content_length
    if expected_size and partial.stat().st_size != expected_size:
        raise IOError(f"Incomplete download: {destination.name}")
    os.replace(partial, destination)
    return "downloaded", str(destination)


if DOWNLOAD_VIDEOS:
    # برای نسخه‌ی نخست پروژه فقط 600 ویدئوی train (300 مثبت، 300 منفی) کافی است.
    # ویدئوهای کامل قبلی را تا سقف هر کلاس نگه می‌داریم تا دوباره دانلود نشوند.
    selected_parts: list[pd.DataFrame] = []
    for label in (0, 1):
        class_rows = train_df[train_df["label"] == label].copy()
        existing = class_rows[class_rows["local_path"].map(lambda path: Path(path).exists())]
        retained = existing.sample(
            n=min(len(existing), TRAIN_VIDEOS_PER_CLASS), random_state=RANDOM_SEED
        )
        needed = TRAIN_VIDEOS_PER_CLASS - len(retained)
        candidates = class_rows.drop(index=retained.index)
        added = candidates.sample(n=needed, random_state=RANDOM_SEED)
        selected_parts.append(pd.concat([retained, added]))

    selected_train = pd.concat(selected_parts).sample(frac=1, random_state=RANDOM_SEED)
    selected_train.to_csv(DATA_ROOT / "selected_train_600.csv", index=False)
    print("Selected videos by class:")
    print(selected_train["label"].value_counts().sort_index())
    download_records = selected_train.to_dict("records")
    outcomes = {"downloaded": 0, "skipped": 0, "failed": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, record) for record in download_records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
            try:
                status, _ = future.result()
                outcomes[status] += 1
            except Exception as error:
                outcomes["failed"].append(str(error))

    (DATA_ROOT / "download_report.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({key: value if key != "failed" else len(value) for key, value in outcomes.items()})


# %% [markdown]
# ## کنترل نهایی

# %%
for split_name, manifest in manifests.items():
    expected = len(manifest)
    present = sum(Path(path).exists() for path in manifest["local_path"])
    print(f"{split_name}: {present:,} / {expected:,} videos present")
