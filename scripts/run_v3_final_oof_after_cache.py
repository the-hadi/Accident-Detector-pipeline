"""Wait for the single cache writer, then run final OOF CV once caches are complete."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_final_oof_cache import Config as CacheConfig
from v3_final_oof_cv import run_final_oof_cv


POLL_SECONDS = 60
MAX_WAIT_HOURS = 60


def cache_is_complete(config: CacheConfig) -> bool:
    if not (config.cache_summary_path.is_file() and config.rgb_features_path.is_file() and config.motion_features_path.is_file()):
        return False
    try:
        payload = json.loads(config.cache_summary_path.read_text(encoding="utf-8"))
        return bool(payload.get("cache_complete"))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    cache_config = CacheConfig()
    started = time.monotonic()
    checks = 0
    while not cache_is_complete(cache_config):
        checks += 1
        elapsed_hours = (time.monotonic() - started) / 3600.0
        print({"utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_resumable_feature_cache", "elapsed_hours": round(elapsed_hours, 2), "checks": checks}, flush=True)
        if elapsed_hours >= MAX_WAIT_HOURS:
            raise TimeoutError("Final OOF cache did not complete within the configured waiting window; inspect cache logs before retrying.")
        time.sleep(POLL_SECONDS)
    print({"utc": datetime.now(timezone.utc).isoformat(), "status": "feature_cache_complete_starting_final_oof_cv"}, flush=True)
    result = run_final_oof_cv()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(result["comparison"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
