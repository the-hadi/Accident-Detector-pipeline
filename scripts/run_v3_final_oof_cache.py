"""Start/continue final OOF feature-cache construction outside Jupyter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_final_oof_cache import Config, build_context, cache_preflight, context_report, ensure_feature_cache, write_summary


def main() -> None:
    context = build_context(Config())
    print(json.dumps(context_report(context), ensure_ascii=False, indent=2), flush=True)
    preflight = cache_preflight(context)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    rgb = ensure_feature_cache(context, "rgb", run_full=True)
    print(json.dumps(rgb, ensure_ascii=False, indent=2), flush=True)
    motion = ensure_feature_cache(context, "motion", run_full=True)
    print(json.dumps(motion, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(write_summary(context, rgb, motion, preflight), ensure_ascii=False, indent=2), flush=True)
    if not (rgb["complete"] and motion["complete"]):
        raise RuntimeError("Final OOF cache remains incomplete; rerun safely to resume.")


if __name__ == "__main__":
    main()
