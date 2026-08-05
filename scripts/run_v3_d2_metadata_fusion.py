"""Run the V3 D2 metadata ablation outside Jupyter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_d2_metadata_fusion import Config, build_context, cache_preflight, context_report, run_experiment


def main() -> None:
    context = build_context(Config())
    print(json.dumps(context_report(context), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(cache_preflight(context), ensure_ascii=False, indent=2), flush=True)
    result = run_experiment()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print("\nMetadata-only metrics:\n", json.dumps(result["metadata_only"]["metrics"], ensure_ascii=False, indent=2), flush=True)
    print("\nVideo-only aggregation:\n", result["video_only"]["aggregation"].to_string(index=False), flush=True)
    print("\nFusion aggregation:\n", result["fusion"]["aggregation"].to_string(index=False), flush=True)
    print("\nMetadata bias report:\n", result["bias_report"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
