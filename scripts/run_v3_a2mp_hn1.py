"""Run the complete resumable V3-4A experiment outside the notebook UI."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_a2mp_hn1 import (
    Config,
    build_context,
    cache_preflight,
    context_report,
    ensure_feature_cache,
    evaluate_best_model,
    train_head,
)


def main() -> None:
    config = Config()
    context = build_context(config)
    print(json.dumps(context_report(context), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(cache_preflight(context), ensure_ascii=False, indent=2), flush=True)
    cache_result = ensure_feature_cache(context, run_full_cache=True)
    cache_report = {key: value for key, value in cache_result.items() if key not in {"features_by_sequence", "feature_source_by_sequence"}}
    print(json.dumps(cache_report, ensure_ascii=False, indent=2), flush=True)
    if not cache_result["complete"]:
        raise RuntimeError("V3-4 feature cache remains incomplete; rerun this same script to resume.")
    print(json.dumps(train_head(context, cache_result), ensure_ascii=False, default=str, indent=2), flush=True)
    evaluation = evaluate_best_model(context, cache_result)
    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2), flush=True)
    print(evaluation["aggregation_ablation"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
