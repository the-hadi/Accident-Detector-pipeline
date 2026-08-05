"""Run the full resumable V3-4B A-MIL experiment outside the notebook UI."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_attention_mil import Config, cache_preflight, context_report, create_selected_bags, ensure_feature_cache, evaluate_best, train_model


def main() -> None:
    context = create_selected_bags(Config())
    print(json.dumps(context_report(context), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(cache_preflight(context), ensure_ascii=False, indent=2), flush=True)
    cache = ensure_feature_cache(context, run_full_cache=True)
    print(json.dumps({key: value for key, value in cache.items() if key not in {"features_by_sequence", "feature_source_by_sequence"}}, ensure_ascii=False, indent=2), flush=True)
    if not cache["complete"]:
        raise RuntimeError("A-MIL feature cache remains incomplete; rerun this script to resume.")
    print(json.dumps(train_model(context, cache), ensure_ascii=False, indent=2), flush=True)
    evaluation = evaluate_best(context, cache)
    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
