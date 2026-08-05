"""Run the CPU-friendly V3 A6 temporal experiment outside Jupyter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_a6_bilstm_additive_attention import (
    Config,
    build_context,
    cache_preflight,
    context_report,
    evaluate_best,
    load_rgb_features,
    train_model,
)


def main() -> None:
    context = build_context(Config())
    print(json.dumps(context_report(context), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(cache_preflight(context), ensure_ascii=False, indent=2), flush=True)
    features = load_rgb_features(context)
    print(json.dumps(train_model(context, features), ensure_ascii=False, indent=2), flush=True)
    evaluation = evaluate_best(context, features)
    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2), flush=True)
    print(evaluation["aggregation_ablation"].to_string(index=False), flush=True)
    print(evaluation["attention_summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
