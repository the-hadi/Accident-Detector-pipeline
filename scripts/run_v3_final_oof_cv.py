"""Run final five-fold OOF training after the final feature caches are complete."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_final_oof_cv import run_final_oof_cv


if __name__ == "__main__":
    result = run_final_oof_cv()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(result["comparison"].to_string(index=False), flush=True)
