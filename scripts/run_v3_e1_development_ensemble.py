"""Run the cheap, non-learned E1 development diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_e1_development_ensemble import context_report, run_diagnostic


def main() -> None:
    print(json.dumps(context_report(), ensure_ascii=False, indent=2), flush=True)
    result = run_diagnostic()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(result["comparison"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
