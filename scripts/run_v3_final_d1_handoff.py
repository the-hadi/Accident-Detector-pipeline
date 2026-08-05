"""Train and save the final selected D1 artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v3_final_d1_handoff import run_final_handoff


if __name__ == "__main__":
    print(json.dumps(run_final_handoff(), ensure_ascii=False, indent=2), flush=True)
