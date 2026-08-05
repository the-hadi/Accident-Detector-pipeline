"""Run all code cells of the resumable V3-3 hard-negative-mining notebook."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "33_v3_hard_negative_mining.ipynb"

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
scope = {"__name__": "__main__"}

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), scope)
