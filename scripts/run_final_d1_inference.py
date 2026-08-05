"""CLI: python run_final_d1_inference.py C:\\path\\to\\video.mp4"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from final_d1_inference import predict_full_mp4


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_final_d1_inference.py <path-to-video.mp4>")
    result = predict_full_mp4(sys.argv[1])
    print(json.dumps({key: value for key, value in result.items() if key != "window_predictions"}, ensure_ascii=False, indent=2))
