"""Fit the selected model on train+valid, calibrate on test, write models/uplift_model.pkl + model_meta.json.

Usage:
    python scripts/train_final.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.train import train_and_save  # noqa: E402

if __name__ == "__main__":
    meta = train_and_save()
    print(json.dumps({k: meta[k] for k in ("test_metrics", "tier_thresholds")}, indent=2))
