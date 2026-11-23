"""Step 4: build data/processed/features.parquet (40 behavioural features per client, ~30 s).

Usage:
    python scripts/build_features.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.features import build_and_save  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    f = build_and_save()
    print(f"features.parquet: {len(f):,} clients x {f.shape[1] - 1} features ({time.time() - t:.0f}s)")
