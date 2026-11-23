"""Step 2: build data/processed/customers.parquet (one row per client + split) and a split report.

Usage:
    python scripts/build_dataset.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.dataset import build_and_save  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    df, report = build_and_save()
    out = ROOT / "reports" / "split_report.csv"
    out.parent.mkdir(exist_ok=True)
    report.to_csv(out, float_format="%.4f")
    print(f"customers.parquet: {len(df):,} rows x {df.shape[1]} columns  ({time.time() - t:.0f}s)")
    print(report.round(4).to_string())
