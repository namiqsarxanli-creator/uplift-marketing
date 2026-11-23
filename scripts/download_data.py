"""Download the datasets into data/raw/.

X5 RetailHero (main dataset): not redistributed in this repository (no explicit licence is
published for it); fetched from the public Hugging Face mirror of the original competition
files (https://ods.ai/competitions/x5-retailhero-uplift-modeling).

Criteo Uplift v2.1 (transfer check, `--criteo`, 311 MB): CC BY-NC-SA 4.0,
Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018.

Usage:
    python scripts/download_data.py            # X5 only (~620 MB)
    python scripts/download_data.py --criteo   # also Criteo
"""
import sys
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/resolve/main/data/"
FILES = [
    "clients.csv.gz",
    "products.csv.gz",
    "purchases.csv.gz",
    "uplift_train.csv.gz",
    "uplift_test.csv.gz",
]
CRITEO = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(name: str, url: str | None = None) -> None:
    target = RAW / name
    if target.exists():
        print(f"skip {name} (already downloaded)")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url or BASE + name, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
        if total and done != total:
            raise IOError(f"{name}: expected {total} bytes, got {done}")
    tmp.rename(target)
    print(f"ok   {name} ({done / 1e6:.1f} MB)")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        download(f)
    if "--criteo" in sys.argv:
        download("criteo-uplift-v2.1.csv.gz", CRITEO)
