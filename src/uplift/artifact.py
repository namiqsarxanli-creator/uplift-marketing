"""Saving and loading the trained model together with what is needed to use it safely."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import sklearn

from .data import ROOT

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "uplift_model.pkl"
META_PATH = MODEL_DIR / "model_meta.json"


def save(model, metadata: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model_class": type(model).__name__,
        "features": model.spec.columns,
        **metadata,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: Path = MODEL_PATH):
    """Only load model files you produced yourself: pickle executes code on load."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_meta(path: Path = META_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
