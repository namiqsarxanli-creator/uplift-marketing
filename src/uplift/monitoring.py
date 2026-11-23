"""Input drift monitoring with the Population Stability Index (PSI).

At training time the decile edges and shares of every numeric feature (and the category
shares of `gender`) are stored in `model_meta.json`. A new batch of clients is binned the same
way; PSI = Σ (new − ref) · ln(new / ref). Common reading: < 0.1 stable, 0.1–0.25 watch, > 0.25 drift.
Drift means the clients being scored no longer look like the ones the effect was estimated on,
so the model should be re-validated on a fresh randomised holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
WATCH, DRIFT = 0.1, 0.25


def reference_profile(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    ref = {}
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
        edges = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        ref[c] = {"type": "numeric", "edges": edges.round(6).tolist(),
                  "shares": (counts / counts.sum()).round(6).tolist(),
                  "missing_share": round(float(df[c].isna().mean()), 6)}
    for c in categorical:
        shares = df[c].astype(str).value_counts(normalize=True)
        ref[c] = {"type": "categorical", "shares": shares.round(6).to_dict()}
    return ref


def _psi(new: np.ndarray, ref: np.ndarray) -> float:
    new, ref = np.clip(new, EPS, None), np.clip(ref, EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def drift_report(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    rows = []
    for c, r in reference.items():
        if c not in df:
            rows.append({"feature": c, "psi": np.nan, "status": "missing column"})
            continue
        if r["type"] == "numeric":
            x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
            if len(x) == 0:
                rows.append({"feature": c, "psi": np.nan, "status": "all values missing"})
                continue
            edges = np.asarray(r["edges"])
            counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
            psi = _psi(counts / counts.sum(), np.asarray(r["shares"]))
        else:
            ref = pd.Series(r["shares"])
            new = df[c].astype(str).value_counts(normalize=True).reindex(ref.index.union(df[c].astype(str).unique()), fill_value=0)
            psi = _psi(new.to_numpy(), ref.reindex(new.index, fill_value=0).to_numpy())
        status = "drift" if psi > DRIFT else "watch" if psi > WATCH else "stable"
        rows.append({"feature": c, "psi": round(psi, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="first").reset_index(drop=True)
