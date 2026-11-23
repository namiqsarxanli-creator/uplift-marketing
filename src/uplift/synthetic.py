"""Synthetic clients for the public demo.

The X5 data cannot be redistributed, so the demo never shows real rows. Clients are drawn from
the *aggregated* decile profile stored in model_meta.json (uniform inside each decile bin) and then
made internally consistent (spend = purchases × basket, first purchase before last, ...).
They look like the training population in each column but are not real people.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _draw(profile: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    edges = np.asarray(profile["edges"], float)
    shares = np.asarray(profile["shares"], float)
    lo = np.concatenate([[edges[0] - (edges[1] - edges[0]) if len(edges) > 1 else 0], edges])
    hi = np.concatenate([edges, [edges[-1] + (edges[-1] - edges[-2]) if len(edges) > 1 else 1]])
    b = rng.choice(len(shares), size=n, p=shares / shares.sum())
    x = rng.uniform(lo[b], hi[b])
    missing = rng.random(n) < profile.get("missing_share", 0)
    return np.where(missing, np.nan, x)


def synthetic_clients(reference: dict, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = {c: _draw(reference[c], n, rng) for c, r in reference.items() if r["type"] == "numeric"}
    df = pd.DataFrame({
        "client_id": [f"synthetic-{i:05d}" for i in range(n)],
        "age": np.clip(g["age"], 16, 90).round(),
        "gender": rng.choice(list(reference["gender"]["shares"]), size=n, p=list(reference["gender"]["shares"].values())),
        "card_age_days": np.clip(g["card_age_days"], 5, 700).round().astype(int),
        "n_transactions": np.clip(g["n_transactions"], 1, 150).round().astype(int),
        "avg_basket": np.clip(g["avg_basket"], 30, 3000).round(2),
        "avg_items": np.clip(g["avg_items"], 1, 40).round(2),
        "n_stores": np.clip(g["n_stores"], 1, 15).round().astype(int),
        "points_received": np.clip(g["points_received"], 0, None).round(1),
        "points_spent": np.clip(np.nan_to_num(g["points_spent"]), 0, None).round(1),
        "days_since_last_purchase": np.clip(g["days_since_last_purchase"], 1, 23).round().astype(int),
    })
    df["total_spend"] = (df["n_transactions"] * df["avg_basket"]).round(2)
    df["days_since_first_purchase"] = np.maximum(
        np.clip(g["days_since_first_purchase"], 4, 118).round(), df["days_since_last_purchase"]).astype(int)
    redeem = np.clip(np.minimum(g["days_since_first_redeem"], df["card_age_days"]), 0, None)
    df["days_since_first_redeem"] = np.where(np.isnan(redeem), np.nan, np.round(redeem))
    df.loc[df["n_stores"] > df["n_transactions"], "n_stores"] = df["n_transactions"]
    return df
