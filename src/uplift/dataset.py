"""Step 2: one row per client, built only from information available before the campaign.

Output columns:
    client_id, treatment_flg, target          campaign design and outcome
    client attributes                         age, gender, card age, redemption status (censored)
    basic activity aggregates                 transactions, spend, recency, tenure in the log, stores
    split                                     train / valid / test

Richer behavioural features belong to step 4 (`features.py`); this module only fixes the
grain, the time cutoff and the evaluation splits so every later step uses the same ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import (
    CAMPAIGN_START,
    PROCESSED,
    connect,
    ensure_purchases_parquet,
    load_campaign,
    load_clients,
)

SPLIT_SIZES = {"train": 0.6, "valid": 0.2, "test": 0.2}
SEED = 42
SMD_LIMIT = 0.05


def activity_aggregates(cutoff: pd.Timestamp = CAMPAIGN_START) -> pd.DataFrame:
    """Per-client purchase aggregates from transactions strictly before `cutoff`.

    `purchase_sum` and the points columns are transaction-level values repeated on every
    line item, so they are taken once per transaction (summing raw rows inflates spend ~10x).
    """
    con = connect()
    path = ensure_purchases_parquet(con).as_posix()
    cut = cutoff.isoformat(sep=" ")
    return con.sql(f"""
        WITH trx AS (
            SELECT client_id, transaction_id,
                   any_value(transaction_datetime)    AS ts,
                   any_value(purchase_sum)            AS trx_sum,
                   any_value(store_id)                AS store_id,
                   any_value(regular_points_received) AS pts_received,
                   any_value(regular_points_spent)    AS pts_spent,
                   sum(product_quantity)              AS items
            FROM '{path}'
            WHERE transaction_datetime < TIMESTAMP '{cut}'
            GROUP BY client_id, transaction_id)
        SELECT client_id,
               count(*)                                              AS n_transactions,
               sum(trx_sum)                                          AS total_spend,
               avg(trx_sum)                                          AS avg_basket,
               avg(items)                                            AS avg_items,
               count(DISTINCT store_id)                              AS n_stores,
               sum(pts_received)                                     AS points_received,
               -sum(pts_spent)                                       AS points_spent,
               date_diff('day', max(ts), TIMESTAMP '{cut}')          AS days_since_last_purchase,
               date_diff('day', min(ts), TIMESTAMP '{cut}')          AS days_since_first_purchase,
               max(ts)                                               AS last_purchase_ts
        FROM trx GROUP BY client_id
    """).df()


def build_base_table(cutoff: pd.Timestamp = CAMPAIGN_START, censor_redeem: bool = True) -> pd.DataFrame:
    """`censor_redeem=False` rebuilds the leaky version, only to measure how much the leak inflates scores."""
    clients = load_clients(censor_at=cutoff if censor_redeem else None)
    df = load_campaign().merge(clients, on="client_id", how="left", validate="one_to_one")
    df = df.merge(activity_aggregates(cutoff), on="client_id", how="left", validate="one_to_one")

    df["card_age_days"] = (cutoff - df["first_issue_date"]).dt.days
    df["days_since_first_redeem"] = (cutoff - df["first_redeem_date"]).dt.days
    # Training/serving consistency: the API only receives whole-day ages, so the flag is defined on
    # those (a redemption a few hours before issue on the same day is not "before issue" here).
    df["redeem_before_issue"] = df["days_since_first_redeem"] > df["card_age_days"]
    df["gender"] = df["gender"].astype("category")
    return df


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratify on treatment x outcome so every split keeps the same design and base rates."""
    strata = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    train_idx, rest_idx = train_test_split(
        df.index, train_size=SPLIT_SIZES["train"], stratify=strata, random_state=seed
    )
    rel_valid = SPLIT_SIZES["valid"] / (SPLIT_SIZES["valid"] + SPLIT_SIZES["test"])
    valid_idx, test_idx = train_test_split(
        rest_idx, train_size=rel_valid, stratify=strata.loc[rest_idx], random_state=seed
    )
    df = df.copy()
    df["split"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[train_idx, "split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx, "split"] = "test"
    return df


def _smd(x: pd.Series, t: pd.Series) -> float:
    a, b = x[t == 1].dropna(), x[t == 0].dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


BALANCE_COLS = [
    "age", "card_age_days", "redeemed_before_campaign", "n_transactions",
    "total_spend", "avg_basket", "n_stores", "days_since_last_purchase",
]


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per split: size, treated share, outcome rates, ATE and the worst covariate imbalance."""
    rows = []
    for name, g in df.groupby("split"):
        t, y = g["treatment_flg"], g["target"]
        smds = {c: _smd(g[c].astype(float), t) for c in BALANCE_COLS}
        worst = max(smds, key=lambda c: abs(smds[c]))
        rows.append({
            "split": name,
            "clients": len(g),
            "treated_share": t.mean(),
            "rate_control": y[t == 0].mean(),
            "rate_treatment": y[t == 1].mean(),
            "ate_pp": (y[t == 1].mean() - y[t == 0].mean()) * 100,
            "max_abs_smd": abs(smds[worst]),
            "worst_covariate": worst,
        })
    return pd.DataFrame(rows).set_index("split").loc[list(SPLIT_SIZES)]


def build_and_save(cutoff: pd.Timestamp = CAMPAIGN_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = assign_splits(build_base_table(cutoff))
    assert df["last_purchase_ts"].max() < cutoff, "purchase after the campaign start leaked in"
    assert df["first_redeem_date"].max() < cutoff, "redeem date after the campaign start leaked in"
    report = split_report(df)
    assert (report["max_abs_smd"] < SMD_LIMIT).all(), f"split imbalance:\n{report}"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["last_purchase_ts"]).to_parquet(PROCESSED / "customers.parquet", index=False)
    return df, report
