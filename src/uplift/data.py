"""Loading the raw X5 RetailHero files and basic cleaning.

All heavy work on the 45.8M-row purchase log goes through DuckDB so it never
has to fit into pandas memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # DuckDB is only needed for data preparation, not by the API container
    import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PURCHASES_PARQUET = PROCESSED / "purchases.parquet"

# The SMS campaign started right after the last purchase in the history file.
# Every feature must be computed strictly before this moment.
CAMPAIGN_START = pd.Timestamp("2019-03-19")

AGE_MIN, AGE_MAX = 14, 100


def connect() -> "duckdb.DuckDBPyConnection":
    import duckdb

    con = duckdb.connect()
    try:
        con.sql("SET enable_progress_bar = false")
    except duckdb.InvalidInputException:
        pass  # inside Jupyter without ipywidgets DuckDB refuses to touch this setting
    return con


def ensure_purchases_parquet(con: "duckdb.DuckDBPyConnection | None" = None) -> Path:
    """Convert purchases.csv.gz to parquet once (≈15 s); later reads are fast."""
    if not PURCHASES_PARQUET.exists():
        con = con or connect()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        con.sql(
            f"COPY (SELECT * FROM read_csv_auto('{(RAW / 'purchases.csv.gz').as_posix()}')) "
            f"TO '{PURCHASES_PARQUET.as_posix()}' (FORMAT parquet)"
        )
    return PURCHASES_PARQUET


def load_clients(censor_at: pd.Timestamp | None = CAMPAIGN_START) -> pd.DataFrame:
    """Client attributes with the data errors made explicit.

    * ages outside [14, 100] (e.g. -7491, 1901) become NaN; `age_invalid` keeps the signal
    * LEAKAGE FIX: `first_redeem_date` runs until 2019-11-20, i.e. 8 months *after* the
      campaign. For 44,953 clients the first redemption happened after the SMS was sent,
      so the raw column partly encodes the outcome. Dates on/after `censor_at` are treated
      as "had not redeemed yet when the campaign started". Pass `censor_at=None` only to
      reproduce the leaky version for comparison.
    * `redeemed_before_campaign` is False for never-redeemed and censored clients.
    """
    c = pd.read_csv(RAW / "clients.csv.gz", parse_dates=["first_issue_date", "first_redeem_date"])
    c["age_invalid"] = ~c["age"].between(AGE_MIN, AGE_MAX)
    c.loc[c["age_invalid"], "age"] = pd.NA
    if censor_at is not None:
        c["redeem_censored"] = c["first_redeem_date"] >= censor_at
        c.loc[c["redeem_censored"], "first_redeem_date"] = pd.NaT
    c["redeemed_before_campaign"] = c["first_redeem_date"].notna()
    # 536 clients have a redemption date earlier than the card issue date: a data error.
    c["redeem_before_issue"] = c["first_redeem_date"] < c["first_issue_date"]
    return c


def load_campaign() -> pd.DataFrame:
    """Labelled campaign participants: client_id, treatment_flg, target.

    `uplift_test.csv` from the competition has no labels, so it is not used for evaluation.
    """
    return pd.read_csv(RAW / "uplift_train.csv.gz")


def load_products() -> pd.DataFrame:
    return pd.read_csv(RAW / "products.csv.gz")
