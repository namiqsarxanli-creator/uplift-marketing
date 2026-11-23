"""Step 4: behavioural features from the 45.8M-row purchase log.

Everything is computed in DuckDB from transactions strictly before `cutoff`, with windows
measured backwards from the cutoff ("last 7 days" = the 7 days before the campaign).
Each feature belongs to a named group so the notebook can run group ablations.

Unit notes:
* `purchase_sum` and points columns are per transaction (repeated on each line) -> de-duplicated.
* `trn_sum_from_iss` is the line price; `trn_sum_from_red` is the part of the line paid with points.
* points spent are stored as negative numbers; they are flipped to positive here.
"""
from __future__ import annotations

import pandas as pd

from .data import CAMPAIGN_START, PROCESSED, RAW, connect, ensure_purchases_parquet

FEATURE_GROUPS: dict[str, list[str]] = {
    "frequency": [
        "trx_7d", "trx_14d", "trx_30d", "trx_60d", "trx_total", "active_days",
        "gap_mean", "gap_std", "gap_max", "overdue_ratio", "trx_trend_30_vs_prev",
    ],
    "monetary": [
        "spend_7d", "spend_30d", "spend_total", "basket_mean", "basket_std", "basket_max",
        "basket_trend_30_vs_all",
    ],
    "loyalty_points": [
        "pts_received", "pts_spent", "pts_balance_proxy", "share_trx_pts_spent",
        "share_spend_paid_with_pts", "express_spent_trx", "express_received_trx",
        "days_since_pts_spent",
    ],
    "basket_content": [
        "items_per_trx", "distinct_products", "products_per_trx", "own_brand_share",
        "alcohol_share", "cat1_a_share", "cat1_b_share", "cat1_c_share",
    ],
    "habits": [
        "weekend_share", "evening_share", "morning_share", "hour_mean",
        "n_stores", "fav_store_share",
    ],
}
ALL_BEHAVIOUR = [f for cols in FEATURE_GROUPS.values() for f in cols]
FEATURES_PARQUET = PROCESSED / "features.parquet"


def build_features(cutoff: pd.Timestamp = CAMPAIGN_START) -> pd.DataFrame:
    con = connect()
    path = ensure_purchases_parquet(con).as_posix()
    products = (RAW / "products.csv.gz").as_posix()
    c = cutoff.isoformat(sep=" ")

    # the three top-level categories are hashed; fix their order by catalogue size
    cats = con.sql(f"""
        SELECT level_1 FROM read_csv_auto('{products}') WHERE level_1 IS NOT NULL
        GROUP BY 1 ORDER BY count(*) DESC""").df()["level_1"].tolist()
    assert len(cats) == 3, cats

    return con.sql(f"""
    WITH lines AS (
        SELECT p.*, pr.level_1, pr.is_own_trademark, pr.is_alcohol
        FROM '{path}' p LEFT JOIN read_csv_auto('{products}') pr USING (product_id)
        WHERE p.transaction_datetime < TIMESTAMP '{c}'
    ),
    trx AS (
        SELECT client_id, transaction_id,
               any_value(transaction_datetime)         AS ts,
               any_value(purchase_sum)                 AS s,
               any_value(store_id)                     AS store_id,
               any_value(regular_points_received)      AS rr,
               -any_value(regular_points_spent)        AS rs,
               any_value(express_points_received)      AS er,
               -any_value(express_points_spent)        AS es,
               sum(product_quantity)                   AS items,
               count(DISTINCT product_id)              AS n_products
        FROM lines GROUP BY client_id, transaction_id
    ),
    trx_d AS (
        SELECT *, date_diff('day', ts, TIMESTAMP '{c}') AS days_ago,
               date_diff('day', lag(ts) OVER (PARTITION BY client_id ORDER BY ts), ts) AS gap
        FROM trx
    ),
    per_client AS (
        SELECT client_id,
            -- frequency
            count(*) FILTER (WHERE days_ago < 7)              AS trx_7d,
            count(*) FILTER (WHERE days_ago < 14)             AS trx_14d,
            count(*) FILTER (WHERE days_ago < 30)             AS trx_30d,
            count(*) FILTER (WHERE days_ago < 60)             AS trx_60d,
            count(*)                                          AS trx_total,
            count(DISTINCT CAST(ts AS DATE))                  AS active_days,
            avg(gap)                                          AS gap_mean,
            stddev_samp(gap)                                  AS gap_std,
            max(gap)                                          AS gap_max,
            min(days_ago)                                     AS days_since_last,
            count(*) FILTER (WHERE days_ago >= 30 AND days_ago < 90) AS trx_30_90d,
            -- monetary
            sum(s) FILTER (WHERE days_ago < 7)                AS spend_7d,
            sum(s) FILTER (WHERE days_ago < 30)               AS spend_30d,
            sum(s)                                            AS spend_total,
            avg(s)                                            AS basket_mean,
            stddev_samp(s)                                    AS basket_std,
            max(s)                                            AS basket_max,
            avg(s) FILTER (WHERE days_ago < 30)               AS basket_30d,
            -- loyalty points
            sum(rr)                                           AS pts_received,
            sum(rs)                                           AS pts_spent,
            avg((rs > 0)::INT)                                AS share_trx_pts_spent,
            count(*) FILTER (WHERE es > 0)                    AS express_spent_trx,
            count(*) FILTER (WHERE er > 0)                    AS express_received_trx,
            min(days_ago) FILTER (WHERE rs > 0)               AS days_since_pts_spent,
            -- basket
            avg(items)                                        AS items_per_trx,
            avg(n_products)                                   AS products_per_trx,
            -- habits
            avg((dayofweek(ts) IN (0, 6))::INT)               AS weekend_share,
            avg((hour(ts) >= 18)::INT)                        AS evening_share,
            avg((hour(ts) < 11)::INT)                         AS morning_share,
            avg(hour(ts))                                     AS hour_mean,
            count(DISTINCT store_id)                          AS n_stores
        FROM trx_d GROUP BY client_id
    ),
    store_share AS (
        SELECT client_id, max(n) / sum(n) AS fav_store_share
        FROM (SELECT client_id, store_id, count(*) AS n FROM trx GROUP BY 1, 2) GROUP BY 1
    ),
    line_level AS (
        SELECT client_id,
            count(DISTINCT product_id)                                             AS distinct_products,
            sum(trn_sum_from_iss * is_own_trademark) / nullif(sum(trn_sum_from_iss), 0) AS own_brand_share,
            sum(trn_sum_from_iss * is_alcohol)       / nullif(sum(trn_sum_from_iss), 0) AS alcohol_share,
            sum(trn_sum_from_iss) FILTER (WHERE level_1 = '{cats[0]}') / nullif(sum(trn_sum_from_iss), 0) AS cat1_a_share,
            sum(trn_sum_from_iss) FILTER (WHERE level_1 = '{cats[1]}') / nullif(sum(trn_sum_from_iss), 0) AS cat1_b_share,
            sum(trn_sum_from_iss) FILTER (WHERE level_1 = '{cats[2]}') / nullif(sum(trn_sum_from_iss), 0) AS cat1_c_share,
            coalesce(sum(trn_sum_from_red), 0) / nullif(sum(trn_sum_from_iss), 0)  AS share_spend_paid_with_pts
        FROM lines GROUP BY client_id
    )
    SELECT pc.*, ss.fav_store_share, ll.* EXCLUDE (client_id),
           pc.days_since_last / nullif(pc.gap_mean, 0)                   AS overdue_ratio,
           pc.trx_30d / nullif(pc.trx_30_90d / 2.0, 0)                   AS trx_trend_30_vs_prev,
           pc.basket_30d / nullif(pc.basket_mean, 0)                     AS basket_trend_30_vs_all,
           pc.pts_received - pc.pts_spent                                AS pts_balance_proxy
    FROM per_client pc
    JOIN store_share ss USING (client_id)
    JOIN line_level ll USING (client_id)
    """).df()


def build_and_save(cutoff: pd.Timestamp = CAMPAIGN_START) -> pd.DataFrame:
    f = build_features(cutoff)
    missing = set(ALL_BEHAVIOUR) - set(f.columns)
    assert not missing, f"features not produced: {missing}"
    f = f[["client_id"] + ALL_BEHAVIOUR].copy()
    # empty FILTER windows mean "nothing happened", not "unknown"
    f["spend_7d"] = f["spend_7d"].fillna(0)
    for col in ["cat1_a_share", "cat1_b_share", "cat1_c_share"]:
        f[col] = f[col].fillna(0)
    # never spent points inside the 4-month history: place them just beyond the window
    f["days_since_pts_spent"] = f["days_since_pts_spent"].fillna(f["days_since_pts_spent"].max() + 1)
    # line prices can be zero/negative (returns, weighted goods), which breaks a few ratios
    f["share_spend_paid_with_pts"] = f["share_spend_paid_with_pts"].clip(0, 1)
    # gap_* and ratios built on them stay NaN for one-purchase clients; the model imputes + flags them
    f.to_parquet(FEATURES_PARQUET, index=False)
    return f


def load_model_table() -> pd.DataFrame:
    """Customer table from step 2 joined with the behavioural features."""
    customers = pd.read_parquet(PROCESSED / "customers.parquet")
    feats = pd.read_parquet(FEATURES_PARQUET)
    shared = [c for c in feats.columns if c in customers.columns and c != "client_id"]  # e.g. n_stores
    return customers.merge(feats.drop(columns=shared), on="client_id", how="left", validate="one_to_one")
