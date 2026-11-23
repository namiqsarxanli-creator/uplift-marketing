"""Guards for the two things that silently ruin uplift results: leakage and broken splits.

Run after `python scripts/download_data.py` and `python scripts/build_dataset.py`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.data import CAMPAIGN_START, PROCESSED, load_clients  # noqa: E402
from uplift.dataset import SPLIT_SIZES, split_report  # noqa: E402

CUSTOMERS = PROCESSED / "customers.parquet"
pytestmark = pytest.mark.skipif(not CUSTOMERS.exists(), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return pd.read_parquet(CUSTOMERS)


def test_one_row_per_client(customers):
    assert customers["client_id"].is_unique


def test_redeem_date_is_censored_at_campaign_start(customers):
    assert customers["first_redeem_date"].max() < CAMPAIGN_START


def test_raw_redeem_date_would_leak():
    """Documents why censoring exists: the raw column reaches months past the campaign."""
    raw = load_clients(censor_at=None)
    assert raw["first_redeem_date"].max() > CAMPAIGN_START


def test_activity_is_pre_campaign(customers):
    assert customers["days_since_last_purchase"].min() >= 1


def test_splits_are_disjoint_and_sized(customers):
    shares = customers["split"].value_counts(normalize=True)
    for name, size in SPLIT_SIZES.items():
        assert abs(shares[name] - size) < 0.005


def test_splits_keep_randomisation(customers):
    report = split_report(customers)
    assert (report["treated_share"].sub(0.5).abs() < 0.01).all()
    assert (report["max_abs_smd"] < 0.05).all()
