import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.features import ALL_BEHAVIOUR, FEATURES_PARQUET  # noqa: E402
from uplift.models import FULL_SPEC, uplift_signal  # noqa: E402


def test_uplift_signal_separates_uplift_from_response():
    """x_resp drives purchases but not the effect; x_upl drives the effect only."""
    rng = np.random.default_rng(0)
    n = 60_000
    x_resp, x_upl = rng.normal(size=n), rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    p = 1 / (1 + np.exp(-(0.2 + 1.5 * x_resp))) + 0.08 * t * (x_upl > 0)
    y = (rng.random(n) < np.clip(p, 0, 1)).astype(int)
    sig = uplift_signal(pd.DataFrame({"x_resp": x_resp, "x_upl": x_upl}), t, y, ["x_resp", "x_upl"])
    assert abs(sig.loc["x_upl", "z_uplift"]) > 4
    assert abs(sig.loc["x_resp", "z_uplift"]) < abs(sig.loc["x_upl", "z_uplift"]) / 2
    assert sig.loc["x_resp", "rho_response"] > 0.3


def test_every_behaviour_feature_is_in_the_model_spec():
    assert set(ALL_BEHAVIOUR) <= set(FULL_SPEC.columns)


@pytest.mark.skipif(not FEATURES_PARQUET.exists(), reason="run scripts/build_features.py first")
def test_feature_table_sanity():
    f = pd.read_parquet(FEATURES_PARQUET)
    assert f["client_id"].is_unique and len(f) == 400_162
    shares = [c for c in ALL_BEHAVIOUR if c.endswith("_share") or c.startswith("share_")]
    assert ((f[shares] >= 0) & (f[shares] <= 1) | f[shares].isna()).all().all()
    # every client bought within the history window, so 30/60-day windows are never empty
    assert f["trx_30d"].min() >= 1 and (f["trx_7d"] <= f["trx_14d"]).all() and (f["trx_14d"] <= f["trx_30d"]).all()
    assert f["spend_7d"].notna().all() and f["days_since_pts_spent"].notna().all()
