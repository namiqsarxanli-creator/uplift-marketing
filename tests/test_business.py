import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import artifact  # noqa: E402
from uplift.business import Economics, best_policy, decile_table, plan_contacts, profit_curve  # noqa: E402


def campaign(n=40_000, seed=0):
    """Top half: +20 pp effect; bottom half: no effect. Base purchase rate 50%."""
    rng = np.random.default_rng(seed)
    persuadable = np.arange(n) < n // 2
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.5 + 0.2 * persuadable * t).astype(int)
    return persuadable.astype(float), t, y


def test_profit_curve_starts_at_zero_and_matches_hand_calculation():
    score, t, y = campaign()
    econ = Economics(value=100, cost_per_contact=2, discount=0)
    c = profit_curve(score, t, y, econ)
    assert c.iloc[0].profit_per_1000 == 0
    half = c.iloc[(c.share_contacted - 0.5).abs().argmin()]
    # contacting the persuadable half: 0.5 * (20 pp * 100 - 2) * 1000 = 9,000 per 1,000 clients
    assert abs(half.profit_per_1000 - 9_000) < 900


def test_best_policy_stops_where_the_effect_ends():
    score, t, y = campaign()
    best = best_policy(profit_curve(score, t, y, Economics(100, 2, 0)))
    assert 0.4 <= best.share_contacted <= 0.6


def test_expensive_offer_means_contact_nobody():
    score, t, y = campaign()
    best = best_policy(profit_curve(score, t, y, Economics(value=100, cost_per_contact=2, discount=60)))
    assert best.share_contacted == 0 and best.profit_per_1000 == 0


def test_decile_table_recovers_effects():
    score, t, y = campaign()
    d = decile_table(score, t, y, n_bins=2)
    assert abs(d.loc[1, "observed_uplift"] - 0.2) < 0.03 and abs(d.loc[2, "observed_uplift"]) < 0.03


def test_plan_contacts_respects_value_and_budget():
    clients = pd.DataFrame({"client_id": list("abcd")})
    plan = plan_contacts(clients, [0.10, 0.05, 0.01, -0.02], Economics(100, 2, 0), budget=2)
    assert plan.client_id.tolist() == list("abcd")
    assert plan.contact.tolist() == [True, False, False, False]   # budget 2 = one SMS
    plan = plan_contacts(clients, [0.10, 0.05, 0.01, -0.02], Economics(100, 2, 0))
    assert plan.contact.tolist() == [True, True, False, False]    # 1 pp * 100 < 2


def test_plan_contacts_prices_the_coupon_for_sure_things():
    """Same effect, but client b buys anyway with 95% probability and would redeem the coupon."""
    clients = pd.DataFrame({"client_id": ["a", "b"]})
    # a: 6 − 2 − 0.30·10 = +1   b: 6 − 2 − 0.95·10 = −5.5
    plan = plan_contacts(clients, [0.06, 0.06], Economics(100, 2, 10), p_buy_contacted=[0.3, 0.95])
    assert plan.set_index("client_id").contact.to_dict() == {"a": True, "b": False}


@pytest.mark.skipif(not artifact.MODEL_PATH.exists(), reason="run notebooks/06 to train and save the model")
def test_saved_model_loads_and_scores():
    model, meta = artifact.load(), artifact.load_meta()
    row = {c: 0.0 for c in meta["features"]} | {"gender": "F", "age": 60.0}
    score = model.predict_uplift(pd.DataFrame([row]))
    assert score.shape == (1,) and np.isfinite(score).all()
    assert meta["test_metrics"]["qini_x1000"] > 0
