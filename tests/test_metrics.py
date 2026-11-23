import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.metrics import qini_coefficient, qini_curve, uplift_at_k  # noqa: E402


def synthetic(n=20_000, seed=0):
    """Half the clients respond to treatment (+40 pp), half do not react at all."""
    rng = np.random.default_rng(seed)
    persuadable = rng.random(n) < 0.5
    t = (rng.random(n) < 0.5).astype(int)
    base = 0.3
    p = base + 0.4 * persuadable * t
    y = (rng.random(n) < p).astype(int)
    return persuadable.astype(float), t, y


def test_uplift_at_k_finds_persuadables():
    true_uplift, t, y = synthetic()
    assert abs(uplift_at_k(true_uplift, t, y, 0.5) - 0.4) < 0.03
    assert abs(uplift_at_k(-true_uplift, t, y, 0.5) - 0.0) < 0.03


def test_uplift_at_full_population_is_ate():
    true_uplift, t, y = synthetic()
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert abs(uplift_at_k(true_uplift, t, y, 1.0) - ate) < 1e-12


def test_qini_orders_models():
    true_uplift, t, y = synthetic()
    rng = np.random.default_rng(1)
    good = qini_coefficient(true_uplift, t, y)
    rand = qini_coefficient(rng.random(len(t)), t, y)
    bad = qini_coefficient(-true_uplift, t, y)
    assert good > 0.02 > abs(rand)
    assert bad < 0


def test_qini_curve_ends_at_total_incremental():
    true_uplift, t, y = synthetic()
    x, q = qini_curve(true_uplift, t, y)
    expected = (y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1 - t).sum()) / len(t)
    assert x[0] == 0 and abs(x[-1] - 1) < 1e-9
    assert abs(q[-1] - expected) < 1e-9
