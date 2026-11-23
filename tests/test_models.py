"""Every uplift learner must recover a known heterogeneous effect on synthetic data,
and the selected model must survive pickling (it is what the API will load)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.metrics import qini_coefficient  # noqa: E402
from uplift.models import (ClassTransformation, DRLearner, FeatureSpec, ResponseModel,  # noqa: E402
                           TLearner, XLearner)

SPEC = FeatureSpec(log=("spend",), num=("x_upl",), cat=("g",))


@pytest.fixture(scope="module")
def data():
    """Purchases depend strongly on `spend`; the SMS effect depends only on `x_upl`."""
    rng = np.random.default_rng(0)
    n = 40_000
    df = pd.DataFrame({
        "spend": rng.lognormal(6, 1, n),
        "x_upl": rng.normal(size=n),
        "g": rng.choice(["F", "M", "U"], n),
        "treatment_flg": (rng.random(n) < 0.5).astype(int),
    })
    base = 1 / (1 + np.exp(-(np.log(df.spend) - 6)))
    df["true_uplift"] = 0.15 * (df.x_upl > 0)
    df["target"] = (rng.random(n) < np.clip(base + df.treatment_flg * df.true_uplift, 0, 1)).astype(int)
    return df.iloc[:30_000], df.iloc[30_000:]


@pytest.mark.parametrize("cls", [TLearner, ClassTransformation, XLearner, DRLearner])
def test_learners_find_the_persuadables(data, cls):
    train, test = data
    score = cls(spec=SPEC).fit(train, train.treatment_flg, train.target).predict_uplift(test)
    corr = np.corrcoef(score, test.true_uplift)[0, 1]
    assert corr > 0.5, f"{cls.name}: correlation with true uplift {corr:.2f}"


def test_response_model_ranks_by_purchase_not_by_uplift(data):
    train, test = data
    resp = ResponseModel(spec=SPEC).fit(train, train.treatment_flg, train.target).predict_uplift(test)
    upl = TLearner(spec=SPEC).fit(train, train.treatment_flg, train.target).predict_uplift(test)
    t, y = test.treatment_flg, test.target
    # Qini is in "extra purchases per client" units; here the uplift model gets ~0.008, the response model ~0.001
    assert qini_coefficient(upl, t, y) > qini_coefficient(resp, t, y) + 0.004


def test_fitted_model_pickles(data):
    train, test = data
    m = TLearner(spec=SPEC).fit(train, train.treatment_flg, train.target)
    restored = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(restored.predict_uplift(test), m.predict_uplift(test))
