"""Final training run: fit the selected model on train+valid, calibrate on the test split, save.

Calibration: the T-learner's raw uplift is well ordered but its magnitudes are compressed
(notebook 06). An isotonic map from predicted to observed effect, fitted on the test-split
deciles, turns a score into an expected effect for money calculations. Ranking is unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from . import artifact
from .business import decile_table
from .features import load_model_table
from .metrics import evaluate
from .models import selected_model
from .monitoring import reference_profile

TIERS = {"high": 0.90, "medium": 0.70}   # top 10% and next 20% of training-pool scores


def fit_calibration(predicted, treatment, target, n_bins: int = 10) -> dict:
    dec = decile_table(predicted, treatment, target, predicted_uplift=predicted, n_bins=n_bins)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(dec["predicted_uplift"], dec["observed_uplift"])
    xs = np.sort(dec["predicted_uplift"].to_numpy())
    return {"x": xs.round(6).tolist(), "y": iso.predict(xs).round(6).tolist()}


def train_and_save() -> dict:
    df = load_model_table()
    pool, test = df[df.split != "test"], df[df.split == "test"]
    model = selected_model().fit(pool, pool.treatment_flg, pool.target)

    t, y = test.treatment_flg.to_numpy(), test.target.to_numpy()
    u_test = model.predict_uplift(test)
    m = evaluate({"model": u_test}, t, y, n_boot=0).metrics.loc["model"]
    u_pool = model.predict_uplift(pool)

    meta = {
        "description": "Logistic T-learner predicting the SMS effect on purchase probability (X5 RetailHero).",
        "trained_on": f"{len(pool):,} clients (train + valid), campaign 2019-03",
        "test_metrics": {
            "qini_x1000": round(float(m["qini"]) * 1000, 3),
            "uplift_top10_pp": round(float(m["uplift@10%"]) * 100, 2),
            "uplift_top30_pp": round(float(m["uplift@30%"]) * 100, 2),
            "test_ate_pp": round(float(y[t == 1].mean() - y[t == 0].mean()) * 100, 2),
        },
        "calibration": fit_calibration(u_test, t, y),
        "tier_thresholds": {k: float(np.quantile(u_pool, q)) for k, q in TIERS.items()},
        "base_rate_without_contact": round(float(pool.loc[pool.treatment_flg == 0, "target"].mean()), 4),
        "reference_profile": reference_profile(
            pool, numeric=[*model.spec.log, *model.spec.num], categorical=list(model.spec.cat)),
    }
    artifact.save(model, meta)
    return meta
