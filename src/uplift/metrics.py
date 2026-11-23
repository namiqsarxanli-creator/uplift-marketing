"""Uplift evaluation metrics.

No customer is observed both with and without the campaign, so per-row accuracy does not
exist. Instead clients are ranked by predicted uplift and, inside each top-k slice, the
treated and control groups (randomised) are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sorted(score, treatment, target):
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return np.asarray(treatment)[order], np.asarray(target)[order]


def uplift_at_k(score, treatment, target, k: float) -> float:
    """Treated minus control purchase rate among the top `k` share of clients by score."""
    t, y = _sorted(score, treatment, target)
    n = max(int(round(len(t) * k)), 1)
    t, y = t[:n], y[:n]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return float("nan")
    return float(y[t == 1].mean() - y[t == 0].mean())


def qini_curve(score, treatment, target, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Incremental purchases gained by targeting the top fraction of clients.

    Q(k) = purchases_treated(k) - purchases_control(k) * n_treated(k) / n_control(k),
    divided by the total number of clients so the y-axis is "extra purchases per client".
    """
    t, y = _sorted(score, treatment, target)
    n = len(t)
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt[idx] - np.where(cum_c[idx] > 0, cum_yc[idx] * cum_t[idx] / cum_c[idx], 0.0)
    x = (idx + 1) / n
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], q / n])


def qini_coefficient(score, treatment, target) -> float:
    """Area between the model's Qini curve and the random-targeting line (higher is better, 0 = random)."""
    x, q = qini_curve(score, treatment, target)
    random_line = x * q[-1]
    return float(np.trapezoid(q - random_line, x))


def cv_evaluate(df: pd.DataFrame, make_model, n_splits: int = 5, seed: int = 0,
                ks=(0.1, 0.3)) -> pd.DataFrame:
    """K-fold evaluation on the train+valid pool (never the test split).

    One validation split of 40k clients gives Qini intervals as wide as the differences
    between models, so model/feature choices are made on the mean over folds.
    `make_model()` must return an object with fit(X, t, y) / predict_uplift(X).
    Returns one row per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    strata = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    rows = []
    for fold, (tr, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, strata)):
        a, b = df.iloc[tr], df.iloc[va]
        s = make_model().fit(a, a["treatment_flg"], a["target"]).predict_uplift(b)
        rows.append({"fold": fold, **_all_metrics(s, b["treatment_flg"].to_numpy(), b["target"].to_numpy(), ks)})
    return pd.DataFrame(rows)


@dataclass
class Report:
    metrics: pd.DataFrame       # model x metric: point estimate
    ci_low: pd.DataFrame
    ci_high: pd.DataFrame
    vs_reference: pd.DataFrame  # paired difference to the reference model, with CI


def _all_metrics(score, t, y, ks) -> dict[str, float]:
    out = {f"uplift@{int(k * 100)}%": uplift_at_k(score, t, y, k) for k in ks}
    out["qini"] = qini_coefficient(score, t, y)
    return out


def evaluate(scores: dict[str, np.ndarray], treatment, target, ks=(0.1, 0.3),
             n_boot: int = 200, reference: str | None = None, seed: int = 0) -> Report:
    """Point estimates plus paired bootstrap intervals.

    All models are evaluated on the same bootstrap resamples, so differences between models
    get their own (tighter) intervals: the effect is small and per-model intervals overlap.
    """
    t, y = np.asarray(treatment), np.asarray(target)
    names = list(scores)
    point = pd.DataFrame({m: _all_metrics(scores[m], t, y, ks) for m in names}).T

    if n_boot == 0:
        return Report(point, point * np.nan, point * np.nan, pd.DataFrame())

    rng = np.random.default_rng(seed)
    boots = {m: [] for m in names}
    for _ in range(n_boot):
        i = rng.integers(0, len(t), len(t))
        for m in names:
            boots[m].append(_all_metrics(np.asarray(scores[m])[i], t[i], y[i], ks))
    boots = {m: pd.DataFrame(v) for m, v in boots.items()}
    lo = pd.DataFrame({m: b.quantile(0.025) for m, b in boots.items()}).T
    hi = pd.DataFrame({m: b.quantile(0.975) for m, b in boots.items()}).T

    rows = {}
    if reference is not None:
        for m in names:
            d = boots[m] - boots[reference]
            for col in d.columns:
                rows.setdefault(m, {})[col] = (
                    f"{point.loc[m, col] - point.loc[reference, col]:+.4f} "
                    f"[{d[col].quantile(0.025):+.4f}, {d[col].quantile(0.975):+.4f}]"
                )
    return Report(point, lo, hi, pd.DataFrame(rows).T)
