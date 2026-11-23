"""From uplift scores to money: decile tables, profit curves and the budget-optimal contact list.

Economics of one contact (all amounts in the same currency, per client):
    profit = value · P(extra purchase caused by the contact)
             − cost_per_contact
             − discount · P(contacted client buys)          # e.g. a coupon redeemed by every buyer
The last term is why contacting "sure things" is expensive: they redeem the discount without
changing behaviour. The dataset contains no prices, so value/cost/discount are scenario inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    value: float = 100.0            # margin of one extra purchase
    cost_per_contact: float = 2.0   # SMS price
    discount: float = 0.0           # cost of an offer redeemed by each contacted buyer

    def profit_per_contacted(self, rate_treated: float, rate_control: float) -> float:
        return (rate_treated - rate_control) * self.value - self.cost_per_contact - rate_treated * self.discount


def _order(score):
    return np.argsort(-np.asarray(score, float), kind="stable")


def decile_table(score, treatment, target, predicted_uplift=None, n_bins: int = 10) -> pd.DataFrame:
    """Observed effect per score decile (1 = highest score) with 95% normal-approximation intervals."""
    o = _order(score)
    t, y = np.asarray(treatment)[o], np.asarray(target)[o]
    pred = None if predicted_uplift is None else np.asarray(predicted_uplift)[o]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(t)), n_bins), start=1):
        tt, yy = t[idx], y[idx]
        p1, p0 = yy[tt == 1].mean(), yy[tt == 0].mean()
        se = np.sqrt(p1 * (1 - p1) / (tt == 1).sum() + p0 * (1 - p0) / (tt == 0).sum())
        row = {"decile": i, "clients": len(idx), "rate_without": p0, "rate_with": p1,
               "observed_uplift": p1 - p0, "ci_low": p1 - p0 - 1.96 * se, "ci_high": p1 - p0 + 1.96 * se}
        if pred is not None:
            row["predicted_uplift"] = pred[idx].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("decile")


def profit_curve(score, treatment, target, econ: Economics, n_points: int = 100) -> pd.DataFrame:
    """Profit per 1,000 clients in the base when contacting the top-k share by `score`.

    Rates inside the top-k slice come from the randomised treated/control split of that slice,
    so the curve is an unbiased estimate of what the targeting policy would have earned.
    """
    o = _order(score)
    t, y = np.asarray(treatment)[o], np.asarray(target)[o]
    n = len(t)
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    rows = [{"share_contacted": 0.0, "rate_with": np.nan, "rate_without": np.nan,
             "profit_per_1000": 0.0, "extra_purchases_per_1000": 0.0}]
    for i in np.unique(np.linspace(n * 0.01, n, n_points).astype(int)) - 1:
        if cum_t[i] < 30 or cum_c[i] < 30:
            continue
        rt, rc = cum_yt[i] / cum_t[i], cum_yc[i] / cum_c[i]
        k = (i + 1) / n
        rows.append({
            "share_contacted": k,
            "rate_with": rt,
            "rate_without": rc,
            "profit_per_1000": 1000 * k * econ.profit_per_contacted(rt, rc),
            "extra_purchases_per_1000": 1000 * k * (rt - rc),
        })
    return pd.DataFrame(rows)


def best_policy(curve: pd.DataFrame) -> pd.Series:
    """The contact share with the highest profit (contacting nobody is always an option)."""
    return curve.loc[curve["profit_per_1000"].idxmax()]


def bootstrap_best_share(score, treatment, target, econ: Economics, n_boot: int = 200, seed: int = 0) -> np.ndarray:
    """Distribution of the optimal contact share, to show how certain the recommendation is."""
    rng = np.random.default_rng(seed)
    s, t, y = np.asarray(score), np.asarray(treatment), np.asarray(target)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(t), len(t))
        out.append(best_policy(profit_curve(s[i], t[i], y[i], econ, n_points=50))["share_contacted"])
    return np.asarray(out)


def holdout_flags(client_ids, share: float, salt: str = "uplift-holdout") -> np.ndarray:
    """Deterministic random holdout: the same client always lands in the same group for a given salt,
    so a plan can be regenerated without reshuffling who was held out."""
    import hashlib

    if share <= 0:
        return np.zeros(len(client_ids), dtype=bool)
    u = [int(hashlib.sha256(f"{salt}:{c}".encode()).hexdigest()[:12], 16) / 16**12 for c in client_ids]
    return np.asarray(u) < share


def minimum_detectable_effect(n_contacted: int, n_holdout: int, base_rate: float,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true effect (absolute, in probability) the next campaign's contacted-vs-holdout
    comparison can detect with the given power (two-sided two-proportion z-test)."""
    from scipy.stats import norm

    if n_contacted == 0 or n_holdout == 0:
        return float("inf")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * np.sqrt(base_rate * (1 - base_rate) * (1 / n_contacted + 1 / n_holdout)))


def plan_contacts(clients: pd.DataFrame, predicted_uplift, econ: Economics,
                  budget: float | None = None, p_buy_contacted=None) -> pd.DataFrame:
    """Contact list for new clients, ranked by expected profit of contacting them.

    expected_profit = effect · value − cost − P(buy | contacted) · discount
    A client is contacted when that is positive; the optional budget caps the total spend on
    contacts and redeemed offers. `predicted_uplift` should be the *calibrated* effect
    (see `uplift.train`), so that money estimates match what step 6 observed.
    """
    out = clients[["client_id"]].copy() if "client_id" in clients else pd.DataFrame(index=clients.index)
    out["expected_effect"] = np.asarray(predicted_uplift, float)
    p = np.zeros(len(out)) if p_buy_contacted is None else np.asarray(p_buy_contacted, float)
    out["expected_cost"] = econ.cost_per_contact + p * econ.discount
    out["expected_profit"] = out["expected_effect"] * econ.value - out["expected_cost"]
    out = out.sort_values("expected_profit", ascending=False, kind="stable")
    out["contact"] = out["expected_profit"] > 0
    if budget is not None:
        spent = out["expected_cost"].where(out["contact"], 0).cumsum()
        out.loc[spent > budget, "contact"] = False
    return out
