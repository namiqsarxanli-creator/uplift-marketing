"""Baseline targeting strategies and uplift meta-learners on a shared scikit-learn Pipeline.

Every model exposes `fit(X, treatment, y)` and `predict_uplift(X)`; a higher score means
"contact this client first". The response model is included on purpose: it is what most
marketing teams use and the main thing uplift modelling has to beat.

Which columns are used and how they are transformed is described by a `FeatureSpec`, so the
same model code runs on the step-2 base table, on the step-4 behavioural features, or on
any subset of feature groups (ablations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import FEATURE_GROUPS


@dataclass(frozen=True)
class FeatureSpec:
    log: tuple[str, ...] = ()    # skewed non-negative counts/amounts -> log1p
    num: tuple[str, ...] = ()    # other numeric
    bool: tuple[str, ...] = ()
    cat: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return [*self.log, *self.num, *self.bool, *self.cat]

    def __add__(self, other: "FeatureSpec") -> "FeatureSpec":
        def merge(a, b):
            return tuple(dict.fromkeys([*a, *b]))
        return FeatureSpec(merge(self.log, other.log), merge(self.num, other.num),
                           merge(self.bool, other.bool), merge(self.cat, other.cat))


# step 2: client attributes + basic activity aggregates
BASE_SPEC = FeatureSpec(
    log=("n_transactions", "total_spend", "avg_basket", "avg_items", "n_stores",
         "points_received", "points_spent"),
    num=("age", "card_age_days", "days_since_first_redeem",
         "days_since_last_purchase", "days_since_first_purchase"),
    bool=("age_invalid", "redeemed_before_campaign", "redeem_before_issue"),
    cat=("gender",),
)
CLIENT_SPEC = FeatureSpec(
    num=("age", "card_age_days", "days_since_first_redeem", "days_since_last_purchase",
         "days_since_first_purchase"),
    bool=("age_invalid", "redeemed_before_campaign", "redeem_before_issue"),
    cat=("gender",),
)
_NOT_LOG = {"pts_balance_proxy", "hour_mean", "days_since_pts_spent", "basket_trend_30_vs_all"}


def group_spec(group: str) -> FeatureSpec:
    cols = FEATURE_GROUPS[group]
    is_share = lambda c: c.endswith("_share") or c.startswith("share_")  # noqa: E731
    return FeatureSpec(
        log=tuple(c for c in cols if c not in _NOT_LOG and not is_share(c)),
        num=tuple(c for c in cols if c in _NOT_LOG or is_share(c)),
    )


def spec_for(groups) -> FeatureSpec:
    """Client attributes + the selected behavioural groups (step 4)."""
    spec = CLIENT_SPEC
    for g in groups:
        spec = spec + group_spec(g)
    return spec


FULL_SPEC = spec_for(FEATURE_GROUPS)


def log1p_clip(a):
    """Module-level (not a lambda) so the fitted Pipeline can be pickled for the API."""
    return np.log1p(np.clip(a, 0, None))


def make_preprocessor(spec: FeatureSpec, splines: bool = False) -> ColumnTransformer:
    """Same preparation in training, evaluation and the API (it lives inside the Pipeline).

    `splines=True` expands every numeric column into a cubic B-spline basis, so a linear model
    can follow non-monotonic shapes (e.g. the SMS effect peaks 8–14 days after the last purchase)."""
    from sklearn.preprocessing import SplineTransformer

    def tail():
        return [SplineTransformer(n_knots=5, degree=3, knots="quantile")] if splines else [StandardScaler()]

    parts = []
    if spec.log:
        parts.append(("log", make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(log1p_clip, feature_names_out="one-to-one"),
            *tail()), list(spec.log)))
    if spec.num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), *tail()), list(spec.num)))
    if spec.log or spec.num:
        from sklearn.impute import MissingIndicator
        parts.append(("missing", MissingIndicator(features="missing-only", error_on_new=False),
                      [*spec.log, *spec.num]))
    if spec.bool:
        parts.append(("bool", "passthrough", list(spec.bool)))
    if spec.cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(spec.cat)))
    return ColumnTransformer(parts)


DEFAULT_PARAMS = {
    "logistic": dict(max_iter=3000, C=1.0),
    "boosting": dict(max_iter=300, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0,
                     random_state=0),
    "ridge": dict(alpha=1.0),
    "boosting_reg": dict(max_iter=200, learning_rate=0.05, min_samples_leaf=1000, max_depth=3,
                         l2_regularization=1.0, random_state=0),
}


def make_learner(kind: str, spec: FeatureSpec, **params) -> Pipeline:
    """Classifier (`logistic`, `boosting`) or regressor (`ridge`, `boosting_reg`) behind the shared preprocessing."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    splines = kind.endswith("_spline")
    base = kind.removesuffix("_spline")
    est = {"logistic": LogisticRegression, "boosting": HistGradientBoostingClassifier,
           "ridge": Ridge, "boosting_reg": HistGradientBoostingRegressor}[base]
    return Pipeline([("prep", make_preprocessor(spec, splines=splines)),
                     ("clf", est(**{**DEFAULT_PARAMS[base], **params}))])


def _prepare(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    X = X[spec.columns].copy()
    for c in spec.bool:
        X[c] = X[c].astype(float)
    for c in spec.cat:
        X[c] = X[c].astype(str)
    return X


class _Base:
    name = "base"

    def __init__(self, spec: FeatureSpec = BASE_SPEC, learner: str = "logistic", params: dict | None = None):
        self.spec, self.learner, self.params = spec, learner, params or {}

    def _new(self, spec=None) -> Pipeline:
        return make_learner(self.learner, spec or self.spec, **self.params)


class RandomTargeting(_Base):
    name = "random"

    def fit(self, X, treatment, y):
        return self

    def predict_uplift(self, X):
        return np.random.default_rng(0).random(len(X))


class ResponseModel(_Base):
    """Classic approach: P(buy | contacted), trained on last campaign's contacted clients."""
    name = "response model"

    def fit(self, X, treatment, y):
        t = np.asarray(treatment) == 1
        self.model_ = self._new().fit(_prepare(X, self.spec)[t], np.asarray(y)[t])
        return self

    def predict_uplift(self, X):
        return self.model_.predict_proba(_prepare(X, self.spec))[:, 1]


class SLearner(_Base):
    """One model with treatment as a feature; uplift = P(y | x, t=1) - P(y | x, t=0)."""
    name = "S-learner"

    def fit(self, X, treatment, y):
        spec = self.spec + FeatureSpec(bool=("treatment",))
        self._spec_t = spec
        Xp = _prepare(X.assign(treatment=np.asarray(treatment)), spec)
        self.model_ = self._new(spec).fit(Xp, y)
        return self

    def predict_uplift(self, X):
        p1 = self.model_.predict_proba(_prepare(X.assign(treatment=1), self._spec_t))[:, 1]
        p0 = self.model_.predict_proba(_prepare(X.assign(treatment=0), self._spec_t))[:, 1]
        return p1 - p0


class TLearner(_Base):
    """Separate models for treated and control; uplift = difference of their predictions."""
    name = "T-learner"

    def fit(self, X, treatment, y):
        Xp, t, y = _prepare(X, self.spec), np.asarray(treatment) == 1, np.asarray(y)
        self.m1_ = self._new().fit(Xp[t], y[t])
        self.m0_ = self._new().fit(Xp[~t], y[~t])
        return self

    def predict_uplift(self, X):
        Xp = _prepare(X, self.spec)
        return self.m1_.predict_proba(Xp)[:, 1] - self.m0_.predict_proba(Xp)[:, 1]

    def predict_arms(self, X) -> tuple[np.ndarray, np.ndarray]:
        """P(buy | contacted), P(buy | not contacted): needed to price offers redeemed by every buyer."""
        Xp = _prepare(X, self.spec)
        return self.m1_.predict_proba(Xp)[:, 1], self.m0_.predict_proba(Xp)[:, 1]


class ClassTransformation(_Base):
    """Jaskowski & Jaroszewicz (2012): with 50/50 randomisation, uplift = 2·P(z=1|x) - 1,
    where z = 1 for (treated, bought) and (control, did not buy)."""
    name = "class transformation"

    def fit(self, X, treatment, y):
        z = (np.asarray(y) == np.asarray(treatment)).astype(int)
        self.model_ = self._new().fit(_prepare(X, self.spec), z)
        return self

    def predict_uplift(self, X):
        return 2 * self.model_.predict_proba(_prepare(X, self.spec))[:, 1] - 1


BASELINES = [RandomTargeting, ResponseModel, SLearner, TLearner, ClassTransformation]


class XLearner(_Base):
    """Künzel et al. (2019). Stage 1: outcome models per arm. Stage 2: regress the imputed
    individual effects (treated: y − μ0(x); control: μ1(x) − y) and blend the two effect models
    with the treatment share. Works well when one arm's response surface is simpler."""
    name = "X-learner"

    def __init__(self, spec=BASE_SPEC, learner="logistic", params=None,
                 effect_learner: str = "ridge", effect_params: dict | None = None):
        super().__init__(spec, learner, params)
        self.effect_learner, self.effect_params = effect_learner, effect_params or {}

    def fit(self, X, treatment, y):
        Xp, t, y = _prepare(X, self.spec), np.asarray(treatment) == 1, np.asarray(y, float)
        m1, m0 = self._new().fit(Xp[t], y[t]), self._new().fit(Xp[~t], y[~t])
        d1 = y[t] - m0.predict_proba(Xp[t])[:, 1]
        d0 = m1.predict_proba(Xp[~t])[:, 1] - y[~t]
        reg = lambda: make_learner(self.effect_learner, self.spec, **self.effect_params)  # noqa: E731
        self.tau1_, self.tau0_ = reg().fit(Xp[t], d1), reg().fit(Xp[~t], d0)
        self.g_ = t.mean()
        return self

    def predict_uplift(self, X):
        Xp = _prepare(X, self.spec)
        # weight the effect model fitted on the *other* arm's imputations by the arm share (Künzel's g = e(x))
        return self.g_ * self.tau0_.predict(Xp) + (1 - self.g_) * self.tau1_.predict(Xp)


class DRLearner(_Base):
    """Doubly-robust learner (Kennedy 2020) with 2-fold cross-fitting. In a randomised campaign the
    propensity is known (the treated share), so only the outcome models are estimated:
        φ = μ1 − μ0 + t·(y − μ1)/p − (1 − t)·(y − μ0)/(1 − p),   E[φ | x] = uplift(x)
    and φ is regressed on x."""
    name = "DR-learner"

    def __init__(self, spec=BASE_SPEC, learner="logistic", params=None,
                 effect_learner: str = "ridge", effect_params: dict | None = None, seed: int = 0):
        super().__init__(spec, learner, params)
        self.effect_learner, self.effect_params, self.seed = effect_learner, effect_params or {}, seed

    def fit(self, X, treatment, y):
        from sklearn.model_selection import KFold

        Xp, t, y = _prepare(X, self.spec), np.asarray(treatment), np.asarray(y, float)
        p = t.mean()
        mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
        for a, b in KFold(2, shuffle=True, random_state=self.seed).split(Xp):
            ta = t[a] == 1
            mu1[b] = self._new().fit(Xp.iloc[a][ta], y[a][ta]).predict_proba(Xp.iloc[b])[:, 1]
            mu0[b] = self._new().fit(Xp.iloc[a][~ta], y[a][~ta]).predict_proba(Xp.iloc[b])[:, 1]
        phi = mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)
        self.tau_ = make_learner(self.effect_learner, self.spec, **self.effect_params).fit(Xp, phi)
        return self

    def predict_uplift(self, X):
        return self.tau_.predict(_prepare(X, self.spec))


class CausalForestModel(_Base):
    """Generalised random forest (Athey, Tibshirani & Wager 2019) via econml.grf.CausalForest:
    splits are chosen to maximise heterogeneity of the treatment effect itself."""
    name = "causal forest"

    def __init__(self, spec=BASE_SPEC, n_estimators: int = 300, min_samples_leaf: int = 200,
                 max_depth: int | None = None, max_samples: float = 0.45, seed: int = 0):
        super().__init__(spec)
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                       max_samples=max_samples, random_state=seed, n_jobs=-1)

    def fit(self, X, treatment, y):
        from econml.grf import CausalForest

        self.prep_ = make_preprocessor(self.spec)
        Xp = self.prep_.fit_transform(_prepare(X, self.spec))
        self.cf_ = CausalForest(**self.kw).fit(Xp, np.asarray(treatment, float), np.asarray(y, float))
        return self

    def predict_uplift(self, X):
        return self.cf_.predict(self.prep_.transform(_prepare(X, self.spec))).ravel()


class RankEnsemble:
    """Average of percentile ranks of several uplift models (scale-free, robust to calibration)."""

    def __init__(self, makers: list, name: str = "rank ensemble"):
        self.makers, self.name = makers, name

    def fit(self, X, treatment, y):
        self.models_ = [m().fit(X, treatment, y) for m in self.makers]
        return self

    def predict_uplift(self, X):
        ranks = [pd.Series(m.predict_uplift(X)).rank(pct=True).to_numpy() for m in self.models_]
        return np.mean(ranks, axis=0)


def selected_model() -> TLearner:
    """Step-5 choice (notebook 05): logistic T-learner on the step-2 table.

    No flexible learner, ensemble or larger feature set beat it beyond fold-to-fold noise, and it is
    the simplest, fastest and most explainable option, so it goes to the test evaluation and the API."""
    return TLearner(spec=BASE_SPEC, learner="logistic", params={"C": 1.0})


def uplift_signal(X: pd.DataFrame, treatment, y, columns) -> pd.DataFrame:
    """Per-feature screening statistic for *uplift*, not for response.

    Transformed outcome (Athey & Imbens): with treatment probability p,
    Z = y·(t/p − (1−t)/(1−p)) has E[Z | x] = uplift(x). The Spearman correlation of a
    feature with Z (and its z-statistic) measures whether the effect changes along the
    feature; the correlation with y measures plain purchase propensity, for contrast.
    """
    from scipy.stats import spearmanr

    t, y = np.asarray(treatment, float), np.asarray(y, float)
    p = t.mean()
    z_out = y * (t / p - (1 - t) / (1 - p))
    rows = []
    for c in columns:
        x = X[c].astype(float).to_numpy()
        m = ~np.isnan(x)
        if m.sum() < 100 or np.nanstd(x) == 0:
            continue
        r = spearmanr(x[m], z_out[m]).statistic
        rows.append({
            "feature": c,
            "rho_uplift": r,
            "z_uplift": r * np.sqrt(m.sum() - 2) / np.sqrt(1 - r * r),
            "rho_response": spearmanr(x[m], y[m]).statistic,
        })
    return pd.DataFrame(rows).set_index("feature")


class UpliftScreened(_Base):
    """Keep client attributes, add only behavioural features whose |z_uplift| ≥ threshold
    on the *training* data, then fit an inner meta-learner. Screening happens inside `fit`,
    so cross-validation never sees the validation fold during selection."""

    def __init__(self, inner=TLearner, candidates: FeatureSpec = FULL_SPEC, threshold: float = 2.0,
                 learner: str = "logistic", spec: FeatureSpec = CLIENT_SPEC):
        super().__init__(spec=spec, learner=learner)
        self.inner, self.candidates, self.threshold = inner, candidates, threshold
        self.name = f"{inner.name} + uplift screening"

    def fit(self, X, treatment, y):
        pool = [c for c in (*self.candidates.log, *self.candidates.num) if c not in self.spec.columns]
        sig = uplift_signal(X, treatment, y, pool)
        self.selected_ = sig.index[sig["z_uplift"].abs() >= self.threshold].tolist()
        extra = FeatureSpec(log=tuple(c for c in self.candidates.log if c in self.selected_),
                            num=tuple(c for c in self.candidates.num if c in self.selected_))
        self.model_ = self.inner(spec=self.spec + extra, learner=self.learner).fit(X, treatment, y)
        return self

    def predict_uplift(self, X):
        return self.model_.predict_uplift(X)


def fit_score_all(train: pd.DataFrame, valid: pd.DataFrame, models=BASELINES,
                  spec: FeatureSpec = BASE_SPEC, learner: str = "logistic") -> dict[str, np.ndarray]:
    scores = {}
    for cls in models:
        m = cls(spec=spec, learner=learner).fit(train, train["treatment_flg"], train["target"])
        scores[m.name] = m.predict_uplift(valid)
    return scores
