"""HTTP service around the trained uplift model.

    GET  /health          model version, test metrics (no auth)
    POST /score           uplift per client
    POST /plan            who to contact for given economics and budget, with expected profit
    POST /score/batch     CSV in, CSV out

Run locally:
    uvicorn uplift.api:app --app-dir src --port 8000
Set UPLIFT_API_KEY to require the `X-API-Key` header on every endpoint except /health.

Inputs are the client-level features of `data/processed/customers.parquet` (pre-campaign
attributes and purchase aggregates, as a CRM would export them). Flags derived from them
(invalid age, has redeemed points) are computed here so callers cannot get them wrong.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import artifact
from .business import Economics, holdout_flags, minimum_detectable_effect, plan_contacts
from .monitoring import drift_report
from .synthetic import synthetic_clients
from .data import AGE_MAX, AGE_MIN

MAX_CLIENTS_JSON = 10_000
MAX_CLIENTS_CSV = 200_000


class Client(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "client_id": "c-001", "age": 64, "gender": "F", "card_age_days": 200, "days_since_first_redeem": 150,
        "n_transactions": 8, "total_spend": 2200, "avg_basket": 275, "avg_items": 5.5, "n_stores": 2,
        "points_received": 25, "points_spent": 20, "days_since_last_purchase": 8, "days_since_first_purchase": 110,
    }]})

    client_id: str | None = Field(None, max_length=64)
    age: float | None = Field(None, description="years; values outside 14–100 are treated as unknown")
    gender: Literal["F", "M", "U"] = "U"
    card_age_days: int = Field(..., ge=0, le=20_000, description="days since the loyalty card was issued")
    days_since_first_redeem: int | None = Field(None, ge=0, le=20_000, description="null if points were never redeemed")
    n_transactions: int = Field(..., ge=1, le=100_000, description="purchases in the last ~4 months")
    total_spend: float = Field(..., ge=0)
    avg_basket: float = Field(..., ge=0)
    avg_items: float = Field(..., ge=0)
    n_stores: int = Field(..., ge=1)
    points_received: float = Field(0, ge=0)
    points_spent: float = Field(0, ge=0)
    days_since_last_purchase: int = Field(..., ge=0, le=10_000)
    days_since_first_purchase: int = Field(..., ge=0, le=10_000)


class EconomicsIn(BaseModel):
    value: float = Field(100, gt=0, description="margin of one extra purchase")
    cost_per_contact: float = Field(2, ge=0, description="price of one SMS")
    discount: float = Field(0, ge=0, description="cost of an offer redeemed by every contacted buyer")


class ScoreRequest(BaseModel):
    clients: list[Client] = Field(..., min_length=1, max_length=MAX_CLIENTS_JSON)


class PlanRequest(ScoreRequest):
    economics: EconomicsIn = EconomicsIn()
    budget: float | None = Field(None, ge=0, description="cap on contact + offer spend")
    holdout_share: float = Field(0.0, ge=0, le=0.5, description=(
        "share of the selected clients kept as an untouched random control group, so the next campaign "
        "measures its real effect; assignment is deterministic per client_id"))


class ClientScore(BaseModel):
    client_id: str | None
    expected_effect_pp: float = Field(description="calibrated change in purchase probability, percentage points")
    raw_uplift_pp: float
    p_buy_if_contacted: float
    p_buy_if_not_contacted: float
    tier: Literal["high", "medium", "low"]


class PlanItem(BaseModel):
    client_id: str | None
    group: Literal["contact", "holdout", "skip"]
    contact: bool
    expected_effect_pp: float
    expected_profit: float


class PlanSummary(BaseModel):
    clients: int
    selected: int = Field(description="clients worth contacting (contact + holdout)")
    contacted: int
    holdout: int
    expected_extra_purchases: float
    expected_spend: float
    expected_profit: float
    profit_if_everyone_contacted: float
    profit_forgone_by_holdout: float
    minimum_detectable_effect_pp: float | None = Field(
        description="smallest effect the contacted-vs-holdout comparison can detect (80% power, 5% level)")


class PlanResponse(BaseModel):
    summary: PlanSummary
    clients: list[PlanItem]


# --------------------------------------------------------------------------- model wrapper

class Scorer:
    def __init__(self, model_path: Path = artifact.MODEL_PATH, meta_path: Path = artifact.META_PATH):
        self.model = artifact.load(model_path)
        self.meta = artifact.load_meta(meta_path)
        cal = self.meta["calibration"]
        self._cal_x, self._cal_y = np.asarray(cal["x"]), np.asarray(cal["y"])
        self.tiers = self.meta["tier_thresholds"]

    OPTIONAL_DEFAULTS = {"client_id": None, "age": np.nan, "gender": "U", "days_since_first_redeem": np.nan,
                         "points_received": 0.0, "points_spent": 0.0}

    @classmethod
    def to_frame(cls, clients: pd.DataFrame) -> pd.DataFrame:
        df = clients.copy()
        for col, default in cls.OPTIONAL_DEFAULTS.items():
            if col not in df:
                df[col] = default
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["days_since_first_redeem"] = pd.to_numeric(df["days_since_first_redeem"], errors="coerce")
        df["age_invalid"] = ~df["age"].between(AGE_MIN, AGE_MAX)
        df.loc[df["age_invalid"], "age"] = np.nan
        df["redeemed_before_campaign"] = df["days_since_first_redeem"].notna()
        df["redeem_before_issue"] = df["days_since_first_redeem"] > df["card_age_days"]
        return df

    def score(self, clients: pd.DataFrame) -> pd.DataFrame:
        X = self.to_frame(clients)
        p1, p0 = self.model.predict_arms(X)
        raw = p1 - p0
        effect = np.interp(raw, self._cal_x, self._cal_y)   # clipped at both ends
        tier = np.where(raw >= self.tiers["high"], "high", np.where(raw >= self.tiers["medium"], "medium", "low"))
        return pd.DataFrame({
            "client_id": X["client_id"] if "client_id" in X else None,
            "expected_effect_pp": effect * 100, "raw_uplift_pp": raw * 100,
            "p_buy_if_contacted": p1, "p_buy_if_not_contacted": p0, "tier": tier,
        }, index=X.index)


@lru_cache
def get_scorer() -> Scorer:
    return Scorer()


def require_key(x_api_key: str | None = Header(None)) -> None:
    expected = os.environ.get("UPLIFT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


# --------------------------------------------------------------------------- app

app = FastAPI(
    title="Uplift targeting API",
    version="1.0.0",
    description="Who should receive a campaign: the causal effect of contacting each client, and the "
                "profit-maximising contact list for given costs.",
)


def _frame(clients: list[Client]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in clients])


@app.get("/health")
def health(scorer: Scorer = Depends(get_scorer)) -> dict:
    m = scorer.meta
    return {"status": "ok", "model": m["model_class"], "created_utc": m["created_utc"],
            "trained_on": m["trained_on"], "test_metrics": m["test_metrics"]}


@app.post("/score", response_model=list[ClientScore], dependencies=[Depends(require_key)])
def score(req: ScoreRequest, scorer: Scorer = Depends(get_scorer)):
    return scorer.score(_frame(req.clients)).replace({np.nan: None}).to_dict(orient="records")


@app.post("/plan", response_model=PlanResponse, dependencies=[Depends(require_key)])
def plan(req: PlanRequest, scorer: Scorer = Depends(get_scorer)):
    clients = _frame(req.clients)
    s = scorer.score(clients)
    econ = Economics(**req.economics.model_dump())
    p = plan_contacts(s[["client_id"]], s["expected_effect_pp"] / 100, econ, budget=req.budget,
                      p_buy_contacted=s["p_buy_if_contacted"])
    ids = [c if c is not None else f"row-{i}" for i, c in zip(p.index, p["client_id"])]
    hold = p["contact"].to_numpy() & holdout_flags(ids, req.holdout_share)
    p["group"] = np.where(hold, "holdout", np.where(p["contact"], "contact", "skip"))
    contacted, held = p[p.group == "contact"], p[p.group == "holdout"]
    everyone = (s["expected_effect_pp"] / 100 * econ.value - econ.cost_per_contact
                - s["p_buy_if_contacted"] * econ.discount).sum()
    mde = minimum_detectable_effect(len(contacted), len(held), scorer.meta.get("base_rate_without_contact", 0.6))
    return {
        "summary": {
            "clients": len(p),
            "selected": int(len(contacted) + len(held)),
            "contacted": int(len(contacted)),
            "holdout": int(len(held)),
            "expected_extra_purchases": round(float(contacted["expected_effect"].sum()), 3),
            "expected_spend": round(float(contacted["expected_cost"].sum()), 2),
            "expected_profit": round(float(contacted["expected_profit"].sum()), 2),
            "profit_if_everyone_contacted": round(float(everyone), 2),
            "profit_forgone_by_holdout": round(float(held["expected_profit"].sum()), 2),
            "minimum_detectable_effect_pp": None if not np.isfinite(mde) else round(mde * 100, 2),
        },
        "clients": [
            {"client_id": r.client_id, "group": r.group, "contact": r.group == "contact",
             "expected_effect_pp": round(r.expected_effect * 100, 3), "expected_profit": round(r.expected_profit, 3)}
            for r in p.itertuples()
        ],
    }


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001 - any parse failure is the caller's input problem
        raise HTTPException(status_code=400, detail=f"could not read CSV: {e}") from e
    if len(df) > MAX_CLIENTS_CSV:
        raise HTTPException(status_code=413, detail=f"at most {MAX_CLIENTS_CSV:,} rows per file")
    return df


@app.post("/drift", dependencies=[Depends(require_key)])
async def drift(file: UploadFile = File(..., description="CSV of clients about to be scored"),
                scorer: Scorer = Depends(get_scorer)) -> dict:
    """Population Stability Index of each input against the training population."""
    df = Scorer.to_frame(await _read_csv(file))
    rep = drift_report(df, scorer.meta["reference_profile"])
    worst = rep["status"].map({"drift": 2, "watch": 1, "stable": 0}).max()
    return {"clients": len(df), "overall": {2: "drift", 1: "watch", 0: "stable"}.get(worst, "unknown"),
            "features": rep.replace({np.nan: None}).to_dict(orient="records")}


@app.get("/demo/sample", response_model=list[Client])
def demo_sample(n: int = 1000, seed: int = 0, scorer: Scorer = Depends(get_scorer)):
    """Synthetic clients (never real rows) drawn from the aggregated training profile, for the demo page."""
    n = max(1, min(n, MAX_CLIENTS_JSON))
    df = synthetic_clients(scorer.meta["reference_profile"], n=n, seed=seed)
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/score/batch", dependencies=[Depends(require_key)])
async def score_batch(file: UploadFile = File(..., description="CSV with the /score fields as columns"),
                      scorer: Scorer = Depends(get_scorer)):
    df = await _read_csv(file)
    # validate every row with the same schema as the JSON endpoint
    errors = []
    for i, row in enumerate(df.replace({np.nan: None}).to_dict(orient="records")):
        try:
            # extra columns (names, segments, ...) are allowed in the file and passed through untouched
            Client.model_validate({k: row[k] for k in Client.model_fields if row.get(k) is not None})
        except Exception as e:  # noqa: BLE001
            errors.append(f"row {i + 2}: {str(e).splitlines()[0]}")
            if len(errors) >= 20:
                break
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    out = pd.concat([df, scorer.score(df).drop(columns="client_id")], axis=1)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=uplift_scores.csv"})
