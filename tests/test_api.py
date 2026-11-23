import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import artifact  # noqa: E402

pytestmark = pytest.mark.skipif(not artifact.MODEL_PATH.exists(), reason="run scripts/train_final.py first")

from fastapi.testclient import TestClient  # noqa: E402

from uplift.api import app  # noqa: E402

client = TestClient(app)

# median profiles of the model's top-10% and of everyone else (notebook 06)
PERSUADABLE = {"client_id": "p", "age": 64, "gender": "F", "card_age_days": 200, "days_since_first_redeem": 150,
               "n_transactions": 8, "total_spend": 2200, "avg_basket": 275, "avg_items": 5.5, "n_stores": 2,
               "points_received": 25, "points_spent": 20, "days_since_last_purchase": 8,
               "days_since_first_purchase": 110}
SURE_THING = {"client_id": "s", "age": 38, "gender": "M", "card_age_days": 600, "days_since_first_redeem": 500,
              "n_transactions": 40, "total_spend": 24000, "avg_basket": 600, "avg_items": 9, "n_stores": 4,
              "points_received": 250, "points_spent": 200, "days_since_last_purchase": 1,
              "days_since_first_purchase": 117}


def test_health_is_open_and_reports_metrics():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["test_metrics"]["qini_x1000"] > 0


def test_score_ranks_persuadable_above_sure_thing():
    r = client.post("/score", json={"clients": [PERSUADABLE, SURE_THING]})
    assert r.status_code == 200, r.text
    by_id = {c["client_id"]: c for c in r.json()}
    assert by_id["p"]["raw_uplift_pp"] > by_id["s"]["raw_uplift_pp"]
    assert by_id["s"]["p_buy_if_not_contacted"] > by_id["p"]["p_buy_if_not_contacted"]
    assert by_id["p"]["tier"] in {"high", "medium"}


def test_invalid_input_is_rejected():
    bad = PERSUADABLE | {"n_transactions": -3}
    assert client.post("/score", json={"clients": [bad]}).status_code == 422
    assert client.post("/score", json={"clients": [PERSUADABLE | {"unknown_field": 1}]}).status_code == 422
    assert client.post("/score", json={"clients": []}).status_code == 422


def test_impossible_age_is_treated_as_unknown():
    a = client.post("/score", json={"clients": [PERSUADABLE | {"age": 1901}]}).json()[0]
    b = client.post("/score", json={"clients": [PERSUADABLE | {"age": None}]}).json()[0]
    assert a["raw_uplift_pp"] == pytest.approx(b["raw_uplift_pp"])


def test_plan_contacts_persuadable_and_skips_sure_thing_when_offer_is_costly():
    body = {"clients": [PERSUADABLE, SURE_THING], "economics": {"value": 100, "cost_per_contact": 2, "discount": 10}}
    r = client.post("/plan", json=body)
    assert r.status_code == 200, r.text
    decision = {c["client_id"]: c["contact"] for c in r.json()["clients"]}
    assert decision == {"p": True, "s": False}
    s = r.json()["summary"]
    assert s["contacted"] == 1 and s["expected_profit"] > s["profit_if_everyone_contacted"]


def test_plan_respects_budget():
    body = {"clients": [PERSUADABLE | {"client_id": str(i)} for i in range(10)],
            "economics": {"value": 100, "cost_per_contact": 2, "discount": 0}, "budget": 6}
    assert client.post("/plan", json=body).json()["summary"]["contacted"] == 3


def test_batch_csv_roundtrip_keeps_extra_columns():
    df = pd.DataFrame([PERSUADABLE | {"segment": "x"}, SURE_THING | {"segment": "y"}]).drop(columns=["age"])
    r = client.post("/score/batch", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    out = pd.read_csv(io.StringIO(r.text))
    assert list(out["segment"]) == ["x", "y"] and "expected_effect_pp" in out


def test_batch_reports_bad_rows():
    df = pd.DataFrame([PERSUADABLE, SURE_THING | {"n_stores": 0}])
    r = client.post("/score/batch", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 422 and "row 3" in r.json()["detail"][0]


def test_plan_holdout_is_deterministic_and_reports_power():
    body = {"clients": [PERSUADABLE | {"client_id": f"c{i}"} for i in range(400)],
            "economics": {"value": 100, "cost_per_contact": 2, "discount": 0}, "holdout_share": 0.2}
    a, b = client.post("/plan", json=body).json(), client.post("/plan", json=body).json()
    assert [c["group"] for c in a["clients"]] == [c["group"] for c in b["clients"]]
    s = a["summary"]
    assert s["selected"] == 400 and s["contacted"] + s["holdout"] == 400
    assert 50 < s["holdout"] < 110                      # ~20% of 400
    assert 5 < s["minimum_detectable_effect_pp"] < 40   # small campaign -> large detectable effect


def test_drift_endpoint_flags_shifted_population():
    df = pd.DataFrame(client.get("/demo/sample?n=3000&seed=1").json())
    stable = client.post("/drift", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")}).json()
    df["age"] = df["age"] + 20
    shifted = client.post("/drift", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")}).json()
    status = {f["feature"]: f["status"] for f in shifted["features"]}
    assert status["age"] == "drift" and shifted["overall"] == "drift"
    assert {f["feature"]: f["status"] for f in stable["features"]}["age"] != "drift"


def test_demo_sample_is_valid_input_and_page_loads():
    for seed in range(5):  # response validation fails if any synthetic client breaks the input schema
        assert client.get(f"/demo/sample?n=5000&seed={seed}").status_code == 200
    sample = client.get("/demo/sample?n=200").json()
    assert len(sample) == 200 and all(c["client_id"].startswith("synthetic-") for c in sample)
    assert client.post("/score", json={"clients": sample}).status_code == 200
    page = client.get("/")
    assert page.status_code == 200 and "Uplift Campaign Planner" in page.text


CUSTOMERS = Path(__file__).resolve().parents[1] / "data" / "processed" / "customers.parquet"


@pytest.mark.skipif(not CUSTOMERS.exists(), reason="run scripts/build_dataset.py first")
def test_no_training_serving_skew():
    """The API derives flags from the raw fields; they must equal the training table's flags,
    otherwise the served scores silently differ from the evaluated ones."""
    from uplift.api import Scorer

    d = pd.read_parquet(CUSTOMERS)
    derived = ["age_invalid", "redeemed_before_campaign", "redeem_before_issue"]
    served = Scorer.to_frame(d.drop(columns=derived))
    for col in derived:
        assert (served[col].astype(bool).to_numpy() == d[col].astype(bool).to_numpy()).all(), col
    model = artifact.load()
    sample = d.sample(5_000, random_state=0)
    assert abs(model.predict_uplift(Scorer.to_frame(sample.drop(columns=derived))) - model.predict_uplift(sample)).max() < 1e-12


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("UPLIFT_API_KEY", "secret")
    assert client.post("/score", json={"clients": [PERSUADABLE]}).status_code == 401
    ok = client.post("/score", json={"clients": [PERSUADABLE]}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200
