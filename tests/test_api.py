import pytest
from fastapi.testclient import TestClient

from clientpulse import api
from clientpulse.data import TARGET, generate, split
from clientpulse.features import build_features
from clientpulse.model import fit_lgbm

CLIENT = {
    "client_id": "MC100042",
    "segment": "affluent",
    "channel": "mobile_app",
    "region": "lagos",
    "age": 41,
    "tenure_months": 30,
    "portfolio_value_ngn": 12_500_000,
    "cash_share": 0.6,
    "num_asset_classes": 1,
    "trades_90d": 1,
    "trade_trend": -0.7,
    "logins_30d": 0,
    "days_since_login": 64,
    "net_flow_90d_share": -0.3,
    "complaints_90d": 2,
    "failed_txn_90d": 1,
    "dividend_reinvest": 0,
    "kyc_complete": 1,
}


@pytest.fixture(scope="module")
def client():
    s = split(generate(n_clients=1000, seed=2))
    model = fit_lgbm(build_features(s["train"]), s["train"][TARGET], build_features(s["valid"]), s["valid"][TARGET])
    meta = {"version": "vtest", "stage": "production", "train_window": "", "test_window": "",
            "registered_at": "", "metrics": {"test": {}}}
    api._model.cache_clear()
    original = api._model
    api._model = lambda: (model, meta)
    yield TestClient(api.app)
    api._model = original


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_high_risk_client(client):
    r = client.post("/score", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "vtest"
    assert 0 < body["churn_probability"] < 1
    assert body["reasons"], "a risky client should come with reasons"


def test_rejects_invalid_input(client):
    bad = CLIENT | {"cash_share": 1.7}
    assert client.post("/score", json=bad).status_code == 422
