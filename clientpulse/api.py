"""Real-time scoring API.

    uvicorn clientpulse.api:app --reload

POST /score   one client snapshot -> churn probability, expected value, reasons
GET  /model   which version is serving and how it scored at release
GET  /health  liveness for the load balancer
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import annual_revenue_ngn, build_features
from .model import expected_value, reason_codes
from .registry import Registry

app = FastAPI(title="ClientPulse churn scoring", version="1.0")


class ClientSnapshot(BaseModel):
    client_id: str = Field(examples=["MC100042"])
    segment: Literal["retail", "affluent", "hnwi"]
    channel: Literal["mobile_app", "web", "branch", "relationship_manager"]
    region: Literal["lagos", "abuja", "south_west", "south_south", "north", "east"]
    age: int = Field(ge=18, le=110)
    tenure_months: int = Field(ge=0)
    portfolio_value_ngn: float = Field(ge=0)
    cash_share: float = Field(ge=0, le=1)
    num_asset_classes: int = Field(ge=1, le=10)
    trades_90d: int = Field(ge=0)
    trade_trend: float = Field(ge=-1, le=1)
    logins_30d: int = Field(ge=0)
    days_since_login: int = Field(ge=0)
    net_flow_90d_share: float = Field(ge=-1, le=1)
    complaints_90d: int = Field(ge=0)
    failed_txn_90d: int = Field(ge=0)
    dividend_reinvest: int = Field(ge=0, le=1)
    kyc_complete: int = Field(ge=0, le=1)


class Reason(BaseModel):
    feature: str
    value: float | int | str
    impact: float


class Score(BaseModel):
    client_id: str
    churn_probability: float
    expected_value_ngn: float
    recommend_contact: bool
    reasons: list[Reason]
    model_version: str


@lru_cache(maxsize=1)
def _model():
    return Registry().load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    try:
        _, meta = _model()
    except LookupError as e:
        raise HTTPException(503, str(e)) from e
    return {k: meta[k] for k in ("version", "stage", "train_window", "test_window", "registered_at")} | {
        "test_metrics": meta["metrics"]["test"]
    }


@app.post("/score", response_model=Score)
def score(client: ClientSnapshot) -> Score:
    try:
        model, meta = _model()
    except LookupError as e:
        raise HTTPException(503, str(e)) from e
    df = pd.DataFrame([client.model_dump()])
    X = build_features(df)
    p = float(model.predict_proba(X)[:, 1][0])
    ev = float(expected_value(p, annual_revenue_ngn(df).to_numpy())[0])
    return Score(
        client_id=client.client_id,
        churn_probability=round(p, 4),
        expected_value_ngn=round(ev, 0),
        recommend_contact=ev > 0,
        reasons=reason_codes(model, X)[0],
        model_version=meta["version"],
    )
