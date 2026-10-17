"""Synthetic brokerage and wealth client data.

The real thing would come from the data warehouse (client master, trade
blotter, portal logs, CRM tickets). This generator mimics that shape so the
whole pipeline can run without touching client data.

Design:
- Each client has fixed attributes (segment, channel, age, tenure at start).
- Each month we take a snapshot of behaviour over the trailing window.
- Label: churned = client goes dormant or moves most assets out within the
  next 90 days.
- Months 2026-07 to 2026-09 simulate a market downturn plus a shift in what
  drives churn (service failures start to matter much more). This gives the
  monitoring layer real data drift and concept drift to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ["retail", "affluent", "hnwi"]
CHANNELS = ["mobile_app", "web", "branch", "relationship_manager"]
REGIONS = ["lagos", "abuja", "south_west", "south_south", "north", "east"]

TRAIN_MONTHS = pd.period_range("2025-01", "2025-12", freq="M")
VALID_MONTHS = pd.period_range("2026-01", "2026-03", freq="M")
TEST_MONTHS = pd.period_range("2026-04", "2026-06", freq="M")
DRIFT_MONTHS = pd.period_range("2026-07", "2026-09", freq="M")
ALL_MONTHS = TRAIN_MONTHS.append(VALID_MONTHS).append(TEST_MONTHS).append(DRIFT_MONTHS)

FEATURES_NUMERIC = [
    "age",
    "tenure_months",
    "portfolio_value_ngn",
    "cash_share",
    "num_asset_classes",
    "trades_90d",
    "trade_trend",
    "logins_30d",
    "days_since_login",
    "net_flow_90d_share",
    "complaints_90d",
    "failed_txn_90d",
    "dividend_reinvest",
    "kyc_complete",
]
FEATURES_CATEGORICAL = ["segment", "channel", "region"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
TARGET = "churned_90d"


def _clients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    segment = rng.choice(SEGMENTS, size=n, p=[0.78, 0.18, 0.04])
    channel = np.where(
        segment == "hnwi",
        rng.choice(CHANNELS, size=n, p=[0.15, 0.15, 0.10, 0.60]),
        rng.choice(CHANNELS, size=n, p=[0.55, 0.20, 0.20, 0.05]),
    )
    base_value = {
        "retail": (13.0, 1.1),   # median ~ N440k
        "affluent": (16.2, 0.7),  # median ~ N10.8m
        "hnwi": (18.6, 0.8),      # median ~ N120m
    }
    log_value = np.array([rng.normal(*base_value[s]) for s in segment])
    return pd.DataFrame(
        {
            "client_id": [f"MC{100000 + i}" for i in range(n)],
            "segment": segment,
            "channel": channel,
            "region": rng.choice(REGIONS, size=n, p=[0.48, 0.17, 0.12, 0.10, 0.07, 0.06]),
            "age0": rng.integers(21, 70, size=n),
            "tenure0": rng.gamma(2.0, 18.0, size=n).astype(int) + 1,
            "log_value0": log_value,
            "engagement": rng.normal(0, 1, size=n),  # latent, never observed
            "reinvest": rng.random(n) < 0.35,
            "kyc": rng.random(n) < 0.88,
        }
    )


def _snapshot(clients: pd.DataFrame, month_idx: int, period: pd.Period, rng: np.random.Generator,
              drift: bool) -> pd.DataFrame:
    n = len(clients)
    eng = clients["engagement"].to_numpy() + rng.normal(0, 0.45, n)
    seg = clients["segment"].to_numpy()
    chan = clients["channel"].to_numpy()

    # market regime: downturn in the drift window lowers activity and pushes outflows
    market = -0.9 if drift else 0.0

    log_value = clients["log_value0"].to_numpy() + rng.normal(0, 0.12, n) + (-0.18 if drift else 0.0)
    trades = rng.poisson(np.clip(np.exp(1.3 + 0.55 * eng + 0.35 * market + 0.25 * (seg != "retail")), 0.05, 60))
    trade_trend = np.clip(rng.normal(0.15 * eng + 0.30 * market, 0.55, n), -1, 1)
    logins = rng.poisson(np.clip(np.exp(1.6 + 0.6 * eng + 0.25 * (chan == "mobile_app")), 0.05, 90))
    days_since_login = np.where(
        logins > 0,
        rng.integers(0, 12, n),
        rng.integers(30, 180, n),
    )
    net_flow = np.clip(rng.normal(0.02 + 0.06 * eng + 0.08 * market, 0.15, n), -1, 1)
    complaints = rng.poisson(np.where(drift, 0.22, 0.12) + 0.10 * (eng < -1), n)
    failed = rng.poisson(np.where(drift, 0.35, 0.18) * np.where(chan == "mobile_app", 1.6, 1.0), n)
    cash_share = np.clip(rng.beta(2, 6, n) + 0.08 * (eng < -0.5) + (0.10 if drift else 0.0), 0, 1)
    n_assets = np.clip(rng.poisson(np.clip(1.2 + 0.4 * (seg != "retail") + 0.3 * eng, 0.1, None)), 1, 5)

    df = pd.DataFrame(
        {
            "client_id": clients["client_id"].to_numpy(),
            "snapshot_month": str(period),
            "segment": seg,
            "channel": chan,
            "region": clients["region"].to_numpy(),
            "age": clients["age0"].to_numpy() + month_idx // 12,
            "tenure_months": clients["tenure0"].to_numpy() + month_idx,
            "portfolio_value_ngn": np.round(np.exp(log_value), -2),
            "cash_share": np.round(cash_share, 3),
            "num_asset_classes": n_assets,
            "trades_90d": trades,
            "trade_trend": np.round(trade_trend, 3),
            "logins_30d": logins,
            "days_since_login": days_since_login,
            "net_flow_90d_share": np.round(net_flow, 3),
            "complaints_90d": complaints,
            "failed_txn_90d": failed,
            "dividend_reinvest": clients["reinvest"].astype(int).to_numpy(),
            "kyc_complete": clients["kyc"].astype(int).to_numpy(),
        }
    )

    # churn logit: mostly behavioural, with a few non-linear effects trees should find
    dormant = days_since_login >= 30
    outflow = net_flow < -0.05
    z = (
        -3.1
        - 0.35 * eng
        # interactions a linear model cannot express on its own
        + 1.3 * (dormant & outflow)
        + 0.9 * ((trade_trend < -0.4) & (cash_share > 0.35))
        + 0.8 * ((seg == "retail") & (log_value < 12.0) & (chan == "mobile_app"))
        + 0.6 * ((clients["age0"].to_numpy() > 58) & (n_assets == 1))
        - 0.7 * ((seg != "retail") & (chan == "relationship_manager") & ~dormant)
        + 0.012 * np.clip(days_since_login - 14, 0, None)
        - 3.0 * np.clip(net_flow, None, 0)  # outflows (net_flow < 0) raise risk
        + 0.9 * (cash_share > 0.55)
        - 0.25 * (n_assets - 1)
        - 0.30 * clients["reinvest"].to_numpy()
        - 0.35 * (seg == "hnwi")
        + 0.25 * (clients["tenure0"].to_numpy() + month_idx < 6)
        - 0.45 * (chan == "relationship_manager")
        + 0.35 * complaints
        + 0.15 * failed
        - 0.30 * clients["kyc"].to_numpy()
    )
    if drift:
        # concept drift: after a run of app outages, service failures drive churn much harder,
        # and app-only clients leave faster when the market turns
        z = z + 0.55 * complaints + 0.45 * failed + 0.45 * (chan == "mobile_app")
    p = 1 / (1 + np.exp(-z))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def generate(n_clients: int = 12000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clients = _clients(n_clients, rng)
    frames = []
    for i, period in enumerate(ALL_MONTHS):
        frames.append(_snapshot(clients, i, period, rng, drift=period in DRIFT_MONTHS))
    return pd.concat(frames, ignore_index=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based splits. Never random: we always predict the future from the past."""
    m = df["snapshot_month"]
    return {
        "train": df[m.isin(TRAIN_MONTHS.astype(str))].reset_index(drop=True),
        "valid": df[m.isin(VALID_MONTHS.astype(str))].reset_index(drop=True),
        "test": df[m.isin(TEST_MONTHS.astype(str))].reset_index(drop=True),
        "drift": df[m.isin(DRIFT_MONTHS.astype(str))].reset_index(drop=True),
    }


if __name__ == "__main__":
    data = generate()
    data.to_parquet("data/client_snapshots.parquet", index=False)
    print(data.groupby("snapshot_month")[TARGET].mean().round(3).to_string())
    print(f"rows={len(data):,} clients={data.client_id.nunique():,}")
