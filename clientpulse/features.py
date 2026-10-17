"""Feature engineering shared by training, batch scoring and the API.

One function, used everywhere, so training and serving can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES_CATEGORICAL, FEATURES_NUMERIC, SEGMENTS, CHANNELS, REGIONS

CATEGORY_LEVELS = {"segment": SEGMENTS, "channel": CHANNELS, "region": REGIONS}

ENGINEERED = [
    "log_portfolio_value",
    "is_dormant",
    "has_outflow",
    "service_issues_90d",
    "trades_per_login",
]
MODEL_FEATURES = [c for c in FEATURES_NUMERIC if c != "portfolio_value_ngn"] + ENGINEERED + FEATURES_CATEGORICAL

# Revenue assumptions used for value-aware targeting (documented in docs/MODEL_CARD.md)
ANNUAL_FEE_RATE = 0.008          # blended management and custody fee on AUM
COMMISSION_PER_TRADE_NGN = 2500  # average net commission per trade


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in FEATURES_NUMERIC:
        if col != "portfolio_value_ngn":
            out[col] = pd.to_numeric(df[col], errors="coerce")
    out["log_portfolio_value"] = np.log1p(pd.to_numeric(df["portfolio_value_ngn"], errors="coerce").clip(lower=0))
    out["is_dormant"] = (df["days_since_login"] >= 30).astype(int)
    out["has_outflow"] = (df["net_flow_90d_share"] < -0.05).astype(int)
    out["service_issues_90d"] = df["complaints_90d"] + df["failed_txn_90d"]
    out["trades_per_login"] = df["trades_90d"] / (df["logins_30d"] * 3 + 1)
    for col in FEATURES_CATEGORICAL:
        out[col] = pd.Categorical(df[col], categories=CATEGORY_LEVELS[col])
    return out[MODEL_FEATURES]


def annual_revenue_ngn(df: pd.DataFrame) -> pd.Series:
    """What a client is worth to the business per year, for value-aware targeting."""
    return df["portfolio_value_ngn"] * ANNUAL_FEE_RATE + df["trades_90d"] * 4 * COMMISSION_PER_TRADE_NGN
