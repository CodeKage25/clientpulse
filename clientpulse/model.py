"""Train, evaluate and explain the churn model.

Choices worth defending in an interview:
- Time-based split, never random. Random splits leak future behaviour into
  training and inflate scores.
- PR-AUC as the headline metric. With ~6% churn, ROC-AUC looks great even for
  weak models; PR-AUC tells you how clean the top of the list is.
- A plain logistic regression baseline first. The boosted model has to beat
  it by enough to justify the extra complexity.
- Targeting by expected value, not a fixed probability cut-off. A 30% churn
  risk on a N120m client matters far more than 60% on a N200k client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES_CATEGORICAL, TARGET
from .features import MODEL_FEATURES, annual_revenue_ngn, build_features

LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=7,
    verbose=-1,
)

# Retention economics (assumptions, documented in the model card)
CONTACT_COST_NGN = 7_500   # RM time plus a modest fee waiver
SAVE_RATE = 0.25           # share of contacted would-be churners we keep
VALUE_HORIZON_YEARS = 2    # revenue horizon for a retained client


def baseline_model() -> Pipeline:
    numeric = [c for c in MODEL_FEATURES if c not in FEATURES_CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight=None))])


def fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
             params: dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**(params or LGB_PARAMS))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def lift_at(y: np.ndarray, scores: np.ndarray, top: float = 0.10) -> float:
    k = max(1, int(len(y) * top))
    idx = np.argsort(-scores)[:k]
    return float(y[idx].mean() / max(y.mean(), 1e-9))


def recall_at(y: np.ndarray, scores: np.ndarray, top: float = 0.10) -> float:
    k = max(1, int(len(y) * top))
    idx = np.argsort(-scores)[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def evaluate(y: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    return {
        "rows": int(len(y)),
        "churn_rate": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "brier": round(float(brier_score_loss(y, scores)), 4),
        "lift_top10": round(lift_at(y, scores, 0.10), 2),
        "recall_top10": round(recall_at(y, scores, 0.10), 4),
        "mean_predicted": round(float(scores.mean()), 4),
    }


@dataclass
class TargetingResult:
    policy: str
    contacted: int
    churners_reached: int
    expected_net_value_ngn: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def expected_value(scores: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    """Expected net value of contacting each client."""
    return scores * SAVE_RATE * revenue * VALUE_HORIZON_YEARS - CONTACT_COST_NGN


def targeting_comparison(df: pd.DataFrame, scores: np.ndarray) -> list[TargetingResult]:
    """Compare retention policies on realised outcomes (per month, averaged)."""
    y = df[TARGET].to_numpy()
    revenue = annual_revenue_ngn(df).to_numpy()
    n_months = df["snapshot_month"].nunique()

    def realised(mask: np.ndarray, name: str) -> TargetingResult:
        value = (y[mask] * SAVE_RATE * revenue[mask] * VALUE_HORIZON_YEARS).sum() - CONTACT_COST_NGN * mask.sum()
        return TargetingResult(name, int(mask.sum() / n_months), int(y[mask].sum() / n_months), float(value / n_months))

    top10 = np.zeros(len(y), dtype=bool)
    top10[np.argsort(-scores)[: int(len(y) * 0.10)]] = True
    ev_mask = expected_value(scores, revenue) > 0
    return [
        realised(np.ones(len(y), dtype=bool), "Contact every client"),
        realised(top10, "Top 10% by churn probability"),
        realised(ev_mask, "Expected-value targeting (contact when EV > 0)"),
    ]


def reason_codes(model: lgb.LGBMClassifier, X: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
    """Per-client top drivers from SHAP-style contributions (LightGBM pred_contrib).

    Positive values push churn risk up. These are what an RM sees next to a name.
    """
    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]
    out = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])[:top_k]
        out.append(
            [
                {"feature": X.columns[j], "value": _jsonable(X.iloc[i, j]), "impact": round(float(contrib[i, j]), 4)}
                for j in order
                if contrib[i, j] > 0
            ]
        )
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v if isinstance(v, (int, float)) else str(v)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(df), df[TARGET]
