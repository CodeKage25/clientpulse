"""Drift and performance monitoring.

Three questions, in the order you can answer them in production:
1. Data drift: do today's inputs look like the training inputs? (PSI per feature)
   Available immediately, no labels needed.
2. Prediction drift: has the score distribution moved? (PSI on scores)
   Also label-free. Often the first visible symptom.
3. Performance decay: now that 90-day outcomes are in, is the model still right?
   (PR-AUC, calibration). This is the only check that catches concept drift,
   where inputs look the same but the relationship to churn has changed.

PSI rule of thumb: < 0.10 stable, 0.10 to 0.25 watch, > 0.25 act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PSI_WATCH = 0.10
PSI_ACT = 0.25
ROC_AUC_DROP_ACT = 0.03  # absolute drop in ranking quality
EPS = 1e-4

# Features that drift by construction (they grow with calendar time). We still
# report them, but they never trigger a retrain on their own; otherwise every
# model would be "drifting" a few months after release.
EXPECTED_DRIFT = {"tenure_months", "age"}

# Note on PR-AUC: it moves with the churn base rate, so a market shock that
# doubles churn can make PR-AUC go UP while the model gets worse. Decay is
# therefore judged on ROC-AUC (prevalence-free) and on calibration.


# reference profiles, captured at training time and stored with the model

def numeric_profile(values: pd.Series, n_bins: int = 10) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # near-constant feature, fall back to its distinct values
        edges = np.unique(np.concatenate([[v.min() - 1e-9], np.unique(v), [v.max() + 1e-9]]))
    inner = edges[1:-1]
    counts = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1)
    return {"type": "numeric", "inner_edges": inner.tolist(), "proportions": (counts / counts.sum()).tolist()}


def categorical_profile(values: pd.Series) -> dict[str, Any]:
    p = values.astype(str).value_counts(normalize=True)
    return {"type": "categorical", "proportions": p.to_dict()}


def build_reference(features: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    ref = {}
    for col in features.columns:
        if isinstance(features[col].dtype, pd.CategoricalDtype) or features[col].dtype == object:
            ref[col] = categorical_profile(features[col])
        else:
            ref[col] = numeric_profile(features[col])
    ref["__score__"] = numeric_profile(pd.Series(scores))
    return ref


# drift statistics

def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(profile: dict[str, Any], values: pd.Series) -> float:
    if profile["type"] == "numeric":
        inner = np.asarray(profile["inner_edges"])
        v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        counts = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1)
        return psi(profile["proportions"], counts / max(counts.sum(), 1))
    expected = profile["proportions"]
    actual = values.astype(str).value_counts(normalize=True)
    cats = sorted(set(expected) | set(actual.index))
    return psi([expected.get(c, 0.0) for c in cats], [float(actual.get(c, 0.0)) for c in cats])


def status(value: float) -> str:
    if value >= PSI_ACT:
        return "ACT"
    if value >= PSI_WATCH:
        return "WATCH"
    return "OK"


@dataclass
class MonitoringReport:
    window: str
    model_version: str
    rows: int
    feature_psi: dict[str, float]
    score_psi: float
    baseline_pr_auc: float
    baseline_roc_auc: float
    pr_auc: float | None = None
    roc_auc: float | None = None
    brier: float | None = None
    observed_churn_rate: float | None = None
    mean_predicted: float | None = None
    triggers: list[str] = field(default_factory=list)

    @property
    def retrain(self) -> bool:
        return bool(self.triggers)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["retrain"] = self.retrain
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# Monitoring report: {self.window}",
            "",
            f"Model in production: **{self.model_version}**, rows scored: {self.rows:,}",
            "",
            "## Data drift (PSI vs training reference)",
            "",
            "| Feature | PSI | Status |",
            "|---|---:|---|",
        ]
        for name, value in sorted(self.feature_psi.items(), key=lambda kv: -kv[1]):
            label = status(value) + (" (expected, grows with time)" if name in EXPECTED_DRIFT else "")
            lines.append(f"| {name} | {value:.3f} | {label} |")
        lines += [
            "",
            f"Score distribution PSI: **{self.score_psi:.3f}** ({status(self.score_psi)})",
            "",
            "## Performance (labels matured)",
            "",
        ]
        if self.pr_auc is not None:
            lines += [
                f"- ROC-AUC: {self.roc_auc:.3f} vs {self.baseline_roc_auc:.3f} at release",
                f"- PR-AUC: {self.pr_auc:.3f} vs {self.baseline_pr_auc:.3f} at release "
                "(not comparable on its own: it rises with the churn base rate)",
                f"- Observed churn rate: {self.observed_churn_rate:.1%}, mean predicted: {self.mean_predicted:.1%}",
                f"- Brier score: {self.brier:.4f}",
            ]
        else:
            lines.append("- Labels not matured yet; relying on drift signals only.")
        lines += ["", "## Decision", ""]
        if self.triggers:
            lines.append("**Retrain triggered.** Reasons:")
            lines += [f"- {t}" for t in self.triggers]
        else:
            lines.append("No action. Model is stable.")
        return "\n".join(lines) + "\n"


def monitor(model: Any, metadata: dict[str, Any], features: pd.DataFrame, window: str,
            labels: pd.Series | None = None) -> MonitoringReport:
    ref = metadata["reference_profile"]
    scores = model.predict_proba(features)[:, 1]
    fpsi = {col: round(feature_psi(ref[col], features[col]), 4) for col in features.columns if col in ref}
    spsi = round(feature_psi(ref["__score__"], pd.Series(scores)), 4)
    report = MonitoringReport(
        window=window,
        model_version=metadata["version"],
        rows=len(features),
        feature_psi=fpsi,
        score_psi=spsi,
        baseline_pr_auc=metadata["metrics"]["test"]["pr_auc"],
        baseline_roc_auc=metadata["metrics"]["test"]["roc_auc"],
    )
    drifted = [f for f, v in fpsi.items() if v >= PSI_ACT and f not in EXPECTED_DRIFT]
    if drifted:
        report.triggers.append(f"Data drift above {PSI_ACT} on: {', '.join(drifted)}")
    if spsi >= PSI_ACT:
        report.triggers.append(f"Score distribution shifted (PSI {spsi:.3f})")

    if labels is not None:
        y = labels.to_numpy()
        report.pr_auc = round(float(average_precision_score(y, scores)), 4)
        report.roc_auc = round(float(roc_auc_score(y, scores)), 4)
        report.brier = round(float(brier_score_loss(y, scores)), 4)
        report.observed_churn_rate = round(float(y.mean()), 4)
        report.mean_predicted = round(float(scores.mean()), 4)
        drop = report.baseline_roc_auc - report.roc_auc
        if drop >= ROC_AUC_DROP_ACT:
            report.triggers.append(
                f"ROC-AUC fell {drop:.3f} from release ({report.baseline_roc_auc:.3f} to {report.roc_auc:.3f}); "
                "concept drift suspected"
            )
        if report.observed_churn_rate > 1.5 * report.mean_predicted:
            report.triggers.append(
                f"Under-predicting churn: observed {report.observed_churn_rate:.1%} vs predicted {report.mean_predicted:.1%}"
            )
    return report
