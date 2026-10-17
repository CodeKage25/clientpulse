"""End-to-end ML lifecycle, runnable in one command:

    python -m clientpulse.pipeline

1. Generate data (or load the cached parquet)
2. Train baseline and LightGBM on 2025, early-stop on Q1 2026, test on Q2 2026
3. Register v1 with metrics, lineage and a drift reference profile; promote it
4. Monitor July to September 2026 (market downturn plus concept drift)
5. Retrain a challenger on a rolling window, compare on a held-out recent month
6. Promote only if the challenger wins by a clear margin
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from . import plots
from .data import TARGET, generate, split
from .features import MODEL_FEATURES, annual_revenue_ngn, build_features
from .model import (
    LGB_PARAMS,
    RecalibratedModel,
    recency_weights,
    baseline_model,
    evaluate,
    fit_lgbm,
    targeting_comparison,
)
from .monitoring import EXPECTED_DRIFT, PSI_ACT, PSI_WATCH, build_reference, monitor
from .registry import Registry, data_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "client_snapshots.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

PROMOTION_MARGIN = 0.01  # challenger must beat champion PR-AUC by at least this


def load_data() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_parquet(DATA)
    df = generate()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA, index=False)
    return df


def train_initial(splits: dict[str, pd.DataFrame], registry: Registry) -> str:
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    Xtr, Xva, Xte = build_features(tr), build_features(va), build_features(te)
    ytr, yva, yte = tr[TARGET], va[TARGET], te[TARGET]

    base = baseline_model().fit(Xtr, ytr)
    model = fit_lgbm(Xtr, ytr, Xva, yva)

    s_base, s_lgb = base.predict_proba(Xte)[:, 1], model.predict_proba(Xte)[:, 1]
    metrics = {
        "baseline_logreg_test": evaluate(yte, s_base),
        "valid": evaluate(yva, model.predict_proba(Xva)[:, 1]),
        "test": evaluate(yte, s_lgb),
    }
    targeting = [t.to_dict() for t in targeting_comparison(te, s_lgb)]

    # global explanations
    sample = Xte.sample(4000, random_state=7)
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(zip(MODEL_FEATURES, mean_abs.round(4).tolist()), key=lambda kv: -kv[1]))

    REPORTS.mkdir(exist_ok=True)
    plots.pr_curves(yte.to_numpy(), {"LightGBM": s_lgb, "Logistic baseline": s_base}, REPORTS / "pr_curve.png")
    plots.calibration(yte.to_numpy(), s_lgb, REPORTS / "calibration.png")
    plots.lift_by_decile(yte.to_numpy(), s_lgb, REPORTS / "lift_by_decile.png")
    plots.importance(MODEL_FEATURES, mean_abs, REPORTS / "shap_importance.png")

    metadata = {
        "model_type": "LightGBM binary classifier",
        "target": f"{TARGET}: client dormant or moves most assets out within 90 days",
        "train_window": f"{tr.snapshot_month.min()} to {tr.snapshot_month.max()}",
        "valid_window": f"{va.snapshot_month.min()} to {va.snapshot_month.max()}",
        "test_window": f"{te.snapshot_month.min()} to {te.snapshot_month.max()}",
        "train_rows": len(tr),
        "data_fingerprint": data_fingerprint(tr),
        "features": MODEL_FEATURES,
        "params": {**LGB_PARAMS, "best_iteration": int(model.best_iteration_ or LGB_PARAMS["n_estimators"])},
        "metrics": metrics,
        "targeting_test": targeting,
        "shap_importance": importance,
        "reference_profile": build_reference(Xtr, model.predict_proba(Xtr)[:, 1]),
    }
    version = registry.register(model, metadata)
    registry.promote(version, "Initial release: beats logistic baseline on test PR-AUC")
    return version


def run_fairness(splits: dict[str, pd.DataFrame], registry: Registry, version: str) -> pd.DataFrame:
    from . import fairness

    model, meta = registry.load(version)
    te = splits["test"]
    report = fairness.group_report(te, model.predict_proba(build_features(te))[:, 1], TARGET)
    (REPORTS / "fairness.md").write_text(fairness.to_markdown(report, meta["test_window"], version))
    return report


def run_monitoring(splits: dict[str, pd.DataFrame], registry: Registry) -> dict:
    model, meta = registry.load()
    drift = splits["drift"]
    X = build_features(drift)
    report = monitor(model, meta, X, window="2026-07 to 2026-09", labels=drift[TARGET])
    (REPORTS / "monitoring_report.md").write_text(report.to_markdown())
    (REPORTS / "monitoring_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    plots.drift_bars(report.feature_psi, REPORTS / "drift_psi.png", PSI_WATCH, PSI_ACT, EXPECTED_DRIFT)
    return report.to_dict()


def _write_ablation(ablation: dict, champ: dict, champ_version: str, chosen: str, observed: float) -> None:
    lines = [
        "# Retraining ablation: fixing post-shock calibration",
        "",
        f"Held-out month 2026-09, observed churn {observed:.1%}. Lower Brier is better.",
        "",
        "| Model | PR-AUC | ROC-AUC | Brier | Mean predicted |",
        "|---|---:|---:|---:|---:|",
        f"| Champion {champ_version} (no retrain) | {champ['pr_auc']:.3f} | {champ['roc_auc']:.3f} | "
        f"{champ['brier']:.4f} | {champ['mean_predicted']:.1%} |",
    ]
    for name, m in ablation.items():
        mark = " **(chosen)**" if name == chosen else ""
        lines.append(f"| {name}{mark} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.4f} | "
                     f"{m['mean_predicted']:.1%} |")
    lines += ["", "Rule: the lowest Brier among variants whose PR-AUC is within 0.005 of the plain retrain.", ""]
    (REPORTS / "retrain_ablation.md").write_text("\n".join(lines))


def retrain(df: pd.DataFrame, registry: Registry) -> dict:
    """Champion vs challenger.

    Simulated as of end of December 2026, when 90-day labels for September
    snapshots have matured.
    - Challenger trains on a rolling 12 months (2025-08 to 2026-07)
    - Early stopping on 2026-08
    - Both models scored on 2026-09, which neither has seen
    """
    months = df["snapshot_month"]
    train_m = pd.period_range("2025-08", "2026-07", freq="M").astype(str)
    tr, va, ho = df[months.isin(train_m)], df[months == "2026-08"], df[months == "2026-09"]
    Xtr, Xva, Xho = build_features(tr), build_features(va), build_features(ho)

    champion, champ_meta = registry.load()
    s_champ = champion.predict_proba(Xho)[:, 1]
    m_champ = evaluate(ho[TARGET], s_champ)

    # Ablation: which fix closes the post-shock calibration gap?
    # Recalibration is fitted on 2026-08, the latest matured month; 2026-09 stays untouched.
    plain = fit_lgbm(Xtr, tr[TARGET], Xva, va[TARGET])
    weighted = fit_lgbm(Xtr, tr[TARGET], Xva, va[TARGET], sample_weight=recency_weights(tr["snapshot_month"]))
    candidates = {
        "rolling window": plain,
        "rolling window + recency weights": weighted,
        "rolling window + recalibration": RecalibratedModel(plain, Xva, va[TARGET]),
        "recency weights + recalibration": RecalibratedModel(weighted, Xva, va[TARGET]),
    }
    ablation = {}
    for name, m in candidates.items():
        ablation[name] = evaluate(ho[TARGET], m.predict_proba(Xho)[:, 1])
    # pick the best-calibrated variant that still ranks at least as well as the plain retrain
    floor = ablation["rolling window"]["pr_auc"] - 0.005
    best = min((n for n in ablation if ablation[n]["pr_auc"] >= floor), key=lambda n: ablation[n]["brier"])
    challenger = candidates[best]
    s_chal = challenger.predict_proba(Xho)[:, 1]
    m_chal = ablation[best]
    _write_ablation(ablation, m_champ, champ_meta["version"], best, ho[TARGET].mean())

    gain = m_chal["pr_auc"] - m_champ["pr_auc"]
    calibrated = m_chal["brier"] <= m_champ["brier"]
    promote = gain >= PROMOTION_MARGIN and calibrated

    metadata = {
        "model_type": "LightGBM binary classifier",
        "target": champ_meta["target"],
        "train_window": "2025-08 to 2026-07",
        "valid_window": "2026-08",
        "test_window": "2026-09",
        "train_rows": len(tr),
        "data_fingerprint": data_fingerprint(tr),
        "features": MODEL_FEATURES,
        "params": {**LGB_PARAMS, "best_iteration": int(challenger.best_iteration_ or LGB_PARAMS["n_estimators"])},
        "parent_version": champ_meta["version"],
        "variant": best,
        "ablation_on_holdout": ablation,
        "metrics": {"valid": evaluate(va[TARGET], challenger.predict_proba(Xva)[:, 1]), "test": m_chal},
        "champion_on_same_holdout": m_champ,
        "targeting_test": [t.to_dict() for t in targeting_comparison(ho, s_chal)],
        "reference_profile": build_reference(Xtr, challenger.predict_proba(Xtr)[:, 1]),
    }
    version = registry.register(challenger, metadata)
    reason = (
        f"Challenger PR-AUC {m_chal['pr_auc']:.3f} vs champion {m_champ['pr_auc']:.3f} on 2026-09 "
        f"(+{gain:.3f}); Brier {m_chal['brier']:.4f} vs {m_champ['brier']:.4f}"
    )
    if promote:
        registry.promote(version, reason)
    else:
        registry.reject(version, reason)

    # observed vs predicted churn per month: calibration is what the shock broke
    rows = []
    for m in sorted(months.unique()):
        if m < "2026-01":
            continue
        part = df[months == m]
        Xp = build_features(part)
        rows.append({"month": m, "series": "Observed churn", "value": part[TARGET].mean()})
        rows.append({"month": m, "series": f"Champion {champ_meta['version']} predicted",
                     "value": champion.predict_proba(Xp)[:, 1].mean()})
    rows.append({"month": "2026-09", "series": f"Challenger {version} predicted (held-out month)",
                 "value": float(s_chal.mean())})
    plots.monthly_performance(pd.DataFrame(rows), REPORTS / "observed_vs_predicted.png")

    return {"challenger": version, "promoted": promote, "reason": reason,
            "champion_holdout": m_champ, "challenger_holdout": m_chal}


def batch_score(df: pd.DataFrame, registry: Registry, month: str = "2026-09") -> pd.DataFrame:
    """Monthly retention list for relationship managers, ranked by expected value."""
    from .model import expected_value, reason_codes

    model, meta = registry.load()
    snap = df[df["snapshot_month"] == month].reset_index(drop=True)
    X = build_features(snap)
    p = model.predict_proba(X)[:, 1]
    ev = expected_value(p, annual_revenue_ngn(snap).to_numpy())
    out = snap[["client_id", "segment", "channel", "portfolio_value_ngn"]].copy()
    out["churn_probability"] = p.round(4)
    out["expected_value_ngn"] = ev.round(0)
    out = out[out["expected_value_ngn"] > 0].sort_values("expected_value_ngn", ascending=False)
    top = out.head(500).copy()
    reasons = reason_codes(model, X.loc[top.index])
    top["top_reasons"] = ["; ".join(r["feature"] for r in rs) for rs in reasons]
    top["model_version"] = meta["version"]
    top.to_csv(REPORTS / f"retention_list_{month}.csv", index=False)
    return top


def main(fresh: bool = True) -> dict:
    if fresh and MODELS.exists():
        shutil.rmtree(MODELS)
    registry = Registry(MODELS, mirror_to_mlflow=True)
    df = load_data()
    splits = split(df)

    v1 = train_initial(splits, registry)
    _, meta = registry.load(v1)
    print(f"[train] registered {v1} | test {meta['metrics']['test']}")
    print(f"[train] baseline     | test {meta['metrics']['baseline_logreg_test']}")

    fair = run_fairness(splits, registry, v1)
    flagged = fair[fair["flag"] != ""]
    print(f"[fairness] {len(flagged)} group(s) flagged: " +
          ", ".join(f"{r.dimension}={r.group} ({r.flag})" for r in flagged.itertuples()))

    mon = run_monitoring(splits, registry)
    print(f"[monitor] retrain={mon['retrain']} | triggers:")
    for t in mon["triggers"]:
        print(f"   - {t}")

    result = {"initial_version": v1, "monitoring": mon}
    if mon["retrain"]:
        rt = retrain(df, registry)
        print(f"[retrain] {rt['challenger']} promoted={rt['promoted']} | {rt['reason']}")
        result["retrain"] = rt

    top = batch_score(df, registry)
    print(f"[score] retention list: {len(top)} clients, model {registry.production_version()}")
    (REPORTS / "pipeline_summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
