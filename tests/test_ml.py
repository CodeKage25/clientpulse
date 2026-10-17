import numpy as np
import pandas as pd
import pytest

from clientpulse.data import TARGET, generate, split
from clientpulse.features import MODEL_FEATURES, build_features
from clientpulse.model import evaluate, expected_value, fit_lgbm, reason_codes
from clientpulse.monitoring import EXPECTED_DRIFT, build_reference, feature_psi, monitor, psi
from clientpulse.registry import Registry


@pytest.fixture(scope="module")
def small():
    df = generate(n_clients=1500, seed=1)
    s = split(df)
    Xtr, Xva = build_features(s["train"]), build_features(s["valid"])
    model = fit_lgbm(Xtr, s["train"][TARGET], Xva, s["valid"][TARGET])
    return df, s, model


def test_splits_are_time_ordered_and_disjoint(small):
    _, s, _ = small
    assert s["train"].snapshot_month.max() < s["valid"].snapshot_month.min()
    assert s["valid"].snapshot_month.max() < s["test"].snapshot_month.min()
    assert s["test"].snapshot_month.max() < s["drift"].snapshot_month.min()


def test_features_match_contract(small):
    df, _, _ = small
    X = build_features(df.head(50))
    assert list(X.columns) == MODEL_FEATURES
    assert not X.drop(columns=["segment", "channel", "region"]).isna().any().any()


def test_unknown_category_becomes_missing_not_crash(small):
    df, _, model = small
    row = df.head(1).copy()
    row["channel"] = "carrier_pigeon"
    p = model.predict_proba(build_features(row))[:, 1]
    assert 0 <= p[0] <= 1


def test_model_beats_random(small):
    _, s, model = small
    m = evaluate(s["test"][TARGET], model.predict_proba(build_features(s["test"]))[:, 1])
    assert m["pr_auc"] > 2 * m["churn_rate"]
    assert m["lift_top10"] > 2


def test_expected_value_prefers_valuable_clients():
    ev = expected_value(np.array([0.3, 0.3]), np.array([30_000.0, 5_000_000.0]))
    assert ev[1] > 0 > ev[0]


def test_reason_codes_only_positive_drivers(small):
    df, _, model = small
    reasons = reason_codes(model, build_features(df.head(20)))
    assert len(reasons) == 20
    assert all(r["impact"] > 0 for rs in reasons for r in rs)


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    prof = build_reference(pd.DataFrame({"x": a}), a.to_numpy())["x"]
    assert feature_psi(prof, a) < 0.01
    assert feature_psi(prof, a + 1.5) > 0.25
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_monitor_flags_drift_window_and_ignores_expected_drift(small, tmp_path):
    _, s, model = small
    Xtr, Xte = build_features(s["train"]), build_features(s["test"])
    meta = {
        "version": "v1",
        "metrics": {"test": evaluate(s["test"][TARGET], model.predict_proba(Xte)[:, 1])},
        "reference_profile": build_reference(Xtr, model.predict_proba(Xtr)[:, 1]),
    }
    report = monitor(model, meta, build_features(s["drift"]), "drift", s["drift"][TARGET])
    assert report.retrain
    assert not any(f in t for t in report.triggers for f in EXPECTED_DRIFT if "Data drift" in t)


def test_registry_promotion_keeps_audit_trail(small, tmp_path):
    _, _, model = small
    reg = Registry(tmp_path)
    v1 = reg.register(model, {"metrics": {}})
    reg.promote(v1, "first")
    v2 = reg.register(model, {"metrics": {}})
    reg.promote(v2, "better")
    idx = reg.index()
    assert idx["production"] == "v2"
    assert [v["stage"] for v in idx["versions"]] == ["archived", "production"]
    assert [h["to"] for h in idx["history"]] == ["v1", "v2"]
    _, meta = reg.load()
    assert meta["stage"] == "production"


def test_recency_weights_halve_every_half_life():
    from clientpulse.model import recency_weights

    w = recency_weights(pd.Series(["2026-07", "2026-04", "2026-01"]), half_life_months=3)
    assert w == pytest.approx([1.0, 0.5, 0.25])


def test_recalibration_fixes_level_after_base_rate_jump(small):
    from clientpulse.model import RecalibratedModel

    _, s, model = small
    drift = s["drift"]
    cal, hold = drift[drift.snapshot_month == "2026-08"], drift[drift.snapshot_month == "2026-09"]
    rec = RecalibratedModel(model, build_features(cal), cal[TARGET])
    before = model.predict_proba(build_features(hold))[:, 1].mean()
    after = rec.predict_proba(build_features(hold))[:, 1].mean()
    observed = hold[TARGET].mean()
    assert abs(after - observed) < abs(before - observed)
    assert len(reason_codes(rec, build_features(hold.head(5)))) == 5  # explanations still work


def test_fairness_report_covers_every_group(small):
    from clientpulse.fairness import group_report

    _, s, model = small
    te = s["test"]
    rep = group_report(te, model.predict_proba(build_features(te))[:, 1], TARGET)
    assert set(rep.dimension) == {"segment", "region", "channel", "age band"}
    assert rep.loc[rep.dimension == "segment", "rows"].sum() == len(te)
