"""Static report charts. Thin marks, recessive axes, one fixed colour per series."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

BLUE = "#2a78d6"     # series 1: boosted model / current
ORANGE = "#eb6834"   # series 2: baseline / reference
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ACT = "#e34948"
WATCH = "#eda100"
OK = "#a9a8a2"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def pr_curves(y: np.ndarray, curves: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    colors = [BLUE, ORANGE]
    for (name, scores), c in zip(curves.items(), colors):
        p, r, _ = precision_recall_curve(y, scores)
        ax.plot(r, p, color=c, linewidth=2, label=name)
    ax.axhline(y.mean(), color=INK_2, linewidth=1, linestyle=":", label=f"Random ({y.mean():.1%})")
    _style(ax, "Precision vs recall on the April to June 2026 test window", "Recall (share of churners found)", "Precision")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    frac, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    lim = max(frac.max(), mean_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, linestyle=":", label="Perfect calibration")
    ax.plot(mean_pred, frac, color=BLUE, linewidth=2, marker="o", markersize=5, label="Model")
    _style(ax, "Calibration: predicted vs observed churn", "Mean predicted probability", "Observed churn rate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lift_by_decile(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    order = np.argsort(-scores)
    deciles = np.array_split(y[order], 10)
    rates = [d.mean() for d in deciles]
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    ax.bar(range(1, 11), rates, color=BLUE, width=0.75)
    ax.axhline(y.mean(), color=INK_2, linewidth=1, linestyle=":")
    ax.text(10.4, y.mean(), f"avg {y.mean():.1%}", color=INK_2, fontsize=8, va="bottom", ha="right")
    ax.text(1, rates[0], f"{rates[0]:.0%}", color=INK, fontsize=8, ha="center", va="bottom")
    _style(ax, "Churn rate by risk decile (1 = highest risk)", "Risk decile", "Actual churn rate")
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance(names: list[str], values: np.ndarray, path: Path, top: int = 12) -> None:
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=150)
    ax.barh([names[i] for i in order], values[order], color=BLUE, height=0.65)
    _style(ax, "What drives churn risk (mean |SHAP|)", "Average impact on log-odds", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def drift_bars(psi: dict[str, float], path: Path, watch: float, act: float,
               expected: set[str] = frozenset()) -> None:
    items = sorted(psi.items(), key=lambda kv: kv[1])
    names = [k + (" (expected)" if k in expected else "") for k, _ in items]
    vals = np.array([v for _, v in items])
    colors = [OK if k in expected else ACT if v >= act else WATCH if v >= watch else OK
              for (k, _), v in zip(items, vals)]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.barh(names, vals, color=colors, height=0.65)
    ax.axvline(watch, color=WATCH, linewidth=1, linestyle="--")
    ax.axvline(act, color=ACT, linewidth=1, linestyle="--")
    ax.text(act, len(names) - 0.4, " act", color=INK_2, fontsize=8)
    ax.text(watch, len(names) - 0.4, " watch", color=INK_2, fontsize=8)
    _style(ax, "Feature drift, July to September 2026 vs training", "Population Stability Index", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def monthly_performance(rows: pd.DataFrame, path: Path) -> None:
    """rows: columns month, series, value. Observed churn vs what each model predicted."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    styles = {0: dict(color=INK_2, linestyle=":"), 1: dict(color=ORANGE), 2: dict(color=BLUE)}
    for i, name in enumerate(rows["series"].unique()):
        r = rows[rows["series"] == name]
        ax.plot(r["month"], r["value"], linewidth=2, marker="o", markersize=5, label=name, **styles[i])
    ax.axvspan(5.5, 8.5, color=GRID, alpha=0.5, linewidth=0)
    ax.text(8.4, ax.get_ylim()[0], "market downturn ", color=INK_2, fontsize=8, va="bottom", ha="right")
    _style(ax, "Observed vs predicted churn: the old model under-predicts after the shock",
           "Snapshot month", "Monthly churn rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
