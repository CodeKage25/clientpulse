"""Automation value report: FTE savings, processing time and error rate.

Every number here is either measured from the bot run log or an explicit
assumption listed in ASSUMPTIONS. In a real deployment the assumptions come
from a time-and-motion study of the team before go-live; that baseline is
what makes the savings claim credible to finance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ASSUMPTIONS = {
    "manual_minutes_per_request": {
        "statement_request": 8,
        "address_update": 12,
        "dividend_enquiry": 10,
        "redemption_request": 15,
        "complaint": 10,
    },
    "exception_minutes_saved_share": 0.30,  # agent starts from the bot's diagnosis, not from scratch
    "monthly_request_volume": 2500,
    "productive_hours_per_fte_month": 140,
    "manual_turnaround_hours": 6.0,         # average wait in the shared inbox today
    "bot_turnaround_minutes": 5.0,          # queue polled every 5 minutes
    "manual_error_rate": 0.025,             # keying or routing errors in manual handling
}


def value_report(log: pd.DataFrame) -> dict:
    a = ASSUMPTIONS
    n = len(log)
    mins = log["true_intent"].map(a["manual_minutes_per_request"])
    auto = log["status"] == "Successful"
    exceptions = log["status"].isin(["BusinessException", "SystemException"])

    # measured
    stp_rate = auto.mean()
    misrouted = (auto & (log["predicted_intent"] != log["true_intent"])).sum()
    bot_error_rate = misrouted / max(auto.sum(), 1)

    # minutes saved on this run, then scaled to monthly volume
    saved_run = mins[auto].sum() + a["exception_minutes_saved_share"] * mins[exceptions].sum()
    manual_run = mins.sum()
    scale = a["monthly_request_volume"] / n
    hours_saved_month = saved_run * scale / 60
    fte = hours_saved_month / a["productive_hours_per_fte_month"]

    return {
        "requests_processed": n,
        "straight_through_rate": round(float(stp_rate), 3),
        "outcomes": log["status"].value_counts().to_dict(),
        "exception_reasons_top": log.loc[log["status"] != "Successful", "reason"]
        .str.replace(r"MC\d{6}", "<acct>", regex=True)
        .str.replace(r"N[\d,]+", "N<amt>", regex=True)
        .str.replace(r"\(\d\.\d+\); could be .*", "(below 0.70)", regex=True)
        .value_counts()
        .head(6)
        .to_dict(),
        "manual_hours_per_month_before": round(manual_run * scale / 60, 1),
        "hours_saved_per_month": round(hours_saved_month, 1),
        "fte_saved": round(fte, 2),
        "share_of_effort_removed": round(saved_run / manual_run, 3),
        "turnaround_before": f"{a['manual_turnaround_hours']} hours",
        "turnaround_after_auto": f"{a['bot_turnaround_minutes']:.0f} minutes",
        "manual_error_rate_assumed": a["manual_error_rate"],
        "bot_error_rate_measured": round(float(bot_error_rate), 4),
        "misrouted_auto_actions": int(misrouted),
        "assumptions": a,
    }


def to_markdown(v: dict) -> str:
    lines = [
        "# Automation value report: client service inbox bot",
        "",
        f"Requests processed: **{v['requests_processed']:,}**  ",
        f"Straight-through (no human touch): **{v['straight_through_rate']:.0%}**",
        "",
        "## Value",
        "",
        "| Measure | Before | After |",
        "|---|---:|---:|",
        f"| Team hours per month on this inbox | {v['manual_hours_per_month_before']:,} | "
        f"{v['manual_hours_per_month_before'] - v['hours_saved_per_month']:,.1f} |",
        f"| Turnaround for automated requests | {v['turnaround_before']} | {v['turnaround_after_auto']} |",
        f"| Error rate on processed requests | {v['manual_error_rate_assumed']:.1%} (assumed) | "
        f"{v['bot_error_rate_measured']:.1%} (measured) |",
        "",
        f"**FTE capacity released: {v['fte_saved']}** "
        f"({v['hours_saved_per_month']:,} hours a month, {v['share_of_effort_removed']:.0%} of the effort). "
        "That capacity moves to complaints and advisory work rather than headcount cuts.",
        "",
        "## Outcomes",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {c:,} |" for k, c in v["outcomes"].items()]
    lines += ["", "## Top reasons a person was needed", "", "| Reason | Count |", "|---|---:|"]
    lines += [f"| {k} | {c} |" for k, c in v["exception_reasons_top"].items()]
    lines += ["", "## Assumptions", "", "```json", json.dumps(v["assumptions"], indent=2), "```", ""]
    return "\n".join(lines)


def outcome_chart(log: pd.DataFrame, path: Path) -> None:
    order = ["Successful", "BusinessException", "HumanReview", "SystemException"]
    labels = {
        "Successful": "Fully automated",
        "BusinessException": "Business rule hold",
        "HumanReview": "Low confidence, to agent",
        "SystemException": "Failed after retries",
    }
    colors = {"Successful": "#2a78d6", "BusinessException": "#eda100", "HumanReview": "#a9a8a2", "SystemException": "#e34948"}
    pivot = pd.crosstab(log["true_intent"], log["status"]).reindex(columns=order, fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    left = pd.Series(0, index=pivot.index)
    for s in order:
        if pivot[s].sum() == 0:
            continue
        ax.barh(pivot.index, pivot[s], left=left, color=colors[s], label=labels[s], height=0.6,
                edgecolor="white", linewidth=1.5)
        left += pivot[s]
    ax.set_title("Where each request ended up, by request type", loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("Requests", color="#52514e", fontsize=9)
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
