"""Group-level error analysis before release.

The question is not whether churn rates differ by group (they will), but
whether the model serves each group equally well: does it find churners in
every group at a similar rate, and are its probabilities honest everywhere?

Checks at the operating point (top 10% of clients flagged for contact):
- recall: share of a group's churners the list actually reaches
- false positive rate: share of a group's stayers called unnecessarily
- calibration gap: mean predicted minus observed churn

A group is flagged when its recall falls below 80% of the overall recall (the
same ratio as the four-fifths rule used in fair lending reviews), or when its
calibration is off by more than 3 percentage points. Flags are prompts for a
human review, not automatic verdicts. Small groups are noisy, so the report
also gives group size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECALL_RATIO_FLAG = 0.80
CALIBRATION_GAP_FLAG = 0.03
MIN_GROUP_ROWS = 300


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 29, 44, 59, 200], labels=["under 30", "30 to 44", "45 to 59", "60 plus"])


def group_report(df: pd.DataFrame, scores: np.ndarray, target: str, top: float = 0.10) -> pd.DataFrame:
    y = df[target].to_numpy()
    cutoff = np.quantile(scores, 1 - top)
    flagged = scores >= cutoff
    overall_recall = flagged[y == 1].mean()

    groups = {
        "segment": df["segment"],
        "region": df["region"],
        "channel": df["channel"],
        "age band": age_band(df["age"]),
    }
    rows = []
    for dim, col in groups.items():
        for level in col.dropna().unique():
            m = (col == level).to_numpy()
            yg, fg, sg = y[m], flagged[m], scores[m]
            recall = fg[yg == 1].mean() if (yg == 1).any() else np.nan
            fpr = fg[yg == 0].mean() if (yg == 0).any() else np.nan
            gap = sg.mean() - yg.mean()
            issues = []
            if m.sum() >= MIN_GROUP_ROWS and recall < RECALL_RATIO_FLAG * overall_recall:
                issues.append("low recall")
            if m.sum() >= MIN_GROUP_ROWS and abs(gap) > CALIBRATION_GAP_FLAG:
                issues.append("miscalibrated")
            rows.append(
                {
                    "dimension": dim,
                    "group": str(level),
                    "rows": int(m.sum()),
                    "churn_rate": yg.mean(),
                    "recall_at_top10": recall,
                    "recall_ratio": recall / overall_recall,
                    "false_positive_rate": fpr,
                    "calibration_gap": gap,
                    "flag": ", ".join(issues),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["overall_recall"] = overall_recall
    return out.sort_values(["dimension", "group"]).reset_index(drop=True)


def to_markdown(report: pd.DataFrame, window: str, version: str) -> str:
    lines = [
        f"# Fairness and group error analysis: {version}, {window}",
        "",
        f"Operating point: top 10% of clients by churn risk. Overall recall {report.attrs['overall_recall']:.1%}.",
        f"Flag when recall is below {RECALL_RATIO_FLAG:.0%} of overall, or calibration is off by more than "
        f"{CALIBRATION_GAP_FLAG * 100:.0f} points (groups of {MIN_GROUP_ROWS}+ rows).",
        "",
    ]
    for dim, part in report.groupby("dimension", sort=False):
        lines += [
            f"## {dim.capitalize()}",
            "",
            "| Group | Rows | Churn rate | Recall | vs overall | False positive rate | Calibration gap | Flag |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in part.iterrows():
            lines.append(
                f"| {r.group} | {r.rows:,} | {r.churn_rate:.1%} | {r.recall_at_top10:.1%} | {r.recall_ratio:.2f} | "
                f"{r.false_positive_rate:.1%} | {r.calibration_gap * 100:+.1f} pts | {r.flag or 'ok'} |"
            )
        lines.append("")
    flagged = report[report["flag"] != ""]
    lines += ["## Summary", ""]
    if flagged.empty:
        lines.append("No group crosses either threshold.")
    else:
        for _, r in flagged.iterrows():
            lines.append(f"- **{r.dimension}: {r.group}** ({r.flag}): recall {r.recall_at_top10:.1%}, "
                         f"calibration gap {r.calibration_gap * 100:+.1f} pts")
    lines += [
        "",
        "Low recall in a group with low churn is often expected: a single top-10% cut-off is dominated by the "
        "groups where churn is common. The fix, if the business wants equal reach, is a per-group operating "
        "point or a quota, which is a business decision to make explicitly, not something to hide in the model.",
        "",
    ]
    return "\n".join(lines)
