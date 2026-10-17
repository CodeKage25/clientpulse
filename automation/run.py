"""Run the client service bot end to end:

    python -m automation.run

Needs the churn model registered first (python -m clientpulse.pipeline),
because complaints are prioritised by the client's churn risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clientpulse.features import build_features
from clientpulse.pipeline import load_data
from clientpulse.registry import Registry

from .bot import ClientServiceBot, write_outputs
from .inbox import client_master, generate_inbox, labelled_history, save
from .intent import IntentModel
from .systems import BackOffice
from .value import outcome_chart, to_markdown, value_report

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "automation"


def churn_scorer(master: pd.DataFrame):
    model, meta = Registry(ROOT / "models").load()
    X = build_features(master)
    probs = dict(zip(master["client_id"], model.predict_proba(X)[:, 1]))
    return lambda client_id: probs[client_id], meta["version"]


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = load_data()
    master = client_master(snapshots)

    intent = IntentModel().fit(labelled_history(master))
    (OUT / "intent_model_report.txt").write_text(intent.report)

    inbox = generate_inbox(master)
    save(inbox, OUT / "inbox.json")

    scorer, churn_version = churn_scorer(master)
    backoffice = BackOffice(master.copy(), OUT)
    bot = ClientServiceBot(backoffice, intent, scorer)
    log = bot.run(inbox)
    write_outputs(log, backoffice, OUT)

    v = value_report(log)
    v["churn_model_version"] = churn_version
    (OUT / "value_report.md").write_text(to_markdown(v))
    (OUT / "value_report.json").write_text(json.dumps(v, indent=2, default=str))
    outcome_chart(log, OUT / "outcomes.png")

    print(f"[bot] {v['requests_processed']} requests | STP {v['straight_through_rate']:.0%} | "
          f"FTE released {v['fte_saved']} | bot error rate {v['bot_error_rate_measured']:.2%}")
    print(f"[bot] outcomes: {v['outcomes']}")
    return v


if __name__ == "__main__":
    main()
