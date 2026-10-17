# ClientPulse

Churn prediction, MLOps and intelligent automation for a brokerage and wealth management business, built end to end on synthetic data.

It covers the full cycle an AI engineer in a financial services data team owns: exploratory analysis, a validated model, value-based targeting, a versioned registry, drift monitoring, champion vs challenger retraining, a scoring API, and an automation bot that combines RPA-style process handling with ML and reports the value it delivers.

All client data is generated. Nothing here touches real people or real accounts.

## What it shows

| Area | What is built | Where |
|---|---|---|
| Churn model | LightGBM vs logistic baseline, time-based splits, PR-AUC, lift, calibration, SHAP | `clientpulse/model.py` |
| Business targeting | Contact clients by expected value in Naira, not by a fixed probability cut-off | `clientpulse/model.py` |
| Model registry | Versions, stages, lineage (data fingerprint), promotion history | `clientpulse/registry.py` |
| Monitoring | PSI data drift, score drift, calibration and ranking decay, retrain triggers | `clientpulse/monitoring.py` |
| Retraining | Rolling window challenger, held-out comparison, promotion gate | `clientpulse/pipeline.py` |
| Serving | FastAPI scoring with input validation and per-client reason codes | `clientpulse/api.py` |
| Intelligent automation | REFramework-style bot for a client service inbox, ML intent routing, controls, audit trail | `automation/` |
| Automation value | FTE released, turnaround, measured error rate, exception analysis | `automation/value.py` |

## Results from the last run

**Churn model (test window April to June 2026, 36,000 client-months, 6.9% churn)**

| Model | PR-AUC | ROC-AUC | Lift in top 10% | Recall in top 10% |
|---|---:|---:|---:|---:|
| Logistic baseline | 0.307 | 0.764 | 4.0x | 40% |
| LightGBM | 0.330 | 0.781 | 4.2x | 42% |

**Retention targeting, per month on the test window**

| Policy | Clients contacted | Expected net value |
|---|---:|---:|
| Contact everyone | 12,000 | minus N55.4m |
| Top 10% by churn probability | 1,200 | N3.5m |
| Expected-value targeting | 612 | N13.5m |

Half the calls, nearly four times the value, because it prioritises clients where the revenue at stake justifies the cost of the call.

**Drift (July to September 2026, simulated market downturn)**

- Monitoring flagged cash share (PSI 1.02) and under-prediction: observed churn 16.4% vs 10.7% predicted.
- Ranking quality held (ROC-AUC 0.797), but calibration broke. That matters because targeting runs on predicted probabilities.
- Tenure also showed high PSI but is marked as expected drift, so it does not trigger a retrain on its own.
- Challenger v2 trained on a rolling 12 months beat the champion on the held-out September month (PR-AUC 0.518 vs 0.488, Brier 0.108 vs 0.115) and was promoted.

**Client service bot (1,200 inbox requests)**

- 82% handled with no human touch; intent model accuracy 97.3%
- 2.7 FTE of capacity released (about 376 team hours a month at 2,500 requests)
- Error rate on automated items 0.4% measured, against 2.5% assumed for manual handling
- Every exception reaches a person with a reason attached: missing account number, sender does not own the account, redemption over balance, low classifier confidence

## Run it

```bash
pip install -r requirements.txt

python -m clientpulse.pipeline   # data, train, register, monitor, retrain, score
python -m automation.run         # bot over the inbox, value report
pytest -q                        # 19 tests

uvicorn clientpulse.api:app --reload   # then open http://127.0.0.1:8000/docs
```

Or `make all`.

## Outputs

```
reports/
  pr_curve.png, calibration.png, lift_by_decile.png, shap_importance.png
  drift_psi.png, observed_vs_predicted.png
  monitoring_report.md          what drifted, what decayed, the decision
  retention_list_2026-09.csv    top 500 clients for relationship managers, with reasons
  automation/
    value_report.md             FTE, turnaround, error rate, exception reasons
    bot_run_log.csv             one row per request, with status and reason
    human_work_queue.csv        what people need to pick up
    audit_trail.csv             every action the bot took
models/
  registry.json                 versions, stages, promotion history
  v1/, v2/                      model plus metadata (lineage, metrics, drift reference)
```

## Docs

- [Model card](docs/MODEL_CARD.md): purpose, data, metrics, limits, assumptions
- [Automation design](docs/AUTOMATION_DESIGN.md): process definition and solution design for the bot
- [Building it in UiPath or Power Automate](docs/RPA_PLATFORM_GUIDE.md): how the same bot maps onto enterprise RPA platforms

## Honest limits

- Synthetic data. The patterns are designed to be realistic, not measured from a real book of clients.
- The intent classifier is trained on generated tickets, so 97% is optimistic. Real tickets would be messier.
- The registry is file based for portability. In production this would be MLflow or a managed registry.
- The bot logic is written in Python. On an enterprise RPA platform the same design maps onto queues, REFramework states and activities (see the platform guide).
