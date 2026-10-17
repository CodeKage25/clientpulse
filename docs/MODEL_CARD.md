# Model card: client churn, 90-day horizon

## Purpose

Rank active brokerage and wealth clients by the risk that they go dormant or move most of their assets out within the next 90 days, so relationship managers can call the right clients first.

Intended users: relationship managers, client service leads, retention campaign owners.

Not intended for: credit decisions, pricing, or any action that changes a client's terms without a person involved.

## Target

`churned_90d = 1` when, in the 90 days after the snapshot, the client either has no login or trade activity, or moves out most of their portfolio.

Labels mature 90 days after each snapshot. A model scored on September data can only be evaluated from December.

## Data

Monthly snapshots, one row per client per month, January 2025 to September 2026. 12,000 clients, 252,000 rows. Fully synthetic.

| Group | Features |
|---|---|
| Profile | segment (retail, affluent, HNWI), channel, region, age, tenure |
| Portfolio | portfolio value, cash share, number of asset classes, dividend reinvestment |
| Behaviour | trades in 90 days, trade trend, logins in 30 days, days since login, net flows |
| Service | complaints, failed transactions |
| Compliance | KYC complete |

Engineered: log portfolio value, dormancy flag, outflow flag, service issues total, trades per login.

## Splits

Time based, never random. Random splits let the model see the future and overstate performance.

| Split | Months | Use |
|---|---|---|
| Train | 2025-01 to 2025-12 | fit |
| Validation | 2026-01 to 2026-03 | early stopping |
| Test | 2026-04 to 2026-06 | release metrics |
| Monitoring | 2026-07 to 2026-09 | drift and decay |

## Model

LightGBM binary classifier, 204 trees after early stopping on validation PR-AUC. A standardised logistic regression is the baseline it must beat.

## Release metrics (test window)

| | Logistic | LightGBM |
|---|---:|---:|
| PR-AUC | 0.307 | 0.330 |
| ROC-AUC | 0.764 | 0.781 |
| Brier | 0.0555 | 0.0544 |
| Lift, top 10% | 4.0x | 4.2x |
| Recall, top 10% | 40% | 42% |

PR-AUC is the headline because churn is rare (6.9%). ROC-AUC flatters weak models on imbalanced data.

The gain over logistic regression is real but modest. LightGBM picks up interactions (dormant and withdrawing at the same time, falling trades with a rising cash share) that a linear model needs hand-built features to see. If explainability requirements were stricter, the logistic model would be a defensible choice.

## Main drivers (mean absolute SHAP)

Number of asset classes, net flows, logins, cash share, age, channel, trade trend, dividend reinvestment. Each score served by the API comes with the top three drivers for that client, so a relationship manager knows what to talk about.

## Targeting rule

Contact a client when expected value is positive:

```
EV = P(churn) x save rate x annual revenue x horizon - contact cost
```

| Assumption | Value |
|---|---|
| Save rate when contacted | 25% |
| Revenue horizon | 2 years |
| Contact cost | N7,500 |
| Annual fee rate on AUM | 0.8% |
| Commission per trade | N2,500 |

These are placeholders. In practice they come from finance and from a past retention campaign, and the save rate should be measured with a holdout group that is not contacted.

## Monitoring

| Check | Signal | Threshold |
|---|---|---|
| Data drift | PSI per feature against training | 0.10 watch, 0.25 act |
| Score drift | PSI of predicted probabilities | 0.25 act |
| Ranking decay | ROC-AUC drop once labels mature | 0.03 absolute |
| Calibration | observed vs mean predicted churn | observed above 1.5x predicted |

Tenure and age drift by construction and never trigger a retrain alone.

PR-AUC is not used as a decay trigger because it rises with the base rate. In the July to September downturn, PR-AUC went up while the model was clearly under-predicting.

## Retraining and promotion

- Challenger trains on a rolling 12 months, early-stops on the next month, and both models are scored on a later month neither has seen.
- Promotion requires at least +0.01 PR-AUC on that month and no worse Brier score.
- Every promotion is written to the registry history with the reason.

Four challenger variants are compared on the held-out month: plain rolling window, with recency weighting (half-life 3 months), with Platt recalibration on the latest matured month, and both. The promoted variant is the best calibrated one whose ranking is no worse than the plain retrain.

Last cycle: recency weighting plus recalibration won (PR-AUC 0.524 vs 0.488 for v1, Brier 0.105 vs 0.115, mean predicted 15.9% against 16.3% observed) and was promoted as v2. Each version is also logged to MLflow, and promotion moves the `champion` alias.

## Fairness review

At the release operating point (top 10% flagged), recall, false positive rate and calibration were compared across segment, region, channel and age band. No group fell below 80% of overall recall or was off by more than 3 calibration points. Clients aged 60 plus get about double the false positive rate, so a retention script for that group should be low-pressure. Full table: `reports/fairness.md`.

## Known limits

- Synthetic data.
- Recalibration fitted on one month can overshoot if that month was unusual. Watch the calibration gap for the next two months after promotion.
- The fairness review uses attributes in the data. Gender and income were deliberately not generated; a real review would need them, handled under the firm's data protection policy.
- Feedback loop: once relationship managers act on the scores, saved clients look like false positives. Keep a small random holdout uncontacted to measure true lift.
