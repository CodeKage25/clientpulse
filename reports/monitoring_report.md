# Monitoring report: 2026-07 to 2026-09

Model in production: **v1**, rows scored: 36,000

## Data drift (PSI vs training reference)

| Feature | PSI | Status |
|---|---:|---|
| tenure_months | 1.150 | ACT (expected, grows with time) |
| cash_share | 1.015 | ACT |
| trade_trend | 0.210 | WATCH |
| net_flow_90d_share | 0.177 | WATCH |
| trades_per_login | 0.175 | WATCH |
| service_issues_90d | 0.151 | WATCH |
| trades_90d | 0.143 | WATCH |
| failed_txn_90d | 0.121 | WATCH |
| has_outflow | 0.117 | WATCH |
| complaints_90d | 0.051 | OK |
| log_portfolio_value | 0.017 | OK |
| age | 0.008 | OK (expected, grows with time) |
| logins_30d | 0.000 | OK |
| days_since_login | 0.000 | OK |
| num_asset_classes | 0.000 | OK |
| dividend_reinvest | 0.000 | OK |
| kyc_complete | 0.000 | OK |
| is_dormant | 0.000 | OK |
| segment | 0.000 | OK |
| channel | 0.000 | OK |
| region | 0.000 | OK |

Score distribution PSI: **0.240** (WATCH)

## Performance (labels matured)

- ROC-AUC: 0.797 vs 0.781 at release
- PR-AUC: 0.499 vs 0.330 at release (not comparable on its own: it rises with the churn base rate)
- Observed churn rate: 16.4%, mean predicted: 10.7%
- Brier score: 0.1141

## Decision

**Retrain triggered.** Reasons:
- Data drift above 0.25 on: cash_share
- Under-predicting churn: observed 16.4% vs predicted 10.7%
