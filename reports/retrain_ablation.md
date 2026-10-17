# Retraining ablation: fixing post-shock calibration

Held-out month 2026-09, observed churn 16.3%. Lower Brier is better.

| Model | PR-AUC | ROC-AUC | Brier | Mean predicted |
|---|---:|---:|---:|---:|
| Champion v1 (no retrain) | 0.488 | 0.795 | 0.1145 | 10.5% |
| rolling window | 0.518 | 0.810 | 0.1082 | 12.5% |
| rolling window + recency weights | 0.524 | 0.813 | 0.1059 | 13.8% |
| rolling window + recalibration | 0.518 | 0.810 | 0.1058 | 15.9% |
| recency weights + recalibration **(chosen)** | 0.524 | 0.813 | 0.1050 | 15.9% |

Rule: the lowest Brier among variants whose PR-AUC is within 0.005 of the plain retrain.
