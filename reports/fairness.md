# Fairness and group error analysis: v1, 2026-04 to 2026-06

Operating point: top 10% of clients by churn risk. Overall recall 42.3%.
Flag when recall is below 80% of overall, or calibration is off by more than 3 points (groups of 300+ rows).

## Age band

| Group | Rows | Churn rate | Recall | vs overall | False positive rate | Calibration gap | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| 30 to 44 | 11,115 | 6.4% | 38.2% | 0.90 | 6.2% | -0.2 pts | ok |
| 45 to 59 | 11,067 | 6.4% | 39.2% | 0.93 | 6.1% | -0.3 pts | ok |
| 60 plus | 8,031 | 8.7% | 53.1% | 1.26 | 13.3% | -0.1 pts | ok |
| under 30 | 5,787 | 6.0% | 34.8% | 0.82 | 5.7% | -0.0 pts | ok |

## Channel

| Group | Rows | Churn rate | Recall | vs overall | False positive rate | Calibration gap | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| branch | 7,080 | 6.9% | 50.2% | 1.19 | 7.2% | -0.0 pts | ok |
| mobile_app | 18,990 | 7.2% | 39.7% | 0.94 | 8.8% | -0.2 pts | ok |
| relationship_manager | 2,667 | 3.4% | 34.8% | 0.82 | 2.9% | +0.2 pts | ok |
| web | 7,263 | 7.2% | 42.7% | 1.01 | 6.8% | -0.5 pts | ok |

## Region

| Group | Rows | Churn rate | Recall | vs overall | False positive rate | Calibration gap | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| abuja | 6,063 | 6.7% | 39.9% | 0.94 | 8.0% | +0.1 pts | ok |
| east | 2,151 | 6.3% | 43.4% | 1.03 | 6.9% | -0.1 pts | ok |
| lagos | 17,457 | 6.8% | 42.4% | 1.00 | 7.4% | -0.2 pts | ok |
| north | 2,583 | 6.5% | 42.5% | 1.01 | 8.1% | +0.2 pts | ok |
| south_south | 3,429 | 8.0% | 42.0% | 0.99 | 7.8% | -1.1 pts | ok |
| south_west | 4,317 | 7.0% | 44.6% | 1.05 | 7.8% | +0.0 pts | ok |

## Segment

| Group | Rows | Churn rate | Recall | vs overall | False positive rate | Calibration gap | Flag |
|---|---:|---:|---:|---:|---:|---:|---|
| affluent | 6,642 | 5.9% | 38.6% | 0.91 | 5.9% | +0.0 pts | ok |
| hnwi | 1,407 | 4.3% | 45.0% | 1.06 | 3.0% | -0.2 pts | ok |
| retail | 27,951 | 7.2% | 42.9% | 1.01 | 8.3% | -0.2 pts | ok |

## Summary

No group crosses either threshold.

Low recall in a group with low churn is often expected: a single top-10% cut-off is dominated by the groups where churn is common. The fix, if the business wants equal reach, is a per-group operating point or a quota, which is a business decision to make explicitly, not something to hide in the model.
