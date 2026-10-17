# Automation value report: client service inbox bot

Requests processed: **1,200**  
Straight-through (no human touch): **82%**

## Value

| Measure | Before | After |
|---|---:|---:|
| Team hours per month on this inbox | 437.2 | 61.2 |
| Turnaround for automated requests | 6.0 hours | 5 minutes |
| Error rate on processed requests | 2.5% (assumed) | 0.4% (measured) |

**FTE capacity released: 2.69** (376.0 hours a month, 86% of the effort). That capacity moves to complaints and advisory work rather than headcount cuts.

## Outcomes

| Status | Count |
|---|---:|
| Successful | 989 |
| BusinessException | 180 |
| HumanReview | 31 |

## Top reasons a person was needed

| Reason | Count |
|---|---:|
| No account number in message; auto-reply asks client to resend | 79 |
| Requested N<amt> exceeds fund balance N<amt> | 38 |
| Account <acct> not found in client master | 32 |
| Low confidence (below 0.70) | 31 |
| Sender does not own <acct>; possible error or fraud, held for review | 30 |
| Redemption amount not found | 1 |

## Assumptions

```json
{
  "manual_minutes_per_request": {
    "statement_request": 8,
    "address_update": 12,
    "dividend_enquiry": 10,
    "redemption_request": 15,
    "complaint": 10
  },
  "exception_minutes_saved_share": 0.3,
  "monthly_request_volume": 2500,
  "productive_hours_per_fte_month": 140,
  "manual_turnaround_hours": 6.0,
  "bot_turnaround_minutes": 5.0,
  "manual_error_rate": 0.025
}
```
