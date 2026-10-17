# Automation design: client service inbox bot

Written in the shape of a Process Definition Document (the as-is and to-be process) followed by a Solution Design Document (how the bot is built). These are the two documents an RPA team normally signs off before building.

## 1. Process definition

### As-is

A client service team works a shared mailbox. Each request is read, the account looked up in the core system, checks done by hand, the action taken in the back office, and a reply sent.

| Request type | Share of volume | Manual minutes |
|---|---:|---:|
| Statement request | 32% | 8 |
| Dividend enquiry | 22% | 10 |
| Redemption request | 17% | 15 |
| Complaint | 15% | 10 |
| Address (KYC) update | 14% | 12 |

Pain points: average 6 hours before a request is touched, rekeying errors, complaints from high-value clients sitting behind routine statement requests.

### To-be

The bot reads every message, classifies it, extracts the details, checks them, acts where it safely can and hands everything else to a person with the reason attached. People spend their time on judgement calls and on complaints.

### Why this process

High volume, rule based, structured enough to automate, and with a clear value measure. The complaint handling also gives the ML side something real to do: prioritise by churn risk.

### Out of scope

Moving money. The bot raises redemption instructions; operations executes payments.

## 2. Solution design

### Architecture

```
Shared mailbox
     |
     v
Queue (one item per email)
     |
     v
Init -> Get transaction -> Process -> Set status -> (next item) -> End
                              |
           classify (intent model, confidence gate)
           extract (account, amount, address, stock, period)
           validate (account exists, sender owns account, business rules)
           act (back office: statement, KYC staging, dividend lookup,
                redemption instruction, complaint ticket)
                              |
             churn model score for complaints -> P1 and RM alert
```

### States (REFramework)

| State | Does |
|---|---|
| Init | Load config, client master, intent model, churn model |
| Get transaction data | Pull the next queue item |
| Process | Classify, extract, validate, act |
| Set status | Successful, BusinessException, SystemException or HumanReview |
| End | Write run log, audit trail, human work queue, value report |

### Exception policy

| Type | Example | Handling |
|---|---|---|
| System | Core system timeout | Retry up to 2 times, then fail and alert support |
| Business rule | Redemption above fund balance | No retry, route to agent with reason |
| Low confidence | Intent below 0.70 | Route to agent with the top two guesses |

### Controls

| Risk | Control |
|---|---|
| Data sent to the wrong person | Sender's email must own the account before anything is shared or changed |
| Unauthorised money movement | Bot never pays; redemptions of N5m and above need four-eyes approval |
| KYC fraud by address change | Change is staged and a compliance ticket raised to check proof of address |
| Silent misrouting | Confidence gate, plus a measured misrouting rate in every run report |
| No trail | Every action written to the audit trail with time, client and detail |

### Intelligent parts

1. **Intent model.** TF-IDF on words and character n-grams into logistic regression. Character n-grams cope with typos and Pidgin. Accuracy on held-out tickets: 97.3% (optimistic, see limits).
2. **Churn-aware triage.** Complaints from clients the churn model scores at 30% risk or above, or with portfolios of N50m and above, become P1 and alert the relationship manager.

### Monitoring the bot in production

- Straight-through rate per request type, daily
- Exceptions by reason, so the top reason becomes the next improvement
- Misrouting rate from a weekly sample checked by a person
- Queue age, to catch the bot being down before clients notice
- Intent model retrained when the team has relabelled a few hundred corrected tickets

### Value (last run, 1,200 requests)

| Measure | Value |
|---|---|
| Straight-through | 82% |
| FTE capacity released | 2.7 |
| Turnaround on automated items | 6 hours to about 5 minutes |
| Error rate on automated items | 0.4% measured vs 2.5% assumed manual |

The manual minutes and the 2.5% manual error rate are assumptions. Before go-live they should be measured with a short time-and-motion study, because the savings claim is only as good as that baseline.
