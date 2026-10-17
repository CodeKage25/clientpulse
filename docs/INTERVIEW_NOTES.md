# Interview notes: how to talk about this project

Keep it honest: a portfolio build on synthetic data, made to show how you would run the ML and automation lifecycle in a securities and wealth business.

## 60-second version

"I built ClientPulse to show the full cycle for a brokerage. A churn model on monthly client snapshots, split by time, that beats a logistic baseline on PR-AUC. Instead of calling everyone above a cut-off, it targets by expected value in Naira, which gave about four times the value of a top-10% list with half the calls. It sits in a versioned registry with drift monitoring. When I simulated a market downturn, monitoring caught that the model was under-predicting churn even though its ranking still looked fine, retrained a challenger on a rolling window, and promoted it only after it won on a month neither model had seen. On the automation side, a REFramework-style bot works a client service inbox: an intent model routes requests, controls stop it sharing data with the wrong person or moving money, and it reports 82% straight-through and about 2.7 FTE released, with every assumption written down."

## Questions to expect, and the short answer

**Why PR-AUC and not accuracy or ROC-AUC?**
With 7% churn, predicting nobody churns is 93% accurate. ROC-AUC rewards ranking the huge pool of non-churners well. PR-AUC asks how clean the top of the list is, which is what an RM calling 600 clients cares about.

**Why a time-based split?**
Random splits let the model learn from months after the ones it is tested on. That inflates scores and the model disappoints in production.

**How do you know when to retrain?**
Three signals: PSI on inputs, PSI on scores, and once labels mature, ranking and calibration. In my run the ranking held but calibration broke, which is the case people miss if they only watch one metric. I also mark features like tenure as expected drift so they do not trigger retrains forever.

**Why did PR-AUC go up during the downturn?**
PR-AUC depends on the base rate. Churn more than doubled, so precision at every threshold rose even with a worse model. That is why I judge decay on ROC-AUC and calibration.

**How would this run at Meristem?**
Snapshots built nightly in the warehouse, a scheduled job to score and write the retention list to the CRM, the API for real-time scores in the RM app, MLflow for the registry, and a monthly monitoring report once labels mature.

**What would you do before production?**
I already ran a fairness check across segment, region, channel and age band; nothing crossed the thresholds, but clients 60 plus get double the false positives, so their retention call should be gentle. Still to do: a random holdout of clients not contacted, so the save rate is measured not assumed, and agree the revenue and cost figures with finance.

**Your retrained model still under-predicted. What did you do?**
A plain rolling retrain only had one post-shock month, so it predicted 12.5% against 16.3% observed. I compared four variants on a month none had seen: recency weighting improved ranking, recalibration on the latest matured month fixed the level, and the combination won on both. Mean predicted came to 15.9%.

**How do you serve and track models?**
FastAPI behind a slim Docker image, health check for the load balancer, Render blueprint for deployment. Every version is logged to MLflow with metrics, lineage tags and a registered model version; promotion moves the champion alias.

**What goes wrong with RPA in practice?**
Bots breaking when a screen changes, exceptions nobody owns, and savings claims nobody believes. The answers: APIs over screen scraping where possible, every exception lands with a named team and a reason, and a measured baseline before go-live.

**How do you measure FTE savings?**
Time-and-motion baseline per request type, multiply by volume handled straight through, add a partial saving on exceptions because people start from the bot's diagnosis, divide by productive hours per FTE. Frame it as capacity released to higher-value work.

**Why not let the bot process redemptions fully?**
Money movement needs segregation of duties. The bot validates and prepares; a person approves above a threshold and operations pays.

## Numbers to remember

- Churn: PR-AUC 0.330 vs 0.307 baseline, lift 4.2x in the top 10%
- Targeting: N13.5m a month with 612 calls vs N3.5m with 1,200 calls
- Drift: observed 16.4% vs predicted 10.7%; cash share PSI 1.02
- Retrain: v2 PR-AUC 0.524 vs 0.488 on held-out September; mean predicted 15.9% vs 16.3% observed
- Bot: 82% straight-through, 2.7 FTE, 0.4% error rate measured
- Segmentation: 4 behavioural segments; a small 'drifting away' group churns at 45%
