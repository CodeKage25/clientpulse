# Building the same bot in UiPath or Power Automate

The Python bot in `automation/` is the design. This guide maps it onto the two platforms the role names, so the same thinking carries over and you have something small but real built on each.

Both can be done from a Mac in the browser. Menu names shift between versions, so treat the steps as a map rather than click-by-click instructions.

## How the pieces map

| This repo | UiPath | Power Automate |
|---|---|---|
| Inbox email | Outlook or Gmail activities, or an email trigger | "When a new email arrives" trigger |
| Queue of items | Orchestrator queue | Dataverse table or SharePoint list as a work queue |
| REFramework states | REFramework template (Init, Get Transaction Data, Process, Set Transaction Status) | Scopes inside one flow: Try, Catch, Finally |
| `BusinessRuleException` | Throw BusinessRuleException, set item to Failed with Business type | Terminate or route branch with a reason column |
| `SystemException` with retry | Queue retry setting, automatic for system exceptions | Retry policy on each action |
| Intent model | Communications Mining, or call this repo's API over HTTP | AI Builder category classification, or HTTP call to this repo's API |
| Field extraction | Regex activities, Document Understanding for attachments | Expressions, AI Builder entity extraction |
| Churn score | HTTP Request to `/score` | HTTP action to `/score` |
| Audit trail | Orchestrator logs plus a log table | Run history plus a log table |
| Value report | Orchestrator queue analytics, Insights | Power BI on the log table |

## Exercise 1: Power Automate (about 2 hours)

Goal: a cloud flow that triages a client service inbox and logs every item.

1. Sign up for a Power Automate trial with a work or school account (a personal Gmail will not work for most trials; a free Microsoft 365 developer tenant is the usual route).
2. Create a SharePoint list called `ServiceQueue` with columns: MessageId, Sender, Body, Intent, Confidence, Status, Reason, ClientId, Priority.
3. New automated cloud flow, trigger "When a new email arrives" on a test mailbox.
4. Classify the email. Two options:
   - AI Builder category classification trained on 50 or more labelled examples from `reports/automation/inbox.json`
   - Or a Compose action with keyword rules, which is fine for a first pass
5. Extract the account number with an expression or a small regex via an Office Script.
6. Condition branches per intent. For each, write the outcome to `ServiceQueue` with Status and Reason.
7. Add a Scope called Try around the processing and a second Scope called Catch configured to run after Try fails. The Catch writes Status = SystemException.
8. Set a retry policy on the action that would call the back office.
9. For complaints, add an HTTP action that posts the client snapshot to this repo's `/score` endpoint (run it through a tunnel such as ngrok for the demo) and sets Priority to P1 when risk is 0.30 or above. Note: the HTTP connector is a premium connector.
10. Build a simple Power BI or SharePoint view: items by status, straight-through rate, top exception reasons.

What to say about it: "I built the triage flow in Power Automate with a work-queue list, Try and Catch scopes for system errors, business-rule routing, and an HTTP call out to my churn model for complaint priority."

## Exercise 2: UiPath (about 3 hours)

Goal: the REFramework version, which is what UiPath shops look for.

1. Create a free UiPath Automation Cloud account (Community). Studio Web runs in the browser; full Studio needs Windows.
2. In Orchestrator, create a queue called `ClientServiceQueue`. Set max retries to 2 so system exceptions retry automatically.
3. Dispatcher process: read emails (or the `inbox.json` file from this repo), add one queue item per email with Body and Sender as specific content.
4. Performer process from the REFramework template (available in full Studio; in Studio Web, build the same states with a loop over queue items):
   - Init: read config (approval threshold, confidence threshold)
   - Get Transaction Data: Get Transaction Item
   - Process: classify (HTTP call to an intent endpoint or keyword rules), extract account with Matches (regex), validate, act
   - Set Transaction Status: Successful, or Failed with Business or Application exception type
5. Throw a BusinessRuleException for: no account number, sender does not own account, redemption above balance. Each one carries its reason string.
6. Use Orchestrator queue reporting to show success rate and exception reasons.

What to say about it: "I built a dispatcher and performer on an Orchestrator queue using REFramework, with business exceptions routed to people with reasons and application exceptions retried by the queue."

## Talking about RPA honestly in the interview

Do not claim years of UiPath. Say what is true:

> "My automation work has mostly been in code: agents, pipelines and integrations. For this role I built the same intelligent automation pattern on Power Automate and UiPath to make sure I understand how queues, REFramework and exception handling work on enterprise platforms. The design questions are the same ones I deal with in code: what is safe to automate, what needs a person, how you retry, and how you prove the value."

That answer is stronger than bluffing, because the follow-up questions will be about design and controls, which you can answer in depth.
