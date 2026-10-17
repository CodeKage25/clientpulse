"""Client service bot, structured like UiPath's Robotic Enterprise Framework (REFramework).

States:
    Init               load config, client master, intent model, churn model
    GetTransactionData pull the next email from the queue
    Process            classify -> extract -> validate -> act
    SetStatus          Successful | BusinessException | SystemException | HumanReview
    End                write the run log, audit trail and value report

Exception policy (the heart of REFramework):
- SystemException: something technical broke. Retry up to MAX_RETRIES, then fail.
- BusinessRuleException: the request is invalid or needs judgement. No retry.
  Route to a person with the reason attached so they do not start from zero.

Controls a regulated firm would expect:
- The sender's email must own the account before anything is shared or changed.
- The bot never moves money. Redemptions become instructions for operations,
  and large ones need four-eyes approval.
- Address changes are staged for compliance review against proof of address.
- Every action lands in an audit trail.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .inbox import STOCKS, Email
from .intent import IntentModel
from .systems import BackOffice, BusinessRuleException, SystemException

MAX_RETRIES = 2
REDEMPTION_APPROVAL_NGN = 5_000_000
CHURN_PRIORITY_THRESHOLD = 0.30

ACCOUNT_RE = re.compile(r"\bMC\d{6}\b", re.I)
AMOUNT_RE = re.compile(r"N\s?([\d,]{4,})")
ADDRESS_RE = re.compile(
    r"(?:address(?: to| please)?:?|New address:|records to)\s+(.+?)(?:\.\s|\s\(|\s*\.?\s*(?:Account|Acct|My account)|$)",
    re.I,
)
PERIOD_RE = re.compile(r"(January to June 2026|Q\d 20\d\d|the last 6 months|20\d\d|July 2026|last quarter)", re.I)


@dataclass
class ItemResult:
    message_id: str
    true_intent: str
    predicted_intent: str
    confidence: float
    status: str = "Successful"
    action: str = ""
    reason: str = ""
    retries: int = 0
    priority: str = ""
    client_id: str = ""
    runtime_ms: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


class ClientServiceBot:
    def __init__(self, backoffice: BackOffice, intent_model: IntentModel, churn_scorer=None):
        self.bo = backoffice
        self.intent = intent_model
        self.churn_scorer = churn_scorer  # callable(client_id) -> probability, optional
        self.results: list[ItemResult] = []

    # Process state
    def process(self, email: Email) -> ItemResult:
        pred = self.intent.predict(email.body)
        r = ItemResult(email.message_id, email.true_intent, pred.intent, round(pred.confidence, 3))
        if not pred.auto:
            r.status, r.action = "HumanReview", "routed_to_agent"
            r.reason = f"Low confidence ({pred.confidence:.2f}); could be {pred.intent} or {pred.second_intent}"
            return r

        attempt = 0
        while True:
            try:
                self._handle(pred.intent, email, r)
                return r
            except BusinessRuleException as e:
                r.status, r.reason = "BusinessException", str(e)
                r.action = r.action or "routed_to_agent_with_reason"
                return r
            except SystemException as e:
                attempt += 1
                r.retries = attempt
                if attempt > MAX_RETRIES:
                    r.status, r.reason, r.action = "SystemException", str(e), "failed_after_retries"
                    return r

    def _owned_account(self, email: Email) -> str:
        found = ACCOUNT_RE.findall(email.body)
        if not found:
            raise BusinessRuleException("No account number in message; auto-reply asks client to resend")
        acct = found[0].upper()
        client = self.bo.get_client(acct)
        if client["email"].lower() != email.sender.lower():
            raise BusinessRuleException(f"Sender does not own {acct}; possible error or fraud, held for review")
        return acct

    def _handle(self, intent: str, email: Email, r: ItemResult) -> None:
        acct = self._owned_account(email)
        r.client_id = acct
        body = email.body

        if intent == "statement_request":
            m = PERIOD_RE.search(body)
            path = self.bo.generate_statement(acct, m.group(1) if m else "last 6 months")
            r.action, r.extras = "statement_sent", {"file": path.name}

        elif intent == "address_update":
            m = ADDRESS_RE.search(body)
            if not m or len(m.group(1)) < 8:
                raise BusinessRuleException("Could not read a full new address")
            self.bo.update_address(acct, m.group(1).strip())
            ref = self.bo.create_ticket(acct, "P3", "Verify proof of address for KYC change")
            r.action, r.extras = "address_staged_for_kyc_review", {"ticket": ref}

        elif intent == "dividend_enquiry":
            stock = next((s for s in STOCKS if s.lower() in body.lower()), None)
            if stock is None:
                raise BusinessRuleException("Stock not identified in dividend enquiry")
            status = self.bo.dividend_status(acct, stock)
            r.action, r.extras = f"dividend_reply_{status}", {"stock": stock}

        elif intent == "redemption_request":
            m = AMOUNT_RE.search(body)
            if not m:
                raise BusinessRuleException("Redemption amount not found")
            amount = float(m.group(1).replace(",", ""))
            balance = float(self.bo.master.at[acct, "fund_balance_ngn"])
            if amount > balance:
                raise BusinessRuleException(f"Requested N{amount:,.0f} exceeds fund balance N{balance:,.0f}")
            needs_approval = amount >= REDEMPTION_APPROVAL_NGN
            ref = self.bo.create_redemption_instruction(acct, amount, needs_approval)
            r.action = "redemption_pending_approval" if needs_approval else "redemption_instruction_created"
            r.extras = {"amount_ngn": amount, "ref": ref}

        elif intent == "complaint":
            risk = float(self.churn_scorer(acct)) if self.churn_scorer else None
            value = float(self.bo.master.at[acct, "portfolio_value_ngn"])
            at_risk = risk is not None and risk >= CHURN_PRIORITY_THRESHOLD
            r.priority = "P1" if at_risk or value >= 50_000_000 else "P2"
            ref = self.bo.create_ticket(acct, r.priority, body)
            r.action = "complaint_ticket_rm_alerted" if r.priority == "P1" else "complaint_ticket"
            r.extras = {"ticket": ref, "churn_risk": None if risk is None else round(risk, 3)}

    # GetTransactionData + SetStatus loop
    def run(self, queue: list[Email]) -> pd.DataFrame:
        for email in queue:
            t0 = time.perf_counter()
            result = self.process(email)
            result.runtime_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.results.append(result)
        return pd.DataFrame([asdict(r) for r in self.results])


def write_outputs(results: pd.DataFrame, backoffice: BackOffice, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "bot_run_log.csv", index=False)
    pd.DataFrame(backoffice.audit).to_csv(out_dir / "audit_trail.csv", index=False)
    queue = results[results["status"].isin(["HumanReview", "BusinessException", "SystemException"])]
    queue[["message_id", "client_id", "predicted_intent", "status", "reason"]].to_csv(
        out_dir / "human_work_queue.csv", index=False
    )
