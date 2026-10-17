"""Stand-ins for the back-office systems a bot would drive.

In UiPath or Power Automate these would be UI automation against the core
brokerage system, API calls, or database activities. Here they are small
Python classes with the same failure behaviour: occasional timeouts that are
worth retrying, and business rejections that are not.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import pandas as pd


class SystemException(Exception):
    """Technical failure (timeout, app not responding). Retry it."""


class BusinessRuleException(Exception):
    """The request itself is invalid or needs a human decision. Never retry."""


class BackOffice:
    def __init__(self, master: pd.DataFrame, out_dir: Path, failure_rate: float = 0.03, seed: int = 5):
        self.master = master.set_index("client_id")
        self.out_dir = out_dir
        self.rng = random.Random(seed)
        self.failure_rate = failure_rate
        self.audit: list[dict] = []
        (out_dir / "statements").mkdir(parents=True, exist_ok=True)

    def _flaky(self, op: str) -> None:
        if self.rng.random() < self.failure_rate:
            raise SystemException(f"{op}: core system timed out")

    def get_client(self, client_id: str) -> pd.Series:
        self._flaky("lookup")
        if client_id not in self.master.index:
            raise BusinessRuleException(f"Account {client_id} not found in client master")
        return self.master.loc[client_id]

    def generate_statement(self, client_id: str, period: str) -> Path:
        self._flaky("statement")
        c = self.master.loc[client_id]
        path = self.out_dir / "statements" / f"{client_id}_statement.txt"
        path.write_text(
            f"Statement of account (demo)\nClient: {client_id}\nPeriod: {period}\n"
            f"Portfolio value: N{c['portfolio_value_ngn']:,.0f}\nFund balance: N{c['fund_balance_ngn']:,.0f}\n"
        )
        self._log("statement_generated", client_id, period)
        return path

    def update_address(self, client_id: str, new_address: str) -> None:
        self._flaky("address_update")
        old = self.master.at[client_id, "address"]
        self.master.at[client_id, "address"] = new_address
        self._log("address_updated", client_id, f"{old} -> {new_address}")

    def dividend_status(self, client_id: str, stock: str) -> str:
        self._flaky("dividend_lookup")
        # demo rule: unclaimed if bank details missing (proxied by KYC), else paid
        paid = bool(self.master.at[client_id, "kyc_complete"])
        return "paid" if paid else "held_pending_kyc"

    def create_redemption_instruction(self, client_id: str, amount: float, needs_approval: bool) -> str:
        self._flaky("redemption")
        ref = f"RED-{client_id}-{int(amount)}"
        self._log("redemption_instruction", client_id, f"N{amount:,.0f} approval_required={needs_approval}")
        return ref

    def create_ticket(self, client_id: str, priority: str, summary: str) -> str:
        self._flaky("ticket")
        ref = f"TKT-{len(self.audit) + 1:05d}"
        self._log("ticket_created", client_id, f"{priority}: {summary[:60]}")
        return ref

    def _log(self, action: str, client_id: str, detail: str) -> None:
        self.audit.append({"at": datetime.now().isoformat(timespec="seconds"), "action": action,
                           "client_id": client_id, "detail": detail})
