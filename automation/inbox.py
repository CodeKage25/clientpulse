"""Synthetic client service inbox and client master.

Mimics the shared mailbox a brokerage client service team works through every
day: statement requests, address (KYC) updates, dividend enquiries, mutual fund
redemption requests and complaints. Messages are deliberately messy: missing
account numbers, wrong ones, informal wording, Pidgin, typos.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

INTENTS = ["statement_request", "address_update", "dividend_enquiry", "redemption_request", "complaint"]

STOCKS = ["DANGCEM", "MTNN", "ZENITHBANK", "GTCO", "SEPLAT", "NESTLE", "ACCESSCORP", "UBA"]
CITIES = ["Lekki", "Ikeja", "Yaba", "Surulere", "Wuse 2", "Maitama", "Bodija", "GRA Port Harcourt"]
STREETS = ["Admiralty Way", "Allen Avenue", "Herbert Macaulay Way", "Adeola Odeku Street",
           "Aminu Kano Crescent", "Ring Road", "Awolowo Road", "Aba Road"]

TEMPLATES = {
    "statement_request": [
        "Good morning, please send me my account statement for {period}. Account {acct}.",
        "Hi team, I need my portfolio statement from {period} for a visa application. My account number is {acct}",
        "Kindly forward statement of account ({period}). Acct no: {acct}. Thanks",
        "Abeg send my statement for {period}, account {acct}",
        "I would like a copy of my investment statement covering {period}. {acct}",
    ],
    "address_update": [
        "Please update my residential address to {address}. Account {acct}.",
        "I have moved. New address: {address}. My account is {acct}",
        "Kindly change my address on your records to {address} (acct {acct})",
        "Update my KYC address please: {address}. Account number {acct}",
    ],
    "dividend_enquiry": [
        "I have not received my {stock} dividend yet. Account {acct}. Please check.",
        "When will the {stock} dividend be paid? My account is {acct}",
        "My {stock} dividend has not entered my bank account. {acct}",
        "Please confirm status of {stock} dividend payment for account {acct}",
    ],
    "redemption_request": [
        "I want to redeem N{amount} from my money market fund. Account {acct}.",
        "Please liquidate N{amount} of my mutual fund units and pay to my bank. Acct {acct}",
        "Kindly process redemption of N{amount} from the fund. Account number {acct}",
        "I need N{amount} from my fund urgently, please redeem. {acct}",
    ],
    "complaint": [
        "This is the third time your app failed when I tried to buy {stock}. Very disappointing. Account {acct}",
        "I am not happy with the service. My trade was not executed and nobody called me back. {acct}",
        "Your platform keeps logging me out and I missed a good entry on {stock}. Account {acct}. Fix this.",
        "I have been waiting two weeks for a response. I am considering moving my portfolio. Acct {acct}",
        "Una app no dey work since morning, I no fit trade. Account {acct}",
    ],
}

PERIODS = ["January to June 2026", "Q2 2026", "the last 6 months", "2025", "July 2026", "last quarter"]


@dataclass
class Email:
    message_id: str
    received_at: str
    sender: str
    subject: str
    body: str
    true_intent: str  # ground truth for evaluation only; the bot never reads it


def client_master(snapshots: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Latest snapshot per client plus the static fields a back office keeps."""
    rng = random.Random(seed)
    latest = snapshots[snapshots["snapshot_month"] == snapshots["snapshot_month"].max()].copy()
    latest["email"] = [f"{cid.lower()}@example.com" for cid in latest["client_id"]]
    latest["cscs_number"] = [str(rng.randint(10_000_000, 99_999_999)) for _ in range(len(latest))]
    latest["address"] = [f"{rng.randint(1, 80)} {rng.choice(STREETS)}, {rng.choice(CITIES)}" for _ in range(len(latest))]
    latest["fund_balance_ngn"] = (latest["portfolio_value_ngn"] * [rng.uniform(0.05, 0.5) for _ in range(len(latest))]).round(-2)
    return latest.reset_index(drop=True)


def _fill(template: str, acct: str, rng: random.Random, fund_balance: float) -> str:
    # most clients ask for part of what they hold; some ask for more than they have
    amount = max(10_000, round(fund_balance * rng.uniform(0.1, 1.25), -3))
    return template.format(
        acct=acct,
        period=rng.choice(PERIODS),
        address=f"{rng.randint(1, 120)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
        stock=rng.choice(STOCKS),
        amount=f"{amount:,.0f}",
    )


def generate_inbox(master: pd.DataFrame, n: int = 1200, seed: int = 3) -> list[Email]:
    rng = random.Random(seed)
    weights = [0.32, 0.14, 0.22, 0.17, 0.15]
    emails = []
    for i in range(n):
        intent = rng.choices(INTENTS, weights=weights)[0]
        client = master.iloc[rng.randrange(len(master))]
        acct = client["client_id"]
        roll = rng.random()
        if roll < 0.06:
            acct = "(no account number)"          # client forgot
        elif roll < 0.09:
            acct = f"MC{rng.randint(900000, 999999)}"  # typo or wrong account
        elif roll < 0.11:
            acct = master.iloc[rng.randrange(len(master))]["client_id"]  # someone else's account
        balance = float(client["fund_balance_ngn"])
        body = _fill(rng.choice(TEMPLATES[intent]), acct, rng, balance)
        if rng.random() < 0.05:  # ambiguous message mixing two requests
            other = rng.choice([x for x in INTENTS if x != intent])
            body += " Also " + _fill(rng.choice(TEMPLATES[other]), client["client_id"], rng, balance).lower()
        emails.append(
            Email(
                message_id=f"MSG-{i + 1:05d}",
                received_at=f"2026-10-{1 + i * 14 // n:02d}T{8 + rng.randint(0, 9):02d}:{rng.randint(0, 59):02d}:00",
                sender=client["email"],
                subject=body.split(".")[0][:60],
                body=body,
                true_intent=intent,
            )
        )
    return emails


def save(emails: list[Email], path: Path) -> None:
    path.write_text(json.dumps([asdict(e) for e in emails], indent=1))


def load(path: Path) -> list[Email]:
    return [Email(**e) for e in json.loads(path.read_text())]


def labelled_history(master: pd.DataFrame, n: int = 3000, seed: int = 99) -> pd.DataFrame:
    """Past tickets already classified by the service team: the training set for the intent model."""
    emails = generate_inbox(master, n=n, seed=seed)
    return pd.DataFrame({"text": [e.body for e in emails], "intent": [e.true_intent for e in emails]})
