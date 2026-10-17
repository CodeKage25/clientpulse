import pandas as pd
import pytest

from automation.bot import ClientServiceBot
from automation.inbox import Email, labelled_history
from automation.intent import IntentModel
from automation.systems import BackOffice


@pytest.fixture(scope="module")
def master():
    return pd.DataFrame(
        {
            "client_id": ["MC100001", "MC100002"],
            "email": ["a@example.com", "b@example.com"],
            "portfolio_value_ngn": [2_000_000.0, 90_000_000.0],
            "fund_balance_ngn": [500_000.0, 20_000_000.0],
            "kyc_complete": [1, 0],
            "address": ["1 Allen Avenue, Ikeja", "2 Ring Road, Bodija"],
            # columns the generator uses for history; values irrelevant here
            "segment": ["retail", "hnwi"],
        }
    )


@pytest.fixture(scope="module")
def intent(master):
    return IntentModel().fit(labelled_history(master, n=1500))


def make_bot(master, intent, tmp_path, failure_rate=0.0, churn=0.05):
    bo = BackOffice(master.copy(), tmp_path, failure_rate=failure_rate)
    return ClientServiceBot(bo, intent, churn_scorer=lambda _: churn), bo


def mail(body, sender="a@example.com", intent="statement_request"):
    return Email("MSG-1", "2026-10-01T09:00:00", sender, body[:40], body, intent)


def test_statement_happy_path(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path)
    r = bot.process(mail("Kindly forward statement of account (Q2 2026). Acct no: MC100001. Thanks"))
    assert r.status == "Successful" and r.action == "statement_sent"
    assert (tmp_path / "statements" / "MC100001_statement.txt").exists()


def test_sender_must_own_account(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path)
    r = bot.process(mail("Please send me my account statement for 2025. Account MC100002.", sender="a@example.com"))
    assert r.status == "BusinessException" and "does not own" in r.reason


def test_redemption_over_balance_is_held(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path)
    r = bot.process(mail("I want to redeem N900,000 from my money market fund. Account MC100001.",
                         intent="redemption_request"))
    assert r.status == "BusinessException" and "exceeds fund balance" in r.reason


def test_large_redemption_needs_four_eyes(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path)
    r = bot.process(mail("Kindly process redemption of N6,000,000 from the fund. Account number MC100002",
                         sender="b@example.com", intent="redemption_request"))
    assert r.status == "Successful" and r.action == "redemption_pending_approval"


def test_complaint_from_at_risk_client_is_p1(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path, churn=0.45)
    r = bot.process(mail("I am not happy with the service. My trade was not executed and nobody called me back. MC100001",
                         intent="complaint"))
    assert r.priority == "P1" and r.action == "complaint_ticket_rm_alerted"


def test_system_errors_retry_then_fail(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path, failure_rate=1.0)
    r = bot.process(mail("Please send me my account statement for 2025. Account MC100001."))
    assert r.status == "SystemException" and r.retries == 3


def test_missing_account_number_goes_to_agent(master, intent, tmp_path):
    bot, _ = make_bot(master, intent, tmp_path)
    r = bot.process(mail("Please send me my account statement for 2025. Account (no account number)."))
    assert r.status == "BusinessException" and "No account number" in r.reason
