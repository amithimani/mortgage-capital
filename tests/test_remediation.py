"""G1: deterministic owner / priority / amount at risk."""

from __future__ import annotations

import pytest
from conftest import inputs, investor, payment, servicing

from mc_recon.engine import reconcile
from mc_recon.engine.taxonomy import ExceptionCategory
from mc_recon.remediation import ESCALATION_THRESHOLD, OWNER_BY_CATEGORY, assign


def _exc(**kw):
    d = reconcile(inputs(**kw)).discrepancies
    assert len(d) == 1
    return d[0].model_dump(mode="json")


def test_every_category_has_an_owner():
    assert set(OWNER_BY_CATEGORY) == set(ExceptionCategory)


def test_missing_payment_is_p2_collections_with_scheduled_amount():
    exc = _exc(servicing_rows=[servicing()], investor_rows=[investor()])
    out = assign(exc, "MISSING_PAYMENT")
    assert out["owner"] == "collections"
    assert (out["priority"], out["sla_business_days"]) == ("P2", 2)
    assert out["amount_at_risk"] == "1000.00"


def test_large_amount_escalates_one_level():
    from decimal import Decimal

    big = Decimal(ESCALATION_THRESHOLD) + 1
    exc = _exc(
        servicing_rows=[servicing(scheduled_amount=big)],
        investor_rows=[investor(scheduled_amount=big)],
    )
    out = assign(exc, "MISSING_PAYMENT")
    assert out["priority"] == "P1" and "escalated" in out["priority_reason"]


def test_rate_breach_is_p1_with_annualised_impact():
    from decimal import Decimal

    exc = _exc(
        servicing_rows=[servicing(applied_rate=Decimal("5.75"))],
        payment_rows=[payment()],
        investor_rows=[investor(reported_rate=Decimal("5.75"))],
    )
    out = assign(exc, "RATE_MARGIN_BREACH")
    assert (out["owner"], out["priority"]) == ("pricing-and-product", "P1")
    assert out["amount_at_risk"] == "750.00"  # 150,000 x 0.50% per year
    assert "annualised" in out["amount_basis"]


def test_duplicate_debit_is_p1_regardless_of_amount():
    from datetime import date

    exc = _exc(
        servicing_rows=[servicing()],
        payment_rows=[payment("T1", row=1), payment("T2", row=2, value_date=date(2026, 9, 16))],
        investor_rows=[investor()],
    )
    out = assign(exc, "DUPLICATE_DIRECT_DEBIT")
    assert (out["owner"], out["priority"], out["amount_at_risk"]) == (
        "payments-operations",
        "P1",
        "1000.00",
    )


def test_unknown_category_rejected():
    exc = _exc(servicing_rows=[servicing()], investor_rows=[investor()])
    with pytest.raises(ValueError):
        assign(exc, "NOT_A_CATEGORY")
