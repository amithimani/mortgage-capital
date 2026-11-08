"""Owner and priority for each exception: deterministic routing, never the LLM.

* **owner**: the team that remediates the category (a routing table; the PoC values are
  placeholders to be confirmed with Mortgage Capital operations).
* **priority**: a base level per category, raised one level when the amount at risk reaches
  ``ESCALATION_THRESHOLD``. Customer-detriment categories (overcharged rate, double debit) start
  at P1 because they are conduct issues, whatever the amount.
* **amount_at_risk**: derived from the engine's expected/actual values and evidence. For a rate
  breach it is the annualised interest impact on the closing balance.

The reason for the priority is stored with it, so the decision can be audited.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .engine.taxonomy import DiscrepancyKind as K
from .engine.taxonomy import ExceptionCategory as C

OWNER_BY_CATEGORY: dict[C, str] = {
    C.MISSING_PAYMENT: "collections",
    C.PARTIAL_OR_OVERPAYMENT: "collections",
    C.DUPLICATE_DIRECT_DEBIT: "payments-operations",
    C.UNIDENTIFIED_PAYMENT: "payments-operations",
    C.RATE_MARGIN_BREACH: "pricing-and-product",
    C.INVESTOR_BALANCE_BREAK: "investor-reporting",
    C.INVESTOR_POPULATION_BREAK: "investor-reporting",
    C.OTHER_NEEDS_REVIEW: "reconciliation-analyst",
}

BASE_PRIORITY: dict[C, int] = {
    C.RATE_MARGIN_BREACH: 1,
    C.DUPLICATE_DIRECT_DEBIT: 1,
    C.MISSING_PAYMENT: 2,
    C.PARTIAL_OR_OVERPAYMENT: 2,
    C.INVESTOR_BALANCE_BREAK: 2,
    C.INVESTOR_POPULATION_BREAK: 2,
    C.OTHER_NEEDS_REVIEW: 2,
    C.UNIDENTIFIED_PAYMENT: 3,
}

SLA_BUSINESS_DAYS = {1: 1, 2: 2, 3: 5}
ESCALATION_THRESHOLD = Decimal("5000.00")


def _d(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _servicing_field(exc: dict, name: str) -> Decimal | None:
    for e in exc["evidence"]:
        if e["source"] == "servicing" and name in e["fields"]:
            return _d(e["fields"][name])
    return None


def amount_at_risk(exc: dict) -> tuple[Decimal | None, str]:
    kind, exp, act = K(exc["kind"]), exc["expected"], exc["actual"]
    match kind:
        case K.UNMATCHED_EXPECTED_PAYMENT:
            return _d(exp.get("scheduled_amount")), "scheduled payment not received"
        case K.DUPLICATE_PAYMENT:
            return _d(act.get("duplicate_total")), "duplicate collections"
        case K.AMOUNT_MISMATCH | K.BALANCE_MISMATCH:
            diff = _d(act.get("difference"))
            return (abs(diff) if diff is not None else None), "absolute difference"
        case K.UNMATCHED_PAYMENT:
            return _d(act.get("amount")), "unallocated cash"
        case K.RATE_COMPONENT_MISMATCH:
            want = _d(exp.get("applied_rate", exp.get("rate")))
            got = _d(act.get("applied_rate", act.get("rate")))
            balance = _servicing_field(exc, "closing_balance")
            if None in (want, got, balance):
                return None, "rate impact not computable"
            impact = (balance * abs(got - want) / 100).quantize(Decimal("0.01"))
            return impact, "annualised interest impact on closing balance"
        case _:
            return None, "no monetary amount"


def assign(exc: dict, category: str) -> dict:
    """Return ``owner``, ``priority`` (``P1``-``P3``), SLA and the reason, for an exception."""
    cat = C(category)
    amount, basis = amount_at_risk(exc)
    level = BASE_PRIORITY[cat]
    reason = f"base P{level} for {cat.value}"
    if amount is not None and amount >= ESCALATION_THRESHOLD and level > 1:
        level -= 1
        reason += f"; escalated: amount at risk {amount} >= {ESCALATION_THRESHOLD}"
    return {
        "owner": OWNER_BY_CATEGORY[cat],
        "priority": f"P{level}",
        "sla_business_days": SLA_BUSINESS_DAYS[level],
        "amount_at_risk": str(amount) if amount is not None else None,
        "amount_basis": basis,
        "priority_reason": reason,
    }
