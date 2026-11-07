"""Deterministic reconciliation: match, net, compare.

Pure functions over :class:`LoadedInputs`. No I/O, no clock, no randomness, no LLM — the
same inputs and config always produce byte-identical results (including discrepancy ids
and ordering).

Rules, per servicing loan:

1. **Payments** — payments for the loan whose value date is within
   ``±match_window_days`` of the due date are candidates.
   * none -> ``UNMATCHED_EXPECTED_PAYMENT``
   * direct debits sharing ``(mandate_ref, amount)`` within ``duplicate_window_days`` of the
     first -> the later ones are ``DUPLICATE_PAYMENT`` and are excluded from netting
   * net of the remaining payments vs scheduled amount beyond ``amount_tolerance``
     -> ``AMOUNT_MISMATCH`` (split payments that net to the schedule are fine)
   * payments for the loan outside the window -> ``UNMATCHED_PAYMENT``
2. **Rate components** — ``applied_rate`` vs ``base_rate + margin`` and investor-reported
   rate/margin vs servicing, beyond ``rate_tolerance`` -> ``RATE_COMPONENT_MISMATCH``.
3. **Balance** — investor vs servicing closing balance beyond ``balance_tolerance``
   -> ``BALANCE_MISMATCH``.

Across the book: payments for unknown loans -> ``UNMATCHED_PAYMENT``; loans present in only
one of servicing / investor -> ``POPULATION_MISMATCH``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from decimal import Decimal

from .models import (
    Discrepancy,
    EvidenceRef,
    InvestorRecord,
    LoadedInputs,
    PaymentRecord,
    ReconciliationResult,
    ReconConfig,
    ServicingRecord,
)
from .schema import Source
from .taxonomy import DiscrepancyKind as K

ENGINE_VERSION = "0.1.0"

_DD_CHANNELS = frozenset({"DD", "DIRECT DEBIT", "DIRECTDEBIT"})


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"


def _ev_servicing(s: ServicingRecord) -> EvidenceRef:
    return EvidenceRef(
        source=Source.SERVICING,
        row=s.row,
        key=s.loan_id,
        fields={
            "due_date": s.due_date.isoformat(),
            "scheduled_amount": str(s.scheduled_amount),
            "closing_balance": str(s.closing_balance),
            "base_rate": str(s.base_rate),
            "margin": str(s.margin),
            "applied_rate": str(s.applied_rate),
            "mandate_ref": s.mandate_ref or "",
        },
    )


def _ev_payment(p: PaymentRecord) -> EvidenceRef:
    return EvidenceRef(
        source=Source.PAYMENTS,
        row=p.row,
        key=p.txn_id,
        fields={
            "loan_id": p.loan_id,
            "value_date": p.value_date.isoformat(),
            "amount": str(p.amount),
            "channel": p.channel,
            "mandate_ref": p.mandate_ref or "",
        },
    )


def _ev_investor(i: InvestorRecord) -> EvidenceRef:
    return EvidenceRef(
        source=Source.INVESTOR,
        row=i.row,
        key=i.loan_id,
        fields={
            "period_end": i.period_end.isoformat(),
            "reported_rate": str(i.reported_rate),
            "reported_margin": str(i.reported_margin),
            "scheduled_amount": str(i.scheduled_amount),
            "closing_balance": str(i.closing_balance),
        },
    )


def _make(
    kind: K,
    loan_id: str | None,
    summary: str,
    evidence: Iterable[EvidenceRef],
    *,
    scope: str = "",
    expected: dict[str, str] | None = None,
    actual: dict[str, str] | None = None,
) -> Discrepancy:
    ev = tuple(evidence)
    return Discrepancy(
        id=Discrepancy.make_id(kind, loan_id, scope, ev),
        kind=kind,
        loan_id=loan_id,
        summary=summary,
        expected=expected or {},
        actual=actual or {},
        evidence=ev,
    )


def _is_direct_debit(p: PaymentRecord) -> bool:
    return p.channel.strip().upper() in _DD_CHANNELS and bool(p.mandate_ref)


def _find_duplicates(
    payments: list[PaymentRecord], cfg: ReconConfig
) -> list[tuple[PaymentRecord, list[PaymentRecord]]]:
    """Group direct debits by (mandate, amount); later collections inside the window are dupes."""
    groups: dict[tuple[str, Decimal], list[PaymentRecord]] = defaultdict(list)
    for p in payments:
        if _is_direct_debit(p):
            groups[(p.mandate_ref or "", p.amount)].append(p)
    found = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda p: (p.value_date, p.row))
        first, rest = group[0], group[1:]
        dupes = [
            p for p in rest if (p.value_date - first.value_date).days <= cfg.duplicate_window_days
        ]
        if dupes:
            found.append((first, dupes))
    return found


def _check_payments(
    s: ServicingRecord, payments: list[PaymentRecord], cfg: ReconConfig
) -> list[Discrepancy]:
    out: list[Discrepancy] = []
    in_window = [
        p for p in payments if abs((p.value_date - s.due_date).days) <= cfg.match_window_days
    ]
    outside = [p for p in payments if p not in in_window]

    for p in outside:
        out.append(
            _make(
                K.UNMATCHED_PAYMENT,
                s.loan_id,
                f"Payment {p.txn_id} of {_fmt(p.amount)} for {s.loan_id} on "
                f"{p.value_date.isoformat()} is outside ±{cfg.match_window_days} days of due date "
                f"{s.due_date.isoformat()}.",
                [_ev_payment(p), _ev_servicing(s)],
                scope="outside_window",
                actual={"value_date": p.value_date.isoformat(), "amount": str(p.amount)},
                expected={"due_date": s.due_date.isoformat()},
            )
        )

    if not in_window:
        out.append(
            _make(
                K.UNMATCHED_EXPECTED_PAYMENT,
                s.loan_id,
                f"No payment found for {s.loan_id} within ±{cfg.match_window_days} days of due "
                f"date {s.due_date.isoformat()} (scheduled {_fmt(s.scheduled_amount)}).",
                [_ev_servicing(s)],
                expected={"scheduled_amount": str(s.scheduled_amount)},
                actual={"received": "0.00"},
            )
        )
        return out

    duplicate_rows: set[int] = set()
    for first, dupes in _find_duplicates(in_window, cfg):
        duplicate_rows.update(p.row for p in dupes)
        total_dup = sum((p.amount for p in dupes), Decimal(0))
        out.append(
            _make(
                K.DUPLICATE_PAYMENT,
                s.loan_id,
                f"{len(dupes)} duplicate direct debit(s) totalling {_fmt(total_dup)} for "
                f"{s.loan_id} on mandate {first.mandate_ref}: original {first.txn_id}, "
                f"duplicate(s) {', '.join(p.txn_id for p in dupes)}.",
                [_ev_servicing(s), _ev_payment(first), *(_ev_payment(p) for p in dupes)],
                expected={"collections": "1", "amount": str(first.amount)},
                actual={"collections": str(1 + len(dupes)), "duplicate_total": str(total_dup)},
            )
        )

    kept = [p for p in in_window if p.row not in duplicate_rows]
    net = sum((p.amount for p in kept), Decimal(0))
    if abs(net - s.scheduled_amount) > cfg.amount_tolerance:
        diff = net - s.scheduled_amount
        out.append(
            _make(
                K.AMOUNT_MISMATCH,
                s.loan_id,
                f"Net received {_fmt(net)} for {s.loan_id} differs from scheduled "
                f"{_fmt(s.scheduled_amount)} by {_fmt(diff)}.",
                [_ev_servicing(s), *(_ev_payment(p) for p in kept)],
                expected={"scheduled_amount": str(s.scheduled_amount)},
                actual={"net_received": str(net), "difference": str(diff)},
            )
        )
    return out


def _check_rates(
    s: ServicingRecord, inv: InvestorRecord | None, cfg: ReconConfig
) -> list[Discrepancy]:
    out: list[Discrepancy] = []
    contractual = s.base_rate + s.margin
    if abs(s.applied_rate - contractual) > cfg.rate_tolerance:
        out.append(
            _make(
                K.RATE_COMPONENT_MISMATCH,
                s.loan_id,
                f"Applied rate {s.applied_rate}% for {s.loan_id} does not equal base "
                f"{s.base_rate}% + margin {s.margin}% = {contractual}%.",
                [_ev_servicing(s)],
                scope="servicing_internal",
                expected={"applied_rate": str(contractual)},
                actual={"applied_rate": str(s.applied_rate)},
            )
        )
    if inv is not None and (
        abs(inv.reported_rate - s.applied_rate) > cfg.rate_tolerance
        or abs(inv.reported_margin - s.margin) > cfg.rate_tolerance
    ):
        out.append(
            _make(
                K.RATE_COMPONENT_MISMATCH,
                s.loan_id,
                f"Investor report for {s.loan_id} shows rate {inv.reported_rate}% / margin "
                f"{inv.reported_margin}%; servicing shows {s.applied_rate}% / {s.margin}%.",
                [_ev_servicing(s), _ev_investor(inv)],
                scope="investor_vs_servicing",
                expected={"rate": str(s.applied_rate), "margin": str(s.margin)},
                actual={"rate": str(inv.reported_rate), "margin": str(inv.reported_margin)},
            )
        )
    return out


def _check_balance(s: ServicingRecord, inv: InvestorRecord, cfg: ReconConfig) -> list[Discrepancy]:
    diff = inv.closing_balance - s.closing_balance
    if abs(diff) <= cfg.balance_tolerance:
        return []
    return [
        _make(
            K.BALANCE_MISMATCH,
            s.loan_id,
            f"Investor closing balance {_fmt(inv.closing_balance)} for {s.loan_id} differs from "
            f"servicing {_fmt(s.closing_balance)} by {_fmt(diff)}.",
            [_ev_servicing(s), _ev_investor(inv)],
            expected={"closing_balance": str(s.closing_balance)},
            actual={"closing_balance": str(inv.closing_balance), "difference": str(diff)},
        )
    ]


def reconcile(inputs: LoadedInputs, config: ReconConfig | None = None) -> ReconciliationResult:
    cfg = config or ReconConfig()
    servicing = {s.loan_id: s for s in inputs.servicing}
    investor = {i.loan_id: i for i in inputs.investor}
    payments: dict[str, list[PaymentRecord]] = defaultdict(list)
    for p in sorted(inputs.payments, key=lambda p: p.row):
        payments[p.loan_id].append(p)

    found: list[Discrepancy] = []

    for loan_id in sorted(set(payments) - set(servicing)):
        for p in payments[loan_id]:
            found.append(
                _make(
                    K.UNMATCHED_PAYMENT,
                    loan_id,
                    f"Payment {p.txn_id} of {_fmt(p.amount)} references loan {loan_id}, which is "
                    "not in the servicing extract.",
                    [_ev_payment(p)],
                    scope="unknown_loan",
                    actual={"amount": str(p.amount)},
                )
            )

    for loan_id in sorted(set(servicing) - set(investor)):
        found.append(
            _make(
                K.POPULATION_MISMATCH,
                loan_id,
                f"Loan {loan_id} is in the servicing extract but missing from the investor report.",
                [_ev_servicing(servicing[loan_id])],
                scope="missing_from_investor",
            )
        )
    for loan_id in sorted(set(investor) - set(servicing)):
        found.append(
            _make(
                K.POPULATION_MISMATCH,
                loan_id,
                f"Loan {loan_id} is in the investor report but not in the servicing extract.",
                [_ev_investor(investor[loan_id])],
                scope="missing_from_servicing",
            )
        )

    matched_payments = 0
    for loan_id in sorted(servicing):
        s = servicing[loan_id]
        inv = investor.get(loan_id)
        loan_payments = payments.get(loan_id, [])
        matched_payments += sum(
            1
            for p in loan_payments
            if abs((p.value_date - s.due_date).days) <= cfg.match_window_days
        )
        found.extend(_check_payments(s, loan_payments, cfg))
        found.extend(_check_rates(s, inv, cfg))
        if inv is not None:
            found.extend(_check_balance(s, inv, cfg))

    found.sort(key=lambda d: (d.loan_id or "", d.kind.value, d.id))
    by_kind = Counter(d.kind.value for d in found)
    stats = {
        "servicing_loans": len(servicing),
        "payments": len(inputs.payments),
        "payments_in_window": matched_payments,
        "investor_loans": len(investor),
        "discrepancies": len(found),
        **{f"kind:{k}": v for k, v in sorted(by_kind.items())},
    }
    return ReconciliationResult(
        engine_version=ENGINE_VERSION,
        config=cfg,
        file_hashes=inputs.file_hashes,
        discrepancies=tuple(found),
        stats=stats,
    )
