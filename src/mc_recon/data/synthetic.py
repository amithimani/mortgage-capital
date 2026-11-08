"""Seeded synthetic reconciliation sets with a documented error catalog.

Each *pair* directory contains the three raw files plus ground truth::

    servicing_extract.csv   payments.csv   investor_report.csv
    expected_mapping.json   # the correct ColumnMapping for this pair's headers
    seeded_errors.json      # every planted error (with expected kind + category) and benign noise

Generation is fully deterministic for a given :class:`PairSpec` (seeded ``random.Random``,
``Decimal`` arithmetic, fixed line endings), so the committed fixtures can be regenerated
byte-for-byte — ``tests/test_synthetic.py`` enforces that.

Errors (must raise exactly one discrepancy each):
    missing_payment, duplicate_direct_debit, rate_margin_breach, partial_payment,
    investor_balance_break, unidentified_payment
Benign noise (must raise nothing):
    loan_ref_formatting, split_payment, balance_rounding, rate_precision, value_date_shift,
    plus automatic weekend_roll entries for every due date falling on a weekend.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from ..engine.schema import ColumnMapping, Source, SourceMapping
from ..engine.taxonomy import DiscrepancyKind, ExceptionCategory

GENERATOR_VERSION = "1"

FILE_NAMES: dict[Source, str] = {
    Source.SERVICING: "servicing_extract.csv",
    Source.PAYMENTS: "payments.csv",
    Source.INVESTOR: "investor_report.csv",
}


def pair_paths(pair_dir: Path) -> dict[Source, Path]:
    """Raw input file paths for a generated pair directory."""
    return {source: Path(pair_dir) / name for source, name in FILE_NAMES.items()}


def load_expected_mapping(pair_dir: Path) -> ColumnMapping:
    return ColumnMapping.model_validate_json((Path(pair_dir) / "expected_mapping.json").read_text())


def load_catalog(pair_dir: Path) -> dict:
    return json.loads((Path(pair_dir) / "seeded_errors.json").read_text())


ERROR_TYPES: dict[str, tuple[DiscrepancyKind, ExceptionCategory, str]] = {
    "missing_payment": (
        DiscrepancyKind.UNMATCHED_EXPECTED_PAYMENT,
        ExceptionCategory.MISSING_PAYMENT,
        "Scheduled payment has no corresponding payment in the payments file.",
    ),
    "duplicate_direct_debit": (
        DiscrepancyKind.DUPLICATE_PAYMENT,
        ExceptionCategory.DUPLICATE_DIRECT_DEBIT,
        "Direct debit collected twice on the same mandate one day apart.",
    ),
    "rate_margin_breach": (
        DiscrepancyKind.RATE_COMPONENT_MISMATCH,
        ExceptionCategory.RATE_MARGIN_BREACH,
        "Applied rate charged at margin + 0.50pp above contractual base + margin.",
    ),
    "partial_payment": (
        DiscrepancyKind.AMOUNT_MISMATCH,
        ExceptionCategory.PARTIAL_OR_OVERPAYMENT,
        "Borrower paid roughly 60% of the scheduled amount.",
    ),
    "investor_balance_break": (
        DiscrepancyKind.BALANCE_MISMATCH,
        ExceptionCategory.INVESTOR_BALANCE_BREAK,
        "Investor-reported closing balance overstated by 1,250.00.",
    ),
    "unidentified_payment": (
        DiscrepancyKind.UNMATCHED_PAYMENT,
        ExceptionCategory.UNIDENTIFIED_PAYMENT,
        "Payment received against a loan reference not in the servicing book.",
    ),
}

NOISE_TYPES: dict[str, str] = {
    "loan_ref_formatting": "Loan reference in payments file is lower-case with trailing space.",
    "split_payment": "Scheduled amount paid as two Faster Payments that net to the schedule.",
    "balance_rounding": "Investor closing balance differs by 0.01 (rounding).",
    "rate_precision": "Investor reports rate at 4dp, 0.004pp from servicing (within tolerance).",
    "value_date_shift": "Payment received 2 days after due date (within match window).",
    "weekend_roll": "Due date falls on a weekend; payment value-dated the next Monday.",
}


@dataclass(frozen=True)
class HeaderVariant:
    name: str
    servicing: dict[str, str]
    payments: dict[str, str]
    investor: dict[str, str]
    payments_date_format: str
    extra_servicing: tuple[str, ...] = ()
    extra_payments: tuple[str, ...] = ()


VARIANT_A = HeaderVariant(
    name="A",
    servicing={
        "loan_id": "Acct No",
        "pool_id": "Pool",
        "due_date": "Pmt Due Dt",
        "scheduled_amount": "Sched Pmt Amt",
        "closing_balance": "Cur Bal",
        "base_rate": "Base Rt %",
        "margin": "Margin %",
        "applied_rate": "Applied Rt %",
        "mandate_ref": "DD Mandate",
    },
    payments={
        "txn_id": "TXN_ID",
        "loan_id": "LOAN_REF",
        "value_date": "VALUE_DATE",
        "amount": "AMOUNT_GBP",
        "channel": "CHANNEL",
        "mandate_ref": "MANDATE_REF",
    },
    investor={
        "loan_id": "Loan Identifier",
        "pool_id": "Pool",
        "period_end": "Period End",
        "reported_rate": "Reported Rate",
        "reported_margin": "Reported Margin",
        "scheduled_amount": "Scheduled P&I",
        "closing_balance": "Closing Principal Balance",
    },
    payments_date_format="%d/%m/%Y",
    extra_servicing=("Branch Code",),
    extra_payments=("NARRATIVE",),
)

VARIANT_B = HeaderVariant(
    name="B",
    servicing={
        "loan_id": "account_number",
        "pool_id": "funding_pool",
        "due_date": "next_due_date",
        "scheduled_amount": "contractual_payment",
        "closing_balance": "principal_outstanding",
        "base_rate": "reference_rate",
        "margin": "loan_margin",
        "applied_rate": "charged_rate",
        "mandate_ref": "mandate_id",
    },
    payments={
        "txn_id": "Reference",
        "loan_id": "Account",
        "value_date": "Received On",
        "amount": "Amount",
        "channel": "Method",
        "mandate_ref": "DDI Ref",
    },
    investor={
        "loan_id": "LoanID",
        "pool_id": "PoolCode",
        "period_end": "ReportingDate",
        "reported_rate": "InterestRate",
        "reported_margin": "Margin",
        "scheduled_amount": "ScheduledPayment",
        "closing_balance": "EndingBalance",
    },
    payments_date_format="%d-%m-%Y",
    extra_payments=("Narrative",),
)


@dataclass(frozen=True)
class PairSpec:
    name: str
    seed: int
    n_loans: int
    errors: tuple[str, ...]
    noise: tuple[str, ...]
    variant: HeaderVariant
    period: tuple[int, int] = (2026, 9)


_ALL_NOISE = (
    "loan_ref_formatting",
    "split_payment",
    "balance_rounding",
    "rate_precision",
    "value_date_shift",
)

PAIRS: dict[str, PairSpec] = {
    "pair_01": PairSpec(
        "pair_01",
        seed=101,
        n_loans=40,
        errors=("missing_payment", "duplicate_direct_debit", "rate_margin_breach"),
        noise=_ALL_NOISE,
        variant=VARIANT_A,
    ),
    "pair_02": PairSpec(
        "pair_02",
        seed=202,
        n_loans=60,
        errors=(
            "missing_payment",
            "duplicate_direct_debit",
            "rate_margin_breach",
            "partial_payment",
            "investor_balance_break",
        ),
        noise=_ALL_NOISE,
        variant=VARIANT_B,
    ),
    "pair_03": PairSpec(
        "pair_03",
        seed=303,
        n_loans=80,
        errors=tuple(ERROR_TYPES),
        noise=_ALL_NOISE,
        variant=VARIANT_A,
    ),
}

_CENT = Decimal("0.01")
_TERM_MONTHS = 300
_BASE_RATE = Decimal("4.00")
_MARGINS = [Decimal(m) for m in ("0.99", "1.25", "1.49", "1.75", "2.10")]
_DUE_DAYS = (1, 5, 10, 15, 20, 25, 28)


def _q(v: Decimal) -> Decimal:
    return v.quantize(_CENT, rounding=ROUND_HALF_UP)


def _annuity(balance: Decimal, annual_rate_pct: Decimal) -> tuple[Decimal, Decimal]:
    """Return (scheduled payment, closing balance) for one month of a repayment mortgage."""
    r = annual_rate_pct / Decimal(1200)
    payment = _q(balance * r / (1 - (1 + r) ** -_TERM_MONTHS))
    interest = _q(balance * r)
    return payment, balance - (payment - interest)


def _roll_weekend(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


@dataclass
class _Loan:
    loan_id: str
    pool_id: str
    opening: Decimal
    base_rate: Decimal
    margin: Decimal
    applied_rate: Decimal
    scheduled: Decimal
    closing: Decimal
    due_date: date
    channel: str
    mandate_ref: str | None
    branch: str
    inv_rate: Decimal = Decimal(0)
    inv_margin: Decimal = Decimal(0)
    inv_closing: Decimal = Decimal(0)
    inv_rate_text: str | None = None

    def __post_init__(self) -> None:
        self.inv_rate, self.inv_margin, self.inv_closing = (
            self.applied_rate,
            self.margin,
            self.closing,
        )


@dataclass
class _Payment:
    loan_ref: str
    value_date: date
    amount: Decimal
    channel: str
    mandate_ref: str | None
    narrative: str
    tag: str | None = None
    txn_id: str = ""


@dataclass
class _Catalog:
    errors: list[dict] = field(default_factory=list)
    noise: list[dict] = field(default_factory=list)


def _reprice(loan: _Loan) -> None:
    loan.scheduled, loan.closing = _annuity(loan.opening, loan.applied_rate)
    loan.inv_rate, loan.inv_margin, loan.inv_closing = loan.applied_rate, loan.margin, loan.closing


def _make_loans(spec: PairSpec, rng: random.Random) -> list[_Loan]:
    year, month = spec.period
    loans = []
    for i in range(spec.n_loans):
        opening = Decimal(rng.randrange(8_000_000, 45_000_000)) / 100
        margin = rng.choice(_MARGINS)
        applied = _BASE_RATE + margin
        scheduled, closing = _annuity(opening, applied)
        is_dd = rng.random() < 0.85
        loans.append(
            _Loan(
                loan_id=f"MC-{spec.seed:03d}{i:04d}",
                pool_id=rng.choice(("POOL-A", "POOL-B")),
                opening=opening,
                base_rate=_BASE_RATE,
                margin=margin,
                applied_rate=applied,
                scheduled=scheduled,
                closing=closing,
                due_date=date(year, month, rng.choice(_DUE_DAYS)),
                channel="DD" if is_dd else "FPS",
                mandate_ref=f"DDM{spec.seed:03d}{i:05d}" if is_dd else None,
                branch=f"BR{rng.randrange(10, 99)}",
            )
        )
    return loans


def _period_end(spec: PairSpec) -> date:
    year, month = spec.period
    first_next = date(year + month // 12, month % 12 + 1, 1)
    return first_next - timedelta(days=1)


def build_pair(spec: PairSpec) -> dict[str, str | bytes]:
    """Build every file for ``spec``; returns ``{file_name: content}``."""
    unknown = [e for e in spec.errors if e not in ERROR_TYPES] + [
        n for n in spec.noise if n not in NOISE_TYPES
    ]
    if unknown:
        raise ValueError(f"unknown error/noise types: {unknown}")

    rng = random.Random(spec.seed)
    loans = _make_loans(spec, rng)
    catalog = _Catalog()

    loan_errors = [e for e in spec.errors if e != "unidentified_payment"]
    picked = rng.sample(loans, k=len(loan_errors) + len(spec.noise))
    error_loan = dict(zip(loan_errors, picked[: len(loan_errors)], strict=True))
    noise_loan = dict(zip(spec.noise, picked[len(loan_errors) :], strict=True))
    special = {id(loan) for loan in picked}

    # --- mutations that change the servicing / investor view -------------------------------
    if "rate_margin_breach" in error_loan:
        loan = error_loan["rate_margin_breach"]
        loan.applied_rate = loan.base_rate + loan.margin + Decimal("0.50")
        _reprice(loan)  # borrower is charged (and pays) the wrong rate; investor sees it too
    if "duplicate_direct_debit" in error_loan:
        loan = error_loan["duplicate_direct_debit"]
        loan.channel = "DD"
        loan.mandate_ref = loan.mandate_ref or f"DDM{spec.seed:03d}{loan.loan_id[-4:]:0>5}"
    if "split_payment" in noise_loan:
        loan = noise_loan["split_payment"]
        loan.channel, loan.mandate_ref = "FPS", None
    if "investor_balance_break" in error_loan:
        error_loan["investor_balance_break"].inv_closing += Decimal("1250.00")
    if "balance_rounding" in noise_loan:
        noise_loan["balance_rounding"].inv_closing += _CENT
    if "rate_precision" in noise_loan:
        loan = noise_loan["rate_precision"]
        loan.inv_rate = loan.applied_rate + Decimal("0.004")
        loan.inv_rate_text = f"{loan.inv_rate:.4f}"

    # --- payments --------------------------------------------------------------------------
    payments: list[_Payment] = []
    for loan in loans:
        value_date = _roll_weekend(loan.due_date)
        if value_date != loan.due_date and id(loan) not in special:
            catalog.noise.append(
                {
                    "type": "weekend_roll",
                    "loan_id": loan.loan_id,
                    "description": f"{NOISE_TYPES['weekend_roll']} "
                    f"({loan.due_date.isoformat()} -> {value_date.isoformat()})",
                }
            )
        if loan is error_loan.get("missing_payment"):
            continue
        ref = loan.loan_id
        if loan is noise_loan.get("loan_ref_formatting"):
            ref = loan.loan_id.lower() + " "
        if loan is noise_loan.get("value_date_shift"):
            value_date = loan.due_date + timedelta(days=2)
        narrative = f"{loan.channel} COLLECTION {loan.loan_id}"

        if loan is noise_loan.get("split_payment"):
            first = _q(loan.scheduled * Decimal("0.6"))
            for part in (first, loan.scheduled - first):
                payments.append(_Payment(ref, value_date, part, "FPS", None, narrative))
            continue

        amount = loan.scheduled
        if loan is error_loan.get("partial_payment"):
            amount = _q(loan.scheduled * Decimal("0.6"))
        payments.append(
            _Payment(ref, value_date, amount, loan.channel, loan.mandate_ref, narrative)
        )
        if loan is error_loan.get("duplicate_direct_debit"):
            payments.append(
                _Payment(
                    ref,
                    value_date + timedelta(days=1),
                    amount,
                    "DD",
                    loan.mandate_ref,
                    narrative,
                    tag="duplicate",
                )
            )

    if "unidentified_payment" in spec.errors:
        payments.append(
            _Payment(
                f"MC-{spec.seed:03d}9999",
                date(*spec.period, 12),
                Decimal("875.40"),
                "FPS",
                None,
                "FPS PAYMENT REF UNKNOWN",
                tag="unidentified",
            )
        )

    payments.sort(key=lambda p: (p.value_date, p.loan_ref.strip().upper(), p.tag or ""))
    for n, p in enumerate(payments, start=1):
        p.txn_id = f"TX{spec.seed:03d}{n:06d}"

    # --- catalog ---------------------------------------------------------------------------
    for etype in spec.errors:
        kind, category, description = ERROR_TYPES[etype]
        if etype == "unidentified_payment":
            p = next(p for p in payments if p.tag == "unidentified")
            entry = {"loan_id": p.loan_ref, "txn_ids": [p.txn_id]}
        else:
            loan = error_loan[etype]
            entry = {"loan_id": loan.loan_id, "txn_ids": []}
            if etype == "duplicate_direct_debit":
                entry["txn_ids"] = [
                    p.txn_id
                    for p in payments
                    if p.tag == "duplicate" and p.loan_ref == loan.loan_id
                ]
        catalog.errors.append(
            {
                "type": etype,
                **entry,
                "expected_kind": kind.value,
                "expected_category": category.value,
                "description": description,
            }
        )
    for ntype in spec.noise:
        catalog.noise.append(
            {
                "type": ntype,
                "loan_id": noise_loan[ntype].loan_id,
                "description": NOISE_TYPES[ntype],
            }
        )
    catalog.errors.sort(key=lambda e: (e["loan_id"], e["type"]))
    catalog.noise.sort(key=lambda n: (n["loan_id"], n["type"]))

    return {
        "servicing_extract.csv": _servicing_csv(spec, loans),
        "payments.csv": _payments_csv(spec, payments),
        "investor_report.csv": _investor_csv(spec, loans),
        "expected_mapping.json": _json(_expected_mapping(spec.variant).model_dump(mode="json")),
        "seeded_errors.json": _json(
            {
                "generator_version": GENERATOR_VERSION,
                "pair": spec.name,
                "seed": spec.seed,
                "period": f"{spec.period[0]}-{spec.period[1]:02d}",
                "header_variant": spec.variant.name,
                "loans": spec.n_loans,
                "payments": len(payments),
                "errors": catalog.errors,
                "noise": catalog.noise,
            }
        ),
        "manual_process_notes.json": _json(
            _manual_notes(spec, loans, payments, error_loan, noise_loan, catalog)
        ),
    }


def _manual_notes(
    spec: PairSpec,
    loans: list[_Loan],
    payments: list[_Payment],
    error_loan: dict[str, _Loan],
    noise_loan: dict[str, _Loan],
    catalog: _Catalog,
) -> dict:
    """Simulated output of the *current manual process* for the Week 1 baseline.

    Mimics what hand-written exception notes look like: some seeded errors are missed, some
    figures are rounded ("about £1,400"), some notes cite no source row, and one benign item
    (a split payment) is flagged by mistake. Separate RNG, so the input files are unaffected.
    Replace with real anonymised analyst notes (same schema) when available.
    """
    rng = random.Random(spec.seed * 1009 + 17)
    srow = {loan.loan_id: i for i, loan in enumerate(loans, start=1)}

    def prow(ref: str) -> list[int]:
        return [i for i, p in enumerate(payments, start=1) if p.loan_ref.strip().upper() == ref]

    def approx(v: Decimal) -> str:
        return f"{round(v, -2):,.0f}"

    notes = []
    for err in catalog.errors:
        etype = err["type"]
        detected = rng.random() >= 0.15
        rounded = rng.random() < 0.35
        cites = rng.random() >= 0.25
        minutes = rng.randint(8, 40)
        if not detected:
            continue
        if etype == "unidentified_payment":
            p = next(p for p in payments if p.tag == "unidentified")
            ref, row = p.loan_ref, f"payments:{payments.index(p) + 1}"
            text = (
                f"About £{approx(p.amount)} received against {ref}, which is not on our book."
                if rounded
                else f"Payment {p.txn_id} of {p.amount} received against {ref}, not on our book."
            )
            remedy = "Move to suspense and trace the payer with the bank."
            notes.append((ref, "UNIDENTIFIED_PAYMENT", text, remedy, [row], minutes, cites))
            continue
        loan = error_loan[etype]
        lid, row = loan.loan_id, f"servicing:{srow[loan.loan_id]}"
        pays = [payments[i - 1] for i in prow(lid)]
        if etype == "missing_payment":
            cat, remedy = "MISSING_PAYMENT", "Chase the borrower and check the DD mandate."
            text = (
                f"No payment from {lid} this month, roughly £{approx(loan.scheduled)} outstanding."
                if rounded
                else f"No payment received for {lid}; {loan.scheduled} was due on "
                f"{loan.due_date.isoformat()}."
            )
        elif etype == "duplicate_direct_debit":
            cat, remedy = "DUPLICATE_DIRECT_DEBIT", "Refund the second collection to the borrower."
            text = (
                f"{lid} looks double debited, about £{approx(pays[0].amount)} each time."
                if rounded
                else f"{lid} was debited twice on mandate {loan.mandate_ref}: {pays[0].amount} on "
                f"{pays[0].value_date.isoformat()} and again on {pays[-1].value_date.isoformat()}."
            )
        elif etype == "rate_margin_breach":
            cat, remedy = "RATE_MARGIN_BREACH", "Refer to pricing to correct the rate and refund."
            contract = loan.base_rate + loan.margin
            text = (
                f"Rate on {lid} looks about 0.5% too high."
                if rounded
                else f"{lid} is charged {loan.applied_rate}% but base {loan.base_rate}% plus "
                f"margin {loan.margin}% is {contract}%."
            )
        elif etype == "partial_payment":
            cat, remedy = "PARTIAL_OR_OVERPAYMENT", "Contact the borrower about the shortfall."
            paid = pays[0].amount
            text = (
                f"{lid} underpaid by around £{approx(loan.scheduled - paid)}."
                if rounded
                else f"{lid} paid {paid} against {loan.scheduled} scheduled, short by "
                f"{loan.scheduled - paid}."
            )
        else:  # investor_balance_break
            cat, remedy = "INVESTOR_BALANCE_BREAK", "Ask investor reporting to restate the line."
            text = (
                f"Investor balance for {lid} is out by about £1.3k."
                if rounded
                else f"Investor report shows balance {loan.inv_closing:,.2f} for {lid} vs "
                f"servicing {loan.closing:,.2f}."
            )
        notes.append((lid, cat, text, remedy, [row], minutes, cites))

    if "split_payment" in noise_loan:  # a benign item flagged by mistake
        loan = noise_loan["split_payment"]
        a, b = (p.amount for p in payments if p.loan_ref.strip().upper() == loan.loan_id)
        notes.append(
            (
                loan.loan_id,
                "PARTIAL_OR_OVERPAYMENT",
                f"{loan.loan_id} paid in two parts ({a} and {b}); possibly a partial payment.",
                "Check with the borrower.",
                [f"servicing:{srow[loan.loan_id]}"],
                rng.randint(8, 40),
                True,
            )
        )

    return {
        "generator_version": GENERATOR_VERSION,
        "pair": spec.name,
        "simulated": True,
        "notes": [
            {
                "note_id": f"{spec.name}-MN{n:02d}",
                "loan_id": lid,
                "category": cat,
                "rationale": text,
                "remediation_note": remedy,
                "cited_evidence": rows if cites else [],
                "minutes_spent": minutes,
            }
            for n, (lid, cat, text, remedy, rows, minutes, cites) in enumerate(notes, start=1)
        ],
    }


def _expected_mapping(v: HeaderVariant) -> ColumnMapping:
    return ColumnMapping(
        servicing=SourceMapping(columns=dict(v.servicing)),
        payments=SourceMapping(columns=dict(v.payments), date_format=v.payments_date_format),
        investor=SourceMapping(columns=dict(v.investor)),
    )


def _csv(headers: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue()


def _json(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def _servicing_csv(spec: PairSpec, loans: list[_Loan]) -> str:
    h = spec.variant.servicing
    headers = [h[f] for f in h] + list(spec.variant.extra_servicing)
    rows = []
    for loan in loans:
        values = {
            "loan_id": loan.loan_id,
            "pool_id": loan.pool_id,
            "due_date": loan.due_date.isoformat(),
            "scheduled_amount": f"{loan.scheduled:.2f}",
            "closing_balance": f"{loan.closing:.2f}",
            "base_rate": f"{loan.base_rate:.2f}",
            "margin": f"{loan.margin:.2f}",
            "applied_rate": f"{loan.applied_rate:.2f}",
            "mandate_ref": loan.mandate_ref or "",
        }
        extras = {"Branch Code": loan.branch}
        rows.append([values[f] for f in h] + [extras[x] for x in spec.variant.extra_servicing])
    return _csv(headers, rows)


def _payments_csv(spec: PairSpec, payments: list[_Payment]) -> str:
    h = spec.variant.payments
    headers = [h[f] for f in h] + list(spec.variant.extra_payments)
    rows = []
    for p in payments:
        values = {
            "txn_id": p.txn_id,
            "loan_id": p.loan_ref,
            "value_date": p.value_date.strftime(spec.variant.payments_date_format),
            "amount": f"{p.amount:.2f}",
            "channel": p.channel,
            "mandate_ref": p.mandate_ref or "",
        }
        rows.append([values[f] for f in h] + [p.narrative for _ in spec.variant.extra_payments])
    return _csv(headers, rows)


def _investor_csv(spec: PairSpec, loans: list[_Loan]) -> str:
    h = spec.variant.investor
    period_end = _period_end(spec).isoformat()
    rows = []
    for loan in loans:
        values = {
            "loan_id": loan.loan_id,
            "pool_id": loan.pool_id,
            "period_end": period_end,
            "reported_rate": loan.inv_rate_text or f"{loan.inv_rate:.2f}",
            "reported_margin": f"{loan.inv_margin:.2f}",
            "scheduled_amount": f"{loan.scheduled:,.2f}",
            "closing_balance": f"{loan.inv_closing:,.2f}",
        }
        rows.append([values[f] for f in h])
    return _csv([h[f] for f in h], rows)


def write_pair(spec: PairSpec, out_dir: Path) -> Path:
    pair_dir = Path(out_dir) / spec.name
    pair_dir.mkdir(parents=True, exist_ok=True)
    for name, content in build_pair(spec).items():
        (pair_dir / name).write_bytes(
            content if isinstance(content, bytes) else content.encode("utf-8")
        )
    return pair_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic reconciliation pairs.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--pair", choices=sorted(PAIRS), action="append")
    parser.add_argument(
        "--no-recordings", action="store_true", help="skip writing oracle LLM recordings"
    )
    args = parser.parse_args(argv)
    from .oracle import write_oracle_recordings  # imports the engine; keep generator standalone

    for name in args.pair or sorted(PAIRS):
        pair_dir = write_pair(PAIRS[name], args.out)
        if not args.no_recordings:
            write_oracle_recordings(pair_dir)
        print(pair_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
