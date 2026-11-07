from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from mc_recon.engine.models import (
    InvestorRecord,
    LoadedInputs,
    PaymentRecord,
    ServicingRecord,
)
from mc_recon.engine.schema import Source

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

D = Decimal


def servicing(loan_id="MC-1", row=1, **kw) -> ServicingRecord:
    base = dict(
        row=row,
        loan_id=loan_id,
        pool_id="POOL-A",
        due_date=date(2026, 9, 15),
        scheduled_amount=D("1000.00"),
        closing_balance=D("150000.00"),
        base_rate=D("4.00"),
        margin=D("1.25"),
        applied_rate=D("5.25"),
        mandate_ref="DDM1",
    )
    base.update(kw)
    return ServicingRecord(**base)


def payment(txn_id="T1", loan_id="MC-1", row=1, **kw) -> PaymentRecord:
    base = dict(
        row=row,
        txn_id=txn_id,
        loan_id=loan_id,
        value_date=date(2026, 9, 15),
        amount=D("1000.00"),
        channel="DD",
        mandate_ref="DDM1",
    )
    base.update(kw)
    return PaymentRecord(**base)


def investor(loan_id="MC-1", row=1, **kw) -> InvestorRecord:
    base = dict(
        row=row,
        loan_id=loan_id,
        pool_id="POOL-A",
        period_end=date(2026, 9, 30),
        reported_rate=D("5.25"),
        reported_margin=D("1.25"),
        scheduled_amount=D("1000.00"),
        closing_balance=D("150000.00"),
    )
    base.update(kw)
    return InvestorRecord(**base)


def inputs(servicing_rows=(), payment_rows=(), investor_rows=()) -> LoadedInputs:
    return LoadedInputs(
        servicing=tuple(servicing_rows),
        payments=tuple(payment_rows),
        investor=tuple(investor_rows),
        file_hashes={s: "0" * 64 for s in Source},
    )


@pytest.fixture(params=sorted(p.name for p in SYNTHETIC_DIR.glob("pair_*")))
def pair_dir(request) -> Path:
    return SYNTHETIC_DIR / request.param
