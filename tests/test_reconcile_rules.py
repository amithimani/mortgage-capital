"""Unit tests for each reconciliation rule, including the benign cases that must stay quiet."""

from __future__ import annotations

from datetime import date
from decimal import Decimal as D

from conftest import inputs, investor, payment, servicing

from mc_recon.engine import DiscrepancyKind as K
from mc_recon.engine import ReconConfig, reconcile


def kinds(result):
    return [(d.kind, d.loan_id) for d in result.discrepancies]


def test_clean_loan_raises_nothing():
    r = reconcile(inputs([servicing()], [payment()], [investor()]))
    assert r.discrepancies == ()
    assert r.stats["discrepancies"] == 0


def test_missing_payment():
    r = reconcile(inputs([servicing()], [], [investor()]))
    assert kinds(r) == [(K.UNMATCHED_EXPECTED_PAYMENT, "MC-1")]
    assert r.discrepancies[0].expected == {"scheduled_amount": "1000.00"}


def test_payment_outside_window_is_unmatched_and_expected_is_missing():
    r = reconcile(inputs([servicing()], [payment(value_date=date(2026, 9, 25))], [investor()]))
    assert sorted(kinds(r)) == sorted(
        [(K.UNMATCHED_EXPECTED_PAYMENT, "MC-1"), (K.UNMATCHED_PAYMENT, "MC-1")]
    )


def test_payment_at_window_edge_matches():
    cfg = ReconConfig(match_window_days=3)
    r = reconcile(inputs([servicing()], [payment(value_date=date(2026, 9, 18))], [investor()]), cfg)
    assert r.discrepancies == ()


def test_duplicate_direct_debit_flagged_once_and_excluded_from_netting():
    pays = [payment("T1", row=1), payment("T2", row=2, value_date=date(2026, 9, 16))]
    r = reconcile(inputs([servicing()], pays, [investor()]))
    assert kinds(r) == [(K.DUPLICATE_PAYMENT, "MC-1")]
    d = r.discrepancies[0]
    assert d.actual["collections"] == "2"
    assert [e.key for e in d.evidence] == ["MC-1", "T1", "T2"]


def test_same_amount_fps_payments_are_not_duplicates():
    pays = [
        payment("T1", row=1, channel="FPS", mandate_ref=None, amount=D("500.00")),
        payment("T2", row=2, channel="FPS", mandate_ref=None, amount=D("500.00")),
    ]
    r = reconcile(inputs([servicing()], pays, [investor()]))
    assert r.discrepancies == ()


def test_split_payment_netting_to_schedule_is_benign():
    pays = [
        payment("T1", row=1, channel="FPS", mandate_ref=None, amount=D("600.00")),
        payment("T2", row=2, channel="FPS", mandate_ref=None, amount=D("400.00")),
    ]
    r = reconcile(inputs([servicing()], pays, [investor()]))
    assert r.discrepancies == ()


def test_partial_payment_is_amount_mismatch():
    r = reconcile(inputs([servicing()], [payment(amount=D("600.00"))], [investor()]))
    assert kinds(r) == [(K.AMOUNT_MISMATCH, "MC-1")]
    assert r.discrepancies[0].actual["difference"] == "-400.00"


def test_amount_within_tolerance_is_benign():
    r = reconcile(inputs([servicing()], [payment(amount=D("1000.01"))], [investor()]))
    assert r.discrepancies == ()


def test_rate_margin_breach_servicing_internal():
    s = servicing(applied_rate=D("5.75"))
    r = reconcile(inputs([s], [payment()], [investor(reported_rate=D("5.75"))]))
    assert kinds(r) == [(K.RATE_COMPONENT_MISMATCH, "MC-1")]
    assert r.discrepancies[0].expected == {"applied_rate": "5.25"}


def test_investor_margin_disagrees_with_servicing():
    r = reconcile(inputs([servicing()], [payment()], [investor(reported_margin=D("1.49"))]))
    assert kinds(r) == [(K.RATE_COMPONENT_MISMATCH, "MC-1")]


def test_rate_precision_within_tolerance_is_benign():
    r = reconcile(inputs([servicing()], [payment()], [investor(reported_rate=D("5.2540"))]))
    assert r.discrepancies == ()


def test_balance_break_and_rounding():
    broken = reconcile(
        inputs([servicing()], [payment()], [investor(closing_balance=D("151250.00"))])
    )
    assert kinds(broken) == [(K.BALANCE_MISMATCH, "MC-1")]
    rounded = reconcile(
        inputs([servicing()], [payment()], [investor(closing_balance=D("150000.01"))])
    )
    assert rounded.discrepancies == ()


def test_payment_for_unknown_loan():
    r = reconcile(inputs([servicing()], [payment(), payment("T9", "MC-404", row=2)], [investor()]))
    assert kinds(r) == [(K.UNMATCHED_PAYMENT, "MC-404")]


def test_population_mismatch_both_directions():
    r = reconcile(
        inputs(
            [servicing("MC-1"), servicing("MC-2", row=2)],
            [payment(), payment("T2", "MC-2", row=2)],
            [investor("MC-1"), investor("MC-3", row=2)],
        )
    )
    assert sorted(kinds(r)) == [
        (K.POPULATION_MISMATCH, "MC-2"),
        (K.POPULATION_MISMATCH, "MC-3"),
    ]


def test_output_is_deterministic_and_ids_are_stable():
    args = ([servicing()], [payment(amount=D("10.00"))], [investor(reported_margin=D("2.00"))])
    a, b = reconcile(inputs(*args)), reconcile(inputs(*args))
    assert a.model_dump_json() == b.model_dump_json()
    assert all(d.id.startswith("D-") and len(d.id) == 14 for d in a.discrepancies)
    assert len({d.id for d in a.discrepancies}) == len(a.discrepancies)
