"""Golden regression: on every committed synthetic pair the engine must find exactly the seeded
errors — no misses, and no false positives from the benign noise."""

from __future__ import annotations

from collections import Counter

from mc_recon.data.synthetic import load_catalog, load_expected_mapping, pair_paths
from mc_recon.engine import load_inputs, reconcile


def test_engine_finds_exactly_the_seeded_errors(pair_dir):
    result = reconcile(load_inputs(pair_paths(pair_dir), load_expected_mapping(pair_dir)))
    catalog = load_catalog(pair_dir)

    got = Counter((d.kind.value, d.loan_id) for d in result.discrepancies)
    want = Counter((e["expected_kind"], e["loan_id"]) for e in catalog["errors"])
    assert got == want, f"extra={dict(got - want)} missing={dict(want - got)}"


def test_noise_loans_raise_nothing(pair_dir):
    result = reconcile(load_inputs(pair_paths(pair_dir), load_expected_mapping(pair_dir)))
    noisy = {n["loan_id"] for n in load_catalog(pair_dir)["noise"]}
    flagged = {d.loan_id for d in result.discrepancies}
    assert noisy, "pair should carry benign noise"
    assert not (noisy & flagged)


def test_seeded_txn_ids_are_in_evidence(pair_dir):
    result = reconcile(load_inputs(pair_paths(pair_dir), load_expected_mapping(pair_dir)))
    evidence_keys = {e.key for d in result.discrepancies for e in d.evidence}
    for err in load_catalog(pair_dir)["errors"]:
        assert set(err["txn_ids"]) <= evidence_keys, err


def test_required_error_types_are_covered():
    """Plan §8: at least one missing payment, duplicate DD and rate-vs-margin breach."""
    from conftest import SYNTHETIC_DIR

    types = {e["type"] for p in SYNTHETIC_DIR.glob("pair_*") for e in load_catalog(p)["errors"]}
    assert {"missing_payment", "duplicate_direct_debit", "rate_margin_breach"} <= types
