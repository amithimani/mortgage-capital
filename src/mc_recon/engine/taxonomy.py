"""Closed vocabularies shared by the engine and the LLM nodes.

Two layers, deliberately separate:

* :class:`DiscrepancyKind` — *structural* findings produced by deterministic code
  (what the numbers say).
* :class:`ExceptionCategory` — the *business* taxonomy an analyst works in. The
  ``classify_exceptions`` LLM node chooses a category, but only from the set allowed
  for the discrepancy's kind (:data:`ALLOWED_CATEGORIES`). Anything else is coerced to
  ``OTHER_NEEDS_REVIEW`` by :func:`constrain_category` — the model can never invent a
  label or relabel a duplicate as a missing payment.
"""

from __future__ import annotations

from enum import StrEnum


class DiscrepancyKind(StrEnum):
    UNMATCHED_EXPECTED_PAYMENT = "UNMATCHED_EXPECTED_PAYMENT"
    """A scheduled payment in the servicing extract has no payment in the match window."""

    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    """Two or more direct debits with the same mandate and amount in the match window."""

    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    """Net received (after removing duplicates) differs from the scheduled amount."""

    RATE_COMPONENT_MISMATCH = "RATE_COMPONENT_MISMATCH"
    """Applied rate != base + margin, or the investor report disagrees on rate/margin."""

    BALANCE_MISMATCH = "BALANCE_MISMATCH"
    """Investor-reported closing balance differs from the servicing closing balance."""

    UNMATCHED_PAYMENT = "UNMATCHED_PAYMENT"
    """A payment that cannot be tied to a scheduled payment (unknown loan or outside window)."""

    POPULATION_MISMATCH = "POPULATION_MISMATCH"
    """A loan is present in the servicing extract or investor report but not the other."""


class ExceptionCategory(StrEnum):
    MISSING_PAYMENT = "MISSING_PAYMENT"
    DUPLICATE_DIRECT_DEBIT = "DUPLICATE_DIRECT_DEBIT"
    RATE_MARGIN_BREACH = "RATE_MARGIN_BREACH"
    PARTIAL_OR_OVERPAYMENT = "PARTIAL_OR_OVERPAYMENT"
    INVESTOR_BALANCE_BREAK = "INVESTOR_BALANCE_BREAK"
    UNIDENTIFIED_PAYMENT = "UNIDENTIFIED_PAYMENT"
    INVESTOR_POPULATION_BREAK = "INVESTOR_POPULATION_BREAK"
    OTHER_NEEDS_REVIEW = "OTHER_NEEDS_REVIEW"


_OTHER = ExceptionCategory.OTHER_NEEDS_REVIEW

ALLOWED_CATEGORIES: dict[DiscrepancyKind, frozenset[ExceptionCategory]] = {
    DiscrepancyKind.UNMATCHED_EXPECTED_PAYMENT: frozenset(
        {ExceptionCategory.MISSING_PAYMENT, _OTHER}
    ),
    DiscrepancyKind.DUPLICATE_PAYMENT: frozenset(
        {ExceptionCategory.DUPLICATE_DIRECT_DEBIT, _OTHER}
    ),
    DiscrepancyKind.AMOUNT_MISMATCH: frozenset({ExceptionCategory.PARTIAL_OR_OVERPAYMENT, _OTHER}),
    DiscrepancyKind.RATE_COMPONENT_MISMATCH: frozenset(
        {ExceptionCategory.RATE_MARGIN_BREACH, _OTHER}
    ),
    DiscrepancyKind.BALANCE_MISMATCH: frozenset({ExceptionCategory.INVESTOR_BALANCE_BREAK, _OTHER}),
    DiscrepancyKind.UNMATCHED_PAYMENT: frozenset({ExceptionCategory.UNIDENTIFIED_PAYMENT, _OTHER}),
    DiscrepancyKind.POPULATION_MISMATCH: frozenset(
        {ExceptionCategory.INVESTOR_POPULATION_BREAK, _OTHER}
    ),
}


def allowed_categories(kind: DiscrepancyKind) -> list[ExceptionCategory]:
    """Allowed categories for ``kind``, in a stable order (for prompts and payloads)."""
    return sorted(ALLOWED_CATEGORIES[kind])


def constrain_category(kind: DiscrepancyKind, proposed: str) -> ExceptionCategory:
    """Deterministic guard on the classifier's output.

    Returns ``proposed`` if it is a valid category allowed for ``kind``; otherwise
    ``OTHER_NEEDS_REVIEW`` so the exception is routed to a human rather than mislabelled.
    """
    try:
        category = ExceptionCategory(proposed)
    except ValueError:
        return _OTHER
    return category if category in ALLOWED_CATEGORIES[kind] else _OTHER
