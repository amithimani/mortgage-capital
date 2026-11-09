"""Response contracts for the HTTP API. These generate the OpenAPI spec.

``ApiModel`` allows unknown fields at runtime, so a newly added field can't turn a working
response into a 500. ``tests/test_openapi.py`` runs real requests in every run state and fails on
any field a model doesn't declare, so drift is caught in CI instead.

Domain models (verdicts, evidence, mappings) are reused rather than copied, so the spec can't
disagree with what the agent actually produces.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .engine import ColumnMapping, DiscrepancyKind, ExceptionCategory, FileProfile
from .engine.models import EvidenceRef
from .governance import AISystem
from .llm import MappingProposal
from .policy import PolicyVerdict

RunStatus = Literal[
    "created",
    "ingested",
    "awaiting_mapping_confirmation",
    "classifying",
    "awaiting_review",
    "completed",
    "failed",
]
ExceptionStatus = Literal[
    "drafted", "edited", "passed", "blocked", "approved_override", "rejected", "logged"
]
ControlState = Literal["no_evidence", "stale", "breached", "met"]
ReviewAction = Literal["approve", "edit", "reject"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorBody(ApiModel):
    """Every error response has this shape."""

    error: str
    detail: Any = Field(default=None, description="Specifics, such as which fields were invalid")


# --- probes -------------------------------------------------------------------------------


class Health(ApiModel):
    status: Literal["ok"]
    model_id: str
    policy_id: str


class Readiness(ApiModel):
    ready: bool
    store: bool
    llm: bool
    gate: bool


# --- runs ---------------------------------------------------------------------------------


class MappingConfirmationAction(ApiModel):
    type: Literal["mapping_confirmation"]
    run_id: str
    proposal: MappingProposal
    problems: list[str]


class ExceptionReviewAction(ApiModel):
    type: Literal["exception_review"]
    run_id: str
    round: int
    max_rounds: int
    allowed_actions: list[ReviewAction]
    pending: list[str] = Field(description="Exception ids still held in this round")


PendingAction = Annotated[
    MappingConfirmationAction | ExceptionReviewAction, Field(discriminator="type")
]


class RunView(ApiModel):
    run_id: str
    dataset: str
    created_by: str
    created_at: str
    updated_at: str
    status: RunStatus
    error: str | None
    manifest: dict[str, Any] = Field(
        description="Audit record: file hashes, engine, prompt and policy versions, approver"
    )
    pending_action: PendingAction | None = Field(
        description="The human action the run is waiting for, if any"
    )
    exception_counts: dict[str, int]
    recon_stats: dict[str, int] | None


class ReviewResult(RunView):
    review_round: int
    awaiting: list[str] = Field(description="Exceptions still undecided in this round")


class MappingView(ApiModel):
    proposal: MappingProposal
    problems: list[str]
    confirmed_mapping: ColumnMapping | None
    confirmed_by: str | None
    profiles: list[FileProfile]
    input_screen: list[dict[str, Any]] = Field(description="Columns whose samples were withheld")
    input_screen_full: list[dict[str, Any]] = Field(
        description="Columns where the whole-file scan found a problem"
    )


# --- exceptions and the log ---------------------------------------------------------------


class ReviewRecord(ApiModel):
    round: int
    analyst: str
    action: ReviewAction
    rationale: str | None = None
    remediation_note: str | None = None
    note: str | None = None


class _Finding(ApiModel):
    loan_id: str | None
    kind: DiscrepancyKind
    category: ExceptionCategory
    owner: str
    priority: Literal["P1", "P2", "P3"]
    sla_business_days: int
    amount_at_risk: str | None = Field(description="Decimal, as a string")
    amount_basis: str | None
    priority_reason: str
    summary: str
    rationale: str
    remediation_note: str
    evidence: list[EvidenceRef]
    cited_evidence: list[str]
    reviews: list[ReviewRecord]
    degraded: str | None = Field(
        default=None, description="Why automation was skipped for this finding, if it was"
    )


class ExceptionView(_Finding):
    id: str
    expected: dict[str, str]
    actual: dict[str, str]
    proposed_category: ExceptionCategory
    classification_reason: str
    status: ExceptionStatus
    verdicts: list[PolicyVerdict] = Field(description="Every policy evaluation, oldest first")


class RemediationEntry(_Finding):
    run_id: str
    exception_id: str
    final_verdict: PolicyVerdict
    human_override: bool
    logged_at: str


class EventView(ApiModel):
    seq: int
    run_id: str
    event_type: str
    actor: str
    subject_id: str | None
    payload: dict[str, Any]
    dedupe_key: str | None
    at: str


class TrajectoryStep(BaseModel):
    """One step the agent took. Other fields depend on the step."""

    model_config = ConfigDict(extra="allow")

    step: str
    tool: str | None = None


# --- analytics ----------------------------------------------------------------------------


class RunMetrics(ApiModel):
    run_id: str
    status: RunStatus
    dataset: str
    model_id: str | None
    latency: dict[str, Any]
    llm: dict[str, Any]
    policy: dict[str, Any]
    exceptions: dict[str, Any]
    vs_baseline: dict[str, Any] | None


class MetricsSummary(ApiModel):
    runs: int
    by_status: dict[str, int]
    latency: dict[str, Any]
    llm: dict[str, Any]
    policy: dict[str, Any]
    vs_baseline: dict[str, Any] | None


class Baseline(ApiModel):
    baseline_id: str
    label: str
    created_by: str
    created_at: str
    summary: dict[str, Any]
    results: list[dict[str, Any]]


# --- harness and memory -------------------------------------------------------------------


class RunUsage(ApiModel):
    llm_calls: int
    total_tokens: int
    cost_usd: float
    llm_failures: int
    consecutive_llm_failures: int
    llm_seconds: float


class HarnessLimits(ApiModel):
    max_reasks: int
    max_review_rounds: int
    max_llm_calls: int
    max_total_tokens: int
    max_cost_usd: float
    max_llm_seconds: float
    max_input_tokens_per_call: int
    max_consecutive_llm_failures: int
    max_findings: int
    max_file_bytes: int
    max_rows: int
    recursion_limit: int


class HarnessView(ApiModel):
    usage: RunUsage
    review_rounds: int
    limits: HarnessLimits
    limit_events: list[EventView]
    degraded: list[str] = Field(description="Exception ids held because automation was skipped")
    state_bytes: int


class MappingMemoryEntry(ApiModel):
    fingerprint: str
    mapping: ColumnMapping
    confirmed_by: str
    source_run: str
    confirmed_at: str


class ForgetResult(ApiModel):
    forgotten: str


# --- governance ---------------------------------------------------------------------------


class OpenBreach(ApiModel):
    run_id: str
    subject_id: str
    detail: str
    evaluated_at: str


class ControlStatus(ApiModel):
    control_id: str
    title: str
    requirement: str
    severity: str
    test_type: str
    evidence_source: str
    metric: str
    threshold: float
    freshness_hours: int
    status: ControlState
    evidence_records: int
    current_subjects: int
    open_breaches: list[OpenBreach]
    breaches_recorded: int
    newest_evidence_at: str | None
    evidence_expires_at: str | None


class RequirementCoverage(ApiModel):
    requirement: str
    control_id: str
    status: ControlState
    covered: bool


class Coverage(ApiModel):
    requirements: list[RequirementCoverage]
    covered: int
    total: int
    gaps: list[RequirementCoverage]


class EvidenceRecord(ApiModel):
    evidence_id: str
    application_id: str
    control_id: str
    run_id: str
    subject_id: str
    metric: str
    value: float | None
    threshold: float
    outcome: Literal["pass", "breach", "resolved"]
    detail: str
    policy_version: str | None
    source: str
    evaluated_at: str
    expires_at: str


class GovernanceReport(ApiModel):
    title: str
    generated_at: str
    system: AISystem
    controls: list[ControlStatus]
    coverage: Coverage
    scope_note: str
