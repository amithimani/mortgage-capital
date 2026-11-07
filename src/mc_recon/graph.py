"""The reconciliation agent as a LangGraph state machine.

    ingest → propose_mapping → await_mapping_confirm (interrupt) → reconcile
      → classify_exceptions → draft_rationale → validate_policy
      → [all PASS] emit_log
      → [any BLOCK] human_review_queue (interrupt)
            → [any edited] validate_policy  (re-check, plan D1)
            → [else]       emit_log

State holds only JSON-native values (dicts/lists/str), so checkpoints are portable and never
depend on pickling our classes. Dependencies (LLM, policy gate, store, tracer) come in through
:class:`AgentDeps`; nothing in here knows whether they are fakes.

Interrupt payloads (what the API shows a human) and resume values (what it sends back):

* ``mapping_confirmation`` → ``{"analyst": str, "mapping": ColumnMapping-json | None}``
* ``exception_review`` → ``{exception_id: {"analyst", "action": approve|edit|reject,
  "rationale"?, "remediation_note"?, "note"?}}`` covering every pending id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .engine import (
    ENGINE_VERSION,
    ColumnMapping,
    Discrepancy,
    DiscrepancyKind,
    EngineInputError,
    FileProfile,
    MappingError,
    ReconConfig,
    Source,
    constrain_category,
    load_inputs,
    profile_csv,
    reconcile,
)
from .llm import (
    PROMPT_VERSIONS,
    ClassificationOutput,
    LLMOutputValidationError,
    LLMProvider,
    LLMRequest,
    MappingProposal,
    RationaleOutput,
    classification_request,
    mapping_request,
    rationale_request,
)
from .llm.provider import T, canonical_json
from .observability import NullTracer, Tracer
from .policy import PolicyGate, PolicySubject, evaluate_fail_closed, screen_fail_closed
from .store import RunStore

SYSTEM_ACTOR = "agent"
WITHHELD = "[withheld by input screen]"

# exception lifecycle
DRAFTED, EDITED = "drafted", "edited"
PASSED, BLOCKED = "passed", "blocked"
APPROVED_OVERRIDE, REJECTED, LOGGED = "approved_override", "rejected", "logged"
NEEDS_VALIDATION = frozenset({DRAFTED, EDITED})
LOGGABLE = frozenset({PASSED, APPROVED_OVERRIDE})


class InputScreenError(EngineInputError):
    """A column *header* failed the input screen — headers can't be withheld, so stop."""


class RunState(TypedDict, total=False):
    run_id: str
    dataset: str
    input_paths: dict[str, str]
    file_hashes: dict[str, str]
    profiles: list[dict]
    input_screen: list[dict]
    mapping_proposal: dict
    mapping_problems: list[str]
    mapping: dict
    mapping_confirmed_by: str
    recon_stats: dict
    discrepancies: list[dict]
    exceptions: dict[str, dict]
    review_round: int
    status: str


@dataclass
class AgentDeps:
    llm: LLMProvider
    gate: PolicyGate
    store: RunStore
    tracer: Tracer = field(default_factory=NullTracer)
    recon_config: ReconConfig = field(default_factory=ReconConfig)
    max_reasks: int = 1


# --- helpers ------------------------------------------------------------------------------


def _paths(state: RunState) -> dict[Source, Path]:
    return {Source(k): Path(v) for k, v in state["input_paths"].items()}


def _headers(state: RunState) -> dict[Source, list[str]]:
    return {Source(p["source"]): [c["name"] for c in p["columns"]] for p in state["profiles"]}


def evidence_keys(exc: dict) -> list[str]:
    return [f"{e['source']}:{e['row']}" for e in exc["evidence"]]


def policy_subject(exc: dict) -> PolicySubject:
    context = canonical_json(
        {
            "summary": exc["summary"],
            "expected": exc["expected"],
            "actual": exc["actual"],
            "evidence": {
                f"{e['source']}:{e['row']}": {"key": e["key"], **e["fields"]}
                for e in exc["evidence"]
            },
        }
    )
    return PolicySubject(
        exception_id=exc["id"],
        loan_id=exc["loan_id"],
        prompt=(
            f"Explain reconciliation exception {exc['id']} ({exc['category']}) and propose "
            "remediation using only the supplied evidence."
        ),
        context=context,
        response=f"{exc['rationale']}\n\nRemediation: {exc['remediation_note']}",
        evidence_keys=evidence_keys(exc),
        cited_evidence=exc["cited_evidence"],
    )


def _discrepancy(exc: dict) -> Discrepancy:
    return Discrepancy.model_validate(
        {k: exc[k] for k in ("id", "kind", "loan_id", "summary", "expected", "actual", "evidence")}
    )


def build_graph(deps: AgentDeps) -> StateGraph:
    store, tracer, llm, gate = deps.store, deps.tracer, deps.llm, deps.gate

    def call_llm(run_id: str, node: str, request: LLMRequest, schema: type[T]) -> T:
        """One structured call with a bounded re-ask on schema-invalid output."""
        attempts = deps.max_reasks + 1
        for attempt in range(1, attempts + 1):
            with tracer.span(
                run_id,
                node,
                "MODEL_EXEC",
                task=request.task.value,
                prompt_version=request.prompt_version,
                cache_key=request.cache_key,
                model_id=llm.model_id,
                attempt=attempt,
            ) as span:
                try:
                    result = llm.generate(request, schema)
                except LLMOutputValidationError:
                    span.set(outcome="invalid_output")
                    if attempt == attempts:
                        raise
                    continue
                span.set(
                    outcome="ok",
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                )
                return result.output
        raise AssertionError("unreachable")

    # --- nodes ----------------------------------------------------------------------------

    def ingest(state: RunState) -> RunState:
        run_id = state["run_id"]
        with tracer.span(run_id, "ingest", "TOOL_EXEC") as span:
            profiles, findings = [], []
            for source, path in _paths(state).items():
                profile = profile_csv(path, source).model_dump(mode="json")
                for col in profile["columns"]:
                    header_verdict = screen_fail_closed(gate, col["name"])
                    if header_verdict.blocked:
                        raise InputScreenError(
                            f"{source}: header {col['name']!r} failed the input screen"
                        )
                    verdict = screen_fail_closed(gate, "\n".join(col["samples"]))
                    if verdict.blocked:
                        col["samples"] = [WITHHELD] * len(col["samples"])
                        findings.append(
                            {
                                "source": source.value,
                                "column": col["name"],
                                "verdict": verdict.model_dump(mode="json"),
                            }
                        )
                profiles.append(profile)
            hashes = {p["source"]: p["sha256"] for p in profiles}
            for f in findings:
                store.record_event(
                    run_id,
                    "input_screen_block",
                    SYSTEM_ACTOR,
                    f,
                    subject_id=f"{f['source']}.{f['column']}",
                    dedupe_key=f"{run_id}:screen:{f['source']}:{f['column']}",
                )
            store.update_run(run_id, status="ingested", manifest={"file_hashes": hashes})
            span.set(files=len(profiles), screen_blocks=len(findings))
        return {"profiles": profiles, "file_hashes": hashes, "input_screen": findings}

    def propose_mapping(state: RunState) -> RunState:
        profiles = [FileProfile.model_validate(p) for p in state["profiles"]]
        proposal = call_llm(
            state["run_id"], "propose_mapping", mapping_request(profiles), MappingProposal
        )
        problems = proposal.mapping.problems(_headers(state))
        return {
            "mapping_proposal": proposal.model_dump(mode="json"),
            "mapping_problems": problems,
            "status": "awaiting_mapping_confirmation",
        }

    def await_mapping_confirm(state: RunState) -> RunState:
        run_id = state["run_id"]
        answer = interrupt(
            {
                "type": "mapping_confirmation",
                "run_id": run_id,
                "proposal": state["mapping_proposal"],
                "problems": state["mapping_problems"],
            }
        )
        analyst = answer["analyst"]
        edited = answer.get("mapping")
        mapping = ColumnMapping.model_validate(edited or state["mapping_proposal"]["mapping"])
        mapping.validate_against(_headers(state))  # API pre-validates; this is the backstop
        with tracer.span(run_id, "await_mapping_confirm", "OVERSIGHT", analyst=analyst):
            store.record_event(
                run_id,
                "mapping_edited" if edited else "mapping_confirmed",
                analyst,
                {
                    "mapping": mapping.model_dump(mode="json"),
                    "proposal_problems": state["mapping_problems"],
                },
                dedupe_key=f"{run_id}:mapping",
            )
        return {
            "mapping": mapping.model_dump(mode="json"),
            "mapping_confirmed_by": analyst,
            "status": "reconciling",
        }

    def reconcile_node(state: RunState) -> RunState:
        run_id = state["run_id"]
        with tracer.span(run_id, "reconcile", "TOOL_EXEC", engine_version=ENGINE_VERSION) as span:
            inputs = load_inputs(_paths(state), ColumnMapping.model_validate(state["mapping"]))
            hashes = {s.value: h for s, h in inputs.file_hashes.items()}
            if hashes != state["file_hashes"]:
                raise EngineInputError("input files changed between ingest and reconcile")
            result = reconcile(inputs, deps.recon_config)
            span.set(discrepancies=len(result.discrepancies))
        store.update_run(
            run_id,
            status="classifying",
            manifest={
                "engine_version": result.engine_version,
                "recon_config": result.config.model_dump(mode="json"),
                "mapping_confirmed_by": state["mapping_confirmed_by"],
                "recon_stats": result.stats,
            },
        )
        return {
            "discrepancies": [d.model_dump(mode="json") for d in result.discrepancies],
            "recon_stats": result.stats,
        }

    def classify_exceptions(state: RunState) -> RunState:
        exceptions: dict[str, dict] = {}
        for raw in state["discrepancies"]:
            d = Discrepancy.model_validate(raw)
            out = call_llm(
                state["run_id"],
                "classify_exception",
                classification_request(d),
                ClassificationOutput,
            )
            category = constrain_category(d.kind, out.category.value)
            exceptions[d.id] = {
                **raw,
                "proposed_category": out.category.value,
                "category": category.value,
                "classification_reason": out.reason,
                "status": DRAFTED,
                "verdicts": [],
                "reviews": [],
            }
        return {"exceptions": exceptions}

    def draft_rationale(state: RunState) -> RunState:
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        for exc_id in sorted(exceptions):
            exc = exceptions[exc_id]
            out = call_llm(
                state["run_id"],
                "draft_rationale",
                rationale_request(_discrepancy(exc), exc["category"]),
                RationaleOutput,
            )
            exc.update(
                rationale=out.rationale,
                remediation_note=out.remediation_note,
                cited_evidence=out.cited_evidence,
            )
        return {"exceptions": exceptions, "status": "validating"}

    def validate_policy(state: RunState) -> RunState:
        run_id = state["run_id"]
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        for exc_id in sorted(exceptions):
            exc = exceptions[exc_id]
            if exc["status"] not in NEEDS_VALIDATION:
                continue
            with tracer.span(run_id, "validate_policy", "TOOL_EXEC", exception_id=exc_id) as span:
                verdict = evaluate_fail_closed(gate, policy_subject(exc))
                span.set(decision=verdict.decision, policy_version=verdict.policy_version)
            attempt = len(exc["verdicts"]) + 1
            exc["verdicts"] = [*exc["verdicts"], verdict.model_dump(mode="json")]
            exc["status"] = BLOCKED if verdict.blocked else PASSED
            store.record_event(
                run_id,
                "policy_verdict",
                SYSTEM_ACTOR,
                {"attempt": attempt, **verdict.model_dump(mode="json")},
                subject_id=exc_id,
                dedupe_key=f"{run_id}:verdict:{exc_id}:{attempt}",
            )
        policy_ids = sorted({v["policy_id"] for e in exceptions.values() for v in e["verdicts"]})
        versions = sorted(
            {
                f"{v['policy_id']}@{v['policy_version']}"
                for e in exceptions.values()
                for v in e["verdicts"]
                if v["policy_version"] is not None
            }
        )
        store.update_run(run_id, manifest={"policy_ids": policy_ids, "policy_versions": versions})
        return {"exceptions": exceptions}

    def human_review_queue(state: RunState) -> RunState:
        run_id = state["run_id"]
        round_ = state.get("review_round", 0) + 1
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        pending = sorted(k for k, v in exceptions.items() if v["status"] == BLOCKED)
        decisions: dict[str, dict] = interrupt(
            {
                "type": "exception_review",
                "run_id": run_id,
                "round": round_,
                "pending": pending,
            }
        )
        missing = set(pending) - set(decisions)
        if missing:
            raise ValueError(f"review round {round_} is missing decisions for {sorted(missing)}")
        for exc_id in pending:
            exc, dec = exceptions[exc_id], decisions[exc_id]
            action, analyst = dec["action"], dec["analyst"]
            with tracer.span(
                run_id, "human_review", "OVERSIGHT", exception_id=exc_id, action=action
            ):
                if action == "approve":
                    exc["status"] = APPROVED_OVERRIDE
                elif action == "reject":
                    exc["status"] = REJECTED
                elif action == "edit":
                    exc["rationale"] = dec.get("rationale") or exc["rationale"]
                    exc["remediation_note"] = dec.get("remediation_note") or exc["remediation_note"]
                    exc["status"] = EDITED
                else:
                    raise ValueError(f"unknown review action {action!r}")
                exc["reviews"] = [*exc["reviews"], {"round": round_, **dec}]
                store.record_event(
                    run_id,
                    "exception_reviewed",
                    analyst,
                    {"round": round_, **dec},
                    subject_id=exc_id,
                    dedupe_key=f"{run_id}:review:{exc_id}:{round_}",
                )
        return {"exceptions": exceptions, "review_round": round_}

    def emit_log(state: RunState) -> RunState:
        run_id = state["run_id"]
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        written = 0
        with tracer.span(run_id, "emit_log", "TOOL_EXEC") as span:
            for exc_id in sorted(exceptions):
                exc = exceptions[exc_id]
                if exc["status"] not in LOGGABLE:
                    continue
                final = exc["verdicts"][-1]
                entry = {
                    "run_id": run_id,
                    "exception_id": exc_id,
                    "loan_id": exc["loan_id"],
                    "kind": exc["kind"],
                    "category": exc["category"],
                    "summary": exc["summary"],
                    "rationale": exc["rationale"],
                    "remediation_note": exc["remediation_note"],
                    "evidence": exc["evidence"],
                    "cited_evidence": exc["cited_evidence"],
                    "final_verdict": final,
                    "human_override": exc["status"] == APPROVED_OVERRIDE,
                    "reviews": exc["reviews"],
                }
                written += store.append_log(run_id, exc_id, entry)
                exc["status"] = LOGGED
            span.set(written=written)
        store.update_run(run_id, status="completed")
        return {"exceptions": exceptions, "status": "completed"}

    # --- routing --------------------------------------------------------------------------

    def after_validate(state: RunState) -> str:
        blocked = any(e["status"] == BLOCKED for e in state["exceptions"].values())
        return "human_review_queue" if blocked else "emit_log"

    def after_review(state: RunState) -> str:
        edited = any(e["status"] == EDITED for e in state["exceptions"].values())
        return "validate_policy" if edited else "emit_log"

    g = StateGraph(RunState)
    g.add_node("ingest", ingest)
    g.add_node("propose_mapping", propose_mapping)
    g.add_node("await_mapping_confirm", await_mapping_confirm)
    g.add_node("reconcile", reconcile_node)
    g.add_node("classify_exceptions", classify_exceptions)
    g.add_node("draft_rationale", draft_rationale)
    g.add_node("validate_policy", validate_policy)
    g.add_node("human_review_queue", human_review_queue)
    g.add_node("emit_log", emit_log)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "propose_mapping")
    g.add_edge("propose_mapping", "await_mapping_confirm")
    g.add_edge("await_mapping_confirm", "reconcile")
    g.add_edge("reconcile", "classify_exceptions")
    g.add_edge("classify_exceptions", "draft_rationale")
    g.add_edge("draft_rationale", "validate_policy")
    g.add_conditional_edges("validate_policy", after_validate, ["human_review_queue", "emit_log"])
    g.add_conditional_edges("human_review_queue", after_review, ["validate_policy", "emit_log"])
    g.add_edge("emit_log", END)
    return g


def manifest_base(dataset: str, llm: LLMProvider) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "engine_version": ENGINE_VERSION,
        "prompt_versions": {t.value: v for t, v in PROMPT_VERSIONS.items()},
        "model_id": llm.model_id,
    }


__all__ = [
    "AgentDeps",
    "DiscrepancyKind",
    "InputScreenError",
    "MappingError",
    "RunState",
    "build_graph",
    "evidence_keys",
    "manifest_base",
    "policy_subject",
]
