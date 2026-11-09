"""Run lifecycle on top of the graph: start, inspect, confirm mapping, review, read the log.

The API layer is a thin HTTP shell over :class:`RunService`. Everything that decides *whether*
a human action is allowed lives here, before the graph is resumed, so invalid input is a 4xx
and never a half-applied checkpoint.

Graph calls are synchronous. That is fine with the fake LLM; with a live model, ``start_run``
and the resume calls move to a background worker and the API returns ``202``.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, model_validator

from . import metrics as metrics_mod
from .baseline import has_notes, run_baseline
from .data.synthetic import FILE_NAMES
from .engine import ColumnMapping, Source
from .graph import AgentDeps, build_graph, manifest_base
from .limits import RunBudget
from .llm.provider import canonical_json
from .memory import prune_checkpoints
from .observability import StoreExporter

_DATASET = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ServiceError(Exception):
    status_code = 400

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.message, self.detail = message, detail


class Unauthorized(ServiceError):
    status_code = 401


class NotFound(ServiceError):
    status_code = 404


class Conflict(ServiceError):
    status_code = 409


class Invalid(ServiceError):
    status_code = 422


class TooLarge(ServiceError):
    status_code = 413


class RunFailed(ServiceError):
    """The agent itself failed: a server-side problem, so retries and alerts read it correctly."""

    status_code = 500


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "reject"]
    rationale: str | None = None
    remediation_note: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _edit_has_content(self) -> ReviewDecision:
        if self.action == "edit" and not (self.rationale or self.remediation_note):
            raise ValueError("an edit must supply rationale and/or remediation_note")
        if self.action == "approve" and not (self.note and self.note.strip()):
            raise ValueError("approving a blocked exception is an override and needs a note")
        return self


def sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    path = str(path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class RunService:
    def __init__(
        self,
        deps: AgentDeps,
        data_root: Path,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.deps = deps
        self.store = deps.store
        self.data_root = Path(data_root).resolve()
        deps.tracer.add_exporter(StoreExporter(self.store))  # spans feed /metrics
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(deps).compile(checkpointer=self.checkpointer)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        if deps.governance:
            deps.tracer.default_attributes["application_id"] = deps.governance.system.application_id

    # --- internals -------------------------------------------------------------------------

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def _resolve_dataset(self, dataset: str) -> dict[str, str]:
        if not _DATASET.match(dataset):
            raise Invalid(f"invalid dataset name {dataset!r}")
        root = (self.data_root / dataset).resolve()
        if root.parent != self.data_root or not root.is_dir():
            raise Invalid(f"unknown dataset {dataset!r}")
        paths = {s.value: str(root / FILE_NAMES[s]) for s in Source}
        missing = [p for p in paths.values() if not Path(p).is_file()]
        if missing:
            raise Invalid(f"dataset {dataset!r} is missing files", detail=missing)
        limit = self.deps.limits.max_file_bytes
        oversized = [Path(p).name for p in paths.values() if Path(p).stat().st_size > limit]
        if oversized:  # checked before anything reads the file into memory
            raise TooLarge(f"files exceed {limit} bytes", detail=oversized)
        return paths

    def _require_run(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if run is None:
            raise NotFound(f"run {run_id} not found")
        return run

    def _state(self, run_id: str):
        return self.graph.get_state(self._config(run_id))

    def _interrupt(self, run_id: str) -> dict | None:
        interrupts = self._state(run_id).interrupts
        return interrupts[0].value if interrupts else None

    def _drive(self, run_id: str, payload: Any) -> None:
        """Invoke or resume the graph, then derive the run status from where it stopped.

        Each invocation is one ``AGENT_EXEC`` span; node spans nest under it. A run that
        pauses for a human has one ``AGENT_EXEC`` per segment.
        """
        segment = "resume" if isinstance(payload, Command) else "start"
        try:
            with self.deps.tracer.span(
                run_id, "agent_run", "AGENT_EXEC", segment=segment, model_id=self.deps.llm.model_id
            ) as span:
                config = {
                    **self._config(run_id),
                    "recursion_limit": self.deps.limits.recursion_limit,
                }
                self.graph.invoke(payload, config)
                pending = self._interrupt(run_id)
                span.set(stopped_at=pending["type"] if pending else "completed")
        except Exception as exc:
            self.store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._locks.pop(run_id, None)
            raise RunFailed(f"run {run_id} failed: {exc}") from exc
        prune_checkpoints(self.checkpointer, run_id)
        pending = self._interrupt(run_id)
        if pending is None:
            self._locks.pop(run_id, None)  # finished: nothing can contend for this run again
            return  # emit_log already marked it completed
        status = {
            "mapping_confirmation": "awaiting_mapping_confirmation",
            "exception_review": "awaiting_review",
        }[pending["type"]]
        self.store.update_run(run_id, status=status)

    # --- public API ------------------------------------------------------------------------

    def start_run(self, dataset: str, analyst: str) -> dict:
        paths = self._resolve_dataset(dataset)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self.store.create_run(run_id, dataset, analyst, manifest_base(dataset, self.deps.llm))
        with self._locks[run_id]:
            self._drive(run_id, {"run_id": run_id, "dataset": dataset, "input_paths": paths})
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        values = self._state(run_id).values or {}
        exceptions = values.get("exceptions", {})
        counts: dict[str, int] = defaultdict(int)
        for e in exceptions.values():
            counts[e["status"]] += 1
        return {
            **run,
            "pending_action": self._interrupt(run_id),
            "exception_counts": dict(sorted(counts.items())),
            "recon_stats": values.get("recon_stats"),
        }

    def get_mapping(self, run_id: str) -> dict:
        self._require_run(run_id)
        values = self._state(run_id).values or {}
        if "mapping_proposal" not in values:
            raise Conflict("mapping has not been proposed yet")
        return {
            "proposal": values["mapping_proposal"],
            "problems": values.get("mapping_problems", []),
            "confirmed_mapping": values.get("mapping"),
            "confirmed_by": values.get("mapping_confirmed_by"),
            "profiles": values.get("profiles"),
            "input_screen": values.get("input_screen", []),
            "input_screen_full": values.get("input_screen_full", []),
        }

    def confirm_mapping(self, run_id: str, analyst: str, mapping: dict | None) -> dict:
        self._require_run(run_id)
        with self._locks[run_id]:
            pending = self._interrupt(run_id)
            if not pending or pending["type"] != "mapping_confirmation":
                raise Conflict("run is not awaiting mapping confirmation")
            values = self._state(run_id).values
            headers = {
                Source(p["source"]): [c["name"] for c in p["columns"]] for p in values["profiles"]
            }
            if mapping is not None:
                try:
                    candidate = ColumnMapping.model_validate(mapping)
                except ValueError as exc:
                    raise Invalid("mapping is not a valid ColumnMapping", detail=str(exc)) from exc
                problems = candidate.problems(headers)
            else:
                problems = values["mapping_problems"]
            if problems:
                raise Invalid("mapping cannot be applied to these files", detail=problems)
            self._drive(run_id, Command(resume={"analyst": analyst, "mapping": mapping}))
        return self.get_run(run_id)

    def list_exceptions(self, run_id: str) -> list[dict]:
        self._require_run(run_id)
        values = self._state(run_id).values or {}
        return [values["exceptions"][k] for k in sorted(values.get("exceptions", {}))]

    def review_exception(
        self, run_id: str, exception_id: str, analyst: str, decision: ReviewDecision
    ) -> dict:
        self._require_run(run_id)
        with self._locks[run_id]:
            pending = self._interrupt(run_id)
            if not pending or pending["type"] != "exception_review":
                raise Conflict("run is not awaiting exception review")
            if exception_id not in pending["pending"]:
                raise Conflict(f"exception {exception_id} is not pending review")
            if decision.action not in pending["allowed_actions"]:
                raise Conflict(
                    f"review round limit reached ({pending['max_rounds']}); "
                    f"allowed actions: {pending['allowed_actions']}"
                )
            round_ = pending["round"]
            body = {"analyst": analyst, **decision.model_dump(exclude_none=True)}
            try:
                self.store.put_review(run_id, exception_id, round_, analyst, body)
            except RuntimeError as exc:
                raise Conflict(str(exc)) from exc
            submitted = self.store.reviews(run_id, round_)
            outstanding = sorted(set(pending["pending"]) - set(submitted))
            if not outstanding:
                self._drive(run_id, Command(resume=submitted))
        return {**self.get_run(run_id), "review_round": round_, "awaiting": outstanding}

    def remediation_log(self, run_id: str) -> list[dict]:
        self._require_run(run_id)
        return self.store.remediation_log(run_id)

    def events(self, run_id: str) -> list[dict]:
        self._require_run(run_id)
        return self.store.events(run_id)

    # --- baseline + metrics ----------------------------------------------------------------

    def run_baseline(self, analyst: str, label: str, datasets: list[str] | None = None) -> dict:
        """Week 1 baseline: manual-process notes through the same gate; stored for drift."""
        if datasets:
            dirs = []
            for name in datasets:
                self._resolve_dataset(name)
                if not has_notes(self.data_root / name):
                    raise Invalid(f"dataset {name!r} has no manual_process_notes.json")
                dirs.append(self.data_root / name)
        else:
            dirs = sorted(d for d in self.data_root.iterdir() if d.is_dir() and has_notes(d))
        if not dirs:
            raise Invalid("no datasets with manual_process_notes.json under the data root")
        report = run_baseline(dirs, self.deps.gate, self.deps.recon_config)
        baseline_id = f"bl-{uuid.uuid4().hex[:12]}"
        self.store.save_baseline(baseline_id, label, analyst, report["summary"], report["results"])
        return self.store.latest_baseline() | {"baseline_id": baseline_id}

    def latest_baseline(self) -> dict:
        baseline = self.store.latest_baseline()
        if baseline is None:
            raise NotFound("no baseline recorded yet; POST /baselines first")
        return baseline

    def _exceptions_of(self, run_id: str) -> list[dict]:
        values = self._state(run_id).values or {}
        return list(values.get("exceptions", {}).values())

    def metrics(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        return metrics_mod.run_metrics(
            run,
            self.store.spans(run_id),
            self._exceptions_of(run_id),
            self.store.latest_baseline(),
        )

    def metrics_summary(self) -> dict:
        runs = self.store.list_runs()
        per_run, first = [], []
        for run in runs:
            exceptions = self._exceptions_of(run["run_id"])
            per_run.append(
                metrics_mod.run_metrics(run, self.store.spans(run["run_id"]), exceptions, None)
            )
            first.extend(e["verdicts"][0] for e in exceptions if e.get("verdicts"))
        return metrics_mod.summary_metrics(per_run, first, self.store.latest_baseline())

    def trajectory(self, run_id: str) -> list[dict]:
        self._require_run(run_id)
        return (self._state(run_id).values or {}).get("trajectory", [])

    # --- harness + memory ------------------------------------------------------------------

    def harness(self, run_id: str) -> dict:
        """What this run consumed against its limits, and every limit it hit."""
        self._require_run(run_id)
        values = self._state(run_id).values or {}
        limits = self.deps.limits
        return {
            "usage": values.get("usage") or RunBudget(limits).snapshot(),
            "review_rounds": values.get("review_round", 0),
            "limits": {
                **{f.name: getattr(limits, f.name) for f in dataclasses.fields(limits)},
                "recursion_limit": limits.recursion_limit,
            },
            "limit_events": self.store.events(run_id, "limit_reached"),
            "degraded": sorted(
                k for k, v in values.get("exceptions", {}).items() if v.get("degraded")
            ),
            "state_bytes": len(canonical_json(values)),
        }

    def list_memory(self) -> list[dict]:
        return self.store.list_mappings()

    def forget_memory(self, fingerprint: str, analyst: str) -> dict:
        if not self.store.forget_mapping(fingerprint):
            raise NotFound(f"no remembered mapping {fingerprint}")
        self.store.record_event("_system", "mapping_forgotten", analyst, {}, subject_id=fingerprint)
        return {"forgotten": fingerprint}

    # --- governance (Use Case 2) -----------------------------------------------------------

    @property
    def governance(self):
        if self.deps.governance is None:
            raise NotFound("the governance layer is not enabled on this service")
        return self.deps.governance
