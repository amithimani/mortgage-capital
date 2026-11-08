"""Tracing: nested, timed spans with pluggable exporters.

Span kinds match ``disseqt_agentic_sdk.enums.SpanKind`` (``AGENT_EXEC`` for the orchestration
of one graph invocation, ``MODEL_EXEC`` for each LLM call, ``TOOL_EXEC`` for parser, engine
and writer), plus ``OVERSIGHT`` for human checkpoints. Parent/child links come from a
``ContextVar``, so a node's spans nest under the run's ``AGENT_EXEC`` span without any
plumbing. (LangGraph copies the context into worker threads.)

Exporters receive finished spans. Today: :class:`StoreExporter` (persists spans for
``/metrics``) and :class:`MemoryExporter` (tests). The Disseqt exporter is the Week 3 addition.
An exporter that fails never fails the run.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

SpanKind = Literal["AGENT_EXEC", "MODEL_EXEC", "TOOL_EXEC", "OVERSIGHT"]

log = logging.getLogger(__name__)
_current: ContextVar[Span | None] = ContextVar("mc_recon_current_span", default=None)


@dataclass
class Span:
    run_id: str
    name: str
    kind: SpanKind
    attributes: dict[str, Any] = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    started_at: str = ""
    duration_ms: float | None = None
    status: str = "ok"

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class SpanExporter(Protocol):
    def export(self, span: Span) -> None: ...


class MemoryExporter:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class StoreExporter:
    def __init__(self, store: Any) -> None:
        self.store = store

    def export(self, span: Span) -> None:
        self.store.record_span(span)


class Tracer:
    def __init__(self, exporters: list[SpanExporter] | None = None) -> None:
        self.exporters: list[SpanExporter] = list(exporters or [])

    def add_exporter(self, exporter: SpanExporter) -> None:
        self.exporters.append(exporter)

    @contextmanager
    def span(self, run_id: str, name: str, kind: SpanKind, **attributes: Any) -> Iterator[Span]:
        parent = _current.get()
        s = Span(
            run_id=run_id,
            name=name,
            kind=kind,
            attributes=dict(attributes),
            parent_id=parent.span_id if parent and parent.run_id == run_id else None,
            started_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        )
        token = _current.set(s)
        t0 = time.perf_counter()
        try:
            yield s
        except BaseException as exc:
            # GraphInterrupt is how LangGraph pauses for a human; it isn't a failure.
            s.status = (
                "interrupted"
                if type(exc).__name__ == "GraphInterrupt"
                else (f"error: {type(exc).__name__}")
            )
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - t0) * 1000, 3)
            _current.reset(token)
            for exporter in self.exporters:
                try:
                    exporter.export(s)
                except Exception:  # noqa: BLE001 — telemetry must never break a run
                    log.exception("span exporter %r failed", exporter)


class NullTracer(Tracer):
    """No exporters by default (the service adds a StoreExporter)."""


class InMemoryTracer(Tracer):
    """Tracer that keeps finished spans in memory; used by tests and local debugging."""

    def __init__(self) -> None:
        self._memory = MemoryExporter()
        super().__init__([self._memory])

    @property
    def spans(self) -> list[Span]:
        return self._memory.spans

    def names(self, run_id: str | None = None) -> list[str]:
        return [s.name for s in self.spans if run_id is None or s.run_id == run_id]
