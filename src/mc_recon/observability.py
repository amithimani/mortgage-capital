"""Tracing seam. Nodes open spans through :class:`Tracer`; Disseqt plugs in later.

The span kinds match ``disseqt_agentic_sdk.enums.SpanKind`` (``AGENT_EXEC``, ``MODEL_EXEC``,
``TOOL_EXEC``), so the Week 3 ``DisseqtTracer`` is a thin adapter over ``start_trace``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

SpanKind = Literal["AGENT_EXEC", "MODEL_EXEC", "TOOL_EXEC", "OVERSIGHT"]


@dataclass
class Span:
    run_id: str
    name: str
    kind: SpanKind
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class Tracer(Protocol):
    def span(
        self, run_id: str, name: str, kind: SpanKind, **attributes: Any
    ) -> Any: ...  # context manager yielding Span


class NullTracer:
    @contextmanager
    def span(self, run_id: str, name: str, kind: SpanKind, **attributes: Any) -> Iterator[Span]:
        yield Span(run_id, name, kind, dict(attributes))


class InMemoryTracer:
    """Keeps finished spans; used by tests and handy for local debugging."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, run_id: str, name: str, kind: SpanKind, **attributes: Any) -> Iterator[Span]:
        s = Span(run_id, name, kind, dict(attributes))
        try:
            yield s
        except Exception as exc:
            s.status = f"error: {type(exc).__name__}"
            raise
        finally:
            self.spans.append(s)

    def names(self, run_id: str | None = None) -> list[str]:
        return [s.name for s in self.spans if run_id is None or s.run_id == run_id]
