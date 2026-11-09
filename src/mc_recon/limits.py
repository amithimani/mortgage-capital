"""Agent harness: hard limits on what a single run may consume.

The agent can loop in exactly two places (a schema re-ask, the review -> re-check cycle) and can
fan out once per finding. Every one of those is bounded here, in one object, so the worst case
of a run is a number that can be read rather than a behaviour to discover.

What happens at a limit depends on what the limit protects:

* A **per-finding** LLM step that cannot proceed (budget spent, circuit open, output still
  invalid, provider down) *degrades*: the deterministic finding is kept, gets template text,
  and is held for a human. A limit can cost automation. It can never lose a finding.
* A limit that makes the whole run meaningless (mapping unavailable, too many findings, input
  too large) *fails the run loudly*.

``usage`` lives in graph state, so budgets survive a pause for a human and a process restart.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .llm.provider import TokenUsage
from .metrics import MODEL_PRICES_PER_MTOK


@dataclass(frozen=True)
class AgentLimits:
    """Per-run limits. Each is overridable with ``MC_LIMIT_<NAME>`` (see ``config.py``)."""

    max_reasks: int = 1
    """Extra attempts after schema-invalid output, per LLM call."""
    max_review_rounds: int = 3
    """Review -> re-check cycles. In the last round ``edit`` is refused: approve or reject."""
    max_llm_calls: int = 500
    max_total_tokens: int = 500_000
    max_cost_usd: float = 5.0
    max_llm_seconds: float = 600.0
    """Time inside LLM-calling nodes only: a run waiting on a human is not spending."""
    max_input_tokens_per_call: int = 16_000
    """Over this a call is refused. Context is never silently truncated to fit."""
    max_consecutive_llm_failures: int = 3
    """Circuit breaker: stop calling a provider that keeps failing within a run."""
    max_findings: int = 500
    """More findings than this almost always means a wrong mapping, not a bad book."""
    max_file_bytes: int = 10_000_000
    max_rows: int = 50_000

    def __post_init__(self) -> None:
        if self.max_review_rounds < 1:
            raise ValueError("max_review_rounds must be at least 1")
        if self.max_reasks < 0:
            raise ValueError("max_reasks must not be negative")

    @property
    def recursion_limit(self) -> int:
        """LangGraph superstep backstop, derived from the bound on the review loop.

        Eight nodes on the straight path, two more per review round, plus a small margin. The
        review cap is what ends the loop; this only catches a routing bug.
        """
        return 8 + 2 * self.max_review_rounds + 2


class LimitExceeded(RuntimeError):
    def __init__(self, limit: str, detail: str):
        super().__init__(f"{limit}: {detail}")
        self.limit, self.detail = limit, detail


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    return max(1, sum(len(m["content"]) for m in messages) // 4)


class RunBudget:
    """One run's spend. Rebuilt from state at the start of a node and written back at the end."""

    def __init__(
        self,
        limits: AgentLimits,
        usage: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        u = usage or {}
        self.limits = limits
        self.llm_calls: int = u.get("llm_calls", 0)
        self.total_tokens: int = u.get("total_tokens", 0)
        self.cost_usd: float = u.get("cost_usd", 0.0)
        self.failures: int = u.get("llm_failures", 0)
        self.consecutive_failures: int = u.get("consecutive_llm_failures", 0)
        self._prior_seconds: float = u.get("llm_seconds", 0.0)
        self._clock = clock
        self._t0 = clock()

    @property
    def llm_seconds(self) -> float:
        return self._prior_seconds + (self._clock() - self._t0)

    def admit(self, estimated_input_tokens: int) -> None:
        """Called before every attempt. Raises rather than letting the call happen."""
        lim = self.limits
        if estimated_input_tokens > lim.max_input_tokens_per_call:
            raise LimitExceeded(
                "max_input_tokens_per_call",
                f"~{estimated_input_tokens} tokens > {lim.max_input_tokens_per_call}",
            )
        if self.consecutive_failures >= lim.max_consecutive_llm_failures:
            raise LimitExceeded(
                "max_consecutive_llm_failures",
                f"circuit open after {self.consecutive_failures} consecutive failures",
            )
        if self.llm_calls >= lim.max_llm_calls:
            raise LimitExceeded("max_llm_calls", f"{self.llm_calls} calls used")
        if self.total_tokens + estimated_input_tokens > lim.max_total_tokens:
            raise LimitExceeded("max_total_tokens", f"{self.total_tokens} tokens used")
        if self.cost_usd >= lim.max_cost_usd:
            raise LimitExceeded("max_cost_usd", f"${self.cost_usd:.4f} spent")
        if self.llm_seconds >= lim.max_llm_seconds:
            raise LimitExceeded("max_llm_seconds", f"{self.llm_seconds:.1f}s used")
        self.llm_calls += 1

    def charge(self, usage: TokenUsage, model_id: str) -> None:
        self.consecutive_failures = 0
        self.total_tokens += usage.input_tokens + usage.output_tokens
        price_in, price_out = MODEL_PRICES_PER_MTOK.get(model_id, (0.0, 0.0))
        self.cost_usd += (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6

    def charge_failure(self) -> None:
        self.failures += 1
        self.consecutive_failures += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_failures": self.failures,
            "consecutive_llm_failures": self.consecutive_failures,
            "llm_seconds": round(self.llm_seconds, 4),
        }
