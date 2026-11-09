"""HTTP clients for the Disseqt surface: policy evaluation, trace export, registration.

The route and payload shapes are **our assumption**. Disseqt's API docs are behind a login, so
these match the mock in ``mocks/app.py``. Once access exists, only this module should change.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import httpx

from .observability import Span
from .policy import PolicySubject, PolicyVerdict
from .transport import DependencyRejected, DependencyUnavailable, RetryPolicy, post_json, reachable

log = logging.getLogger(__name__)


def register_application(client: httpx.Client, api_key: str, name: str) -> str:
    data = post_json(client, "/api/v1/applications", {"name": name}, headers={"X-API-Key": api_key})
    return data["application_id"]


class HttpPolicyGate:
    """A transport or parse error raises here. ``evaluate_fail_closed`` turns that into BLOCK."""

    def __init__(
        self,
        client: httpx.Client,
        api_key: str,
        application_id: str,
        policy_id: str = "exception-policy",
        retry: RetryPolicy = RetryPolicy(),  # noqa: B008 — immutable
    ):
        self.client = client
        self.policy_id = policy_id
        self.retry = retry
        self._headers = {"X-API-Key": api_key, "X-Application-Id": application_id}

    def ready(self) -> bool:
        return reachable(self.client)

    def _validate(self, scope: str, **body: Any) -> PolicyVerdict:
        data = post_json(
            self.client,
            "/api/v1/validate",
            {"policy_id": self.policy_id, "scope": scope, **body},
            headers=self._headers,
            retry=self.retry,
        )
        return PolicyVerdict.model_validate(data)

    def evaluate(self, subject: PolicySubject) -> PolicyVerdict:
        return self._validate("exception", subject=subject.model_dump(mode="json"))

    def evaluate_run(self, trajectory: Sequence[dict[str, Any]]) -> PolicyVerdict:
        return self._validate("run", trajectory=list(trajectory))

    def screen_input(self, text: str) -> PolicyVerdict:
        return self._validate("input", text=text)


class HttpTraceExporter:
    """Batches spans and ships them when a graph segment ends.

    The buffer is bounded and one failed flush keeps the spans for the next, so a Disseqt outage
    costs at most the oldest telemetry and never memory or the run. The local span store is the
    source of truth.
    """

    def __init__(
        self,
        client: httpx.Client,
        api_key: str,
        application_id: str,
        batch_size: int = 100,
        buffer_size: int = 5000,
    ):
        self.client = client
        self.batch_size = batch_size
        self._headers = {"X-API-Key": api_key, "X-Application-Id": application_id}
        self._buffer: deque[Span] = deque(maxlen=buffer_size)

    def export(self, span: Span) -> None:
        self._buffer.append(span)
        if span.kind == "AGENT_EXEC" or len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> bool:
        while self._buffer:
            batch = [self._buffer[i] for i in range(min(self.batch_size, len(self._buffer)))]
            try:
                post_json(
                    self.client,
                    "/api/v1/traces",
                    {"spans": [asdict(s) for s in batch]},
                    headers=self._headers,
                    retry=RetryPolicy(attempts=1),
                )
            except (DependencyUnavailable, DependencyRejected) as exc:
                log.warning("trace export failed, keeping %d spans: %s", len(self._buffer), exc)
                return False
            for _ in batch:
                self._buffer.popleft()
        return True
