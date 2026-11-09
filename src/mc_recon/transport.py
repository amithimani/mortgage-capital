"""HTTP calls to external dependencies: bounded retries on transient failures only.

Used by the LLM provider and the Disseqt clients, against the mocks today and real endpoints
later. ``429``, ``5xx`` and connection errors are retried with capped, jittered backoff. Any
other ``4xx`` means our request is wrong, so retrying cannot help and it fails immediately.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class DependencyUnavailable(RuntimeError):
    """Retries are spent: the dependency is down or overloaded."""


class DependencyRejected(RuntimeError):
    """A non-retryable 4xx."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_seconds: float = 0.2
    cap_seconds: float = 4.0

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.cap_seconds)
            except ValueError:
                pass  # HTTP-date form: fall through to backoff
        ceiling = min(self.cap_seconds, self.base_seconds * 2 ** (attempt - 1))
        return random.uniform(ceiling / 2, ceiling)  # noqa: S311 — jitter, not security


def post_json(
    client: httpx.Client,
    path: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    retry: RetryPolicy = RetryPolicy(),  # noqa: B008 — immutable
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    last = "no attempt made"
    for attempt in range(1, retry.attempts + 1):
        retry_after = None
        try:
            resp = client.post(path, json=body, headers=headers or {})
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code not in RETRYABLE_STATUS:
                raise DependencyRejected(resp.status_code, resp.text)
            last = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("retry-after")
        if attempt < retry.attempts:
            sleep(retry.delay(attempt, retry_after))
    raise DependencyUnavailable(f"{path} failed after {retry.attempts} attempts: {last}")


def reachable(client: httpx.Client, path: str = "/") -> bool:
    """True on any HTTP response. This asks "is it up", which works for endpoints we don't own."""
    try:
        client.get(path, timeout=2.0)
        return True
    except httpx.HTTPError:
        return False
