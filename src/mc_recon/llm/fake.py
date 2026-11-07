"""Deterministic LLM stand-ins for CI and local development.

* :class:`RecordedFakeProvider` replays stored responses keyed by ``LLMRequest.cache_key``.
  A request with no recording raises :class:`MissingRecordingError` — the fake never
  improvises, so a changed payload or prompt version is caught, not silently passed.
* :class:`RecordingProvider` wraps a real provider and captures its responses, so a live run
  can be frozen into a regression fixture.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .provider import (
    LLMError,
    LLMOutputValidationError,
    LLMProvider,
    LLMRequest,
    LLMResult,
    LLMTask,
    T,
)


class Recording(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: LLMTask
    prompt_version: str
    cache_key: str
    output: dict[str, Any]
    label: str | None = None
    """Human-readable hint (e.g. discrepancy id) — not used for lookup."""
    source_model: str = "oracle"
    """``oracle`` for ground-truth recordings, or the model id that produced it."""


class RecordingSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    recordings: list[Recording] = []

    @classmethod
    def load(cls, path: Path) -> RecordingSet:
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path) -> None:
        ordered = sorted(self.recordings, key=lambda r: (r.task.value, r.label or "", r.cache_key))
        body = RecordingSet(version=self.version, recordings=ordered)
        Path(path).write_text(body.model_dump_json(indent=2) + "\n")


class MissingRecordingError(LLMError):
    def __init__(self, request: LLMRequest):
        self.request = request
        super().__init__(
            f"No recording for task={request.task.value} prompt_version={request.prompt_version} "
            f"cache_key={request.cache_key[:16]}… — re-record or update the fixture."
        )


class RecordedFakeProvider:
    model_id = "recorded-fake"

    def __init__(self, recordings: Iterable[Recording]):
        self._index: dict[str, Recording] = {}
        for rec in recordings:
            existing = self._index.get(rec.cache_key)
            if existing is not None and existing.output != rec.output:
                raise ValueError(f"conflicting recordings for cache_key {rec.cache_key}")
            self._index[rec.cache_key] = rec
        self.calls: list[LLMRequest] = []

    @classmethod
    def from_files(cls, *paths: Path) -> RecordedFakeProvider:
        recs: list[Recording] = []
        for p in paths:
            recs.extend(RecordingSet.load(p).recordings)
        return cls(recs)

    def __len__(self) -> int:
        return len(self._index)

    def generate(self, request: LLMRequest, output_schema: type[T]) -> LLMResult[T]:
        self.calls.append(request)
        rec = self._index.get(request.cache_key)
        if rec is None:
            raise MissingRecordingError(request)
        try:
            output = output_schema.model_validate(rec.output)
        except ValidationError as exc:
            raise LLMOutputValidationError(
                f"recording {request.cache_key[:16]}… does not match {output_schema.__name__}: "
                f"{exc}"
            ) from exc
        return LLMResult(output=output, model_id=self.model_id, cache_key=request.cache_key)


class RecordingProvider:
    """Pass-through to ``delegate`` that keeps every response for later replay."""

    def __init__(self, delegate: LLMProvider):
        self.delegate = delegate
        self.model_id = delegate.model_id
        self.recordings: list[Recording] = []

    def generate(self, request: LLMRequest, output_schema: type[T]) -> LLMResult[T]:
        result = self.delegate.generate(request, output_schema)
        self.recordings.append(
            Recording(
                task=request.task,
                prompt_version=request.prompt_version,
                cache_key=request.cache_key,
                output=result.output.model_dump(mode="json"),
                source_model=result.model_id,
            )
        )
        return result

    def save(self, path: Path) -> None:
        RecordingSet(recordings=self.recordings).save(path)
