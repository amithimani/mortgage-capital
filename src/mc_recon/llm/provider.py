"""Provider-agnostic LLM interface.

Every LLM call in the agent is an :class:`LLMRequest` with:

* a ``task`` (closed set) and a ``prompt_version`` (bumped whenever a template changes),
* a structured, JSON-serialisable ``payload`` — the *only* input the model gets; prompt
  templates render messages from it,
* a Pydantic ``output_schema`` the response must validate against.

``cache_key`` hashes (task, prompt_version, payload). It identifies the call for the
recorded fake, for audit (stored on the MODEL_EXEC span) and for drift comparison between
the fake and a live model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class LLMTask(StrEnum):
    PROPOSE_MAPPING = "propose_mapping"
    CLASSIFY_EXCEPTION = "classify_exception"
    DRAFT_RATIONALE = "draft_rationale"


PROMPT_VERSIONS: dict[LLMTask, str] = {
    LLMTask.PROPOSE_MAPPING: "mapping.v1",
    LLMTask.CLASSIFY_EXCEPTION: "classify.v1",
    LLMTask.DRAFT_RATIONALE: "rationale.v1",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: LLMTask
    prompt_version: str
    payload: dict[str, Any]
    temperature: float = 0.0

    @property
    def cache_key(self) -> str:
        body = canonical_json(
            {
                "task": self.task.value,
                "prompt_version": self.prompt_version,
                "payload": self.payload,
            }
        )
        return hashlib.sha256(body.encode()).hexdigest()


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False
    """True when counts are a chars/4 estimate (the fake), not provider-reported."""


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    output: T
    model_id: str
    cache_key: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_text: str | None = None


class LLMProvider(Protocol):
    model_id: str

    def generate(self, request: LLMRequest, output_schema: type[T]) -> LLMResult[T]: ...


class LLMError(RuntimeError):
    pass


class LLMOutputValidationError(LLMError):
    """The model (or a recording) returned output that doesn't match the schema."""
