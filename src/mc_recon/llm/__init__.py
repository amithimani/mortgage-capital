"""LLM adapter: provider protocol, structured outputs, payload builders and the recorded fake.

The real ``OpenAIProvider`` lands in Week 2 behind the same :class:`LLMProvider` protocol.
"""

from .fake import (
    MissingRecordingError,
    RecordedFakeProvider,
    Recording,
    RecordingProvider,
    RecordingSet,
)
from .payloads import classification_request, mapping_request, rationale_request
from .provider import (
    PROMPT_VERSIONS,
    LLMError,
    LLMOutputValidationError,
    LLMProvider,
    LLMRequest,
    LLMResult,
    LLMTask,
    LLMUnavailable,
    TokenUsage,
)
from .schemas import ClassificationOutput, MappingProposal, RationaleOutput

__all__ = [
    "PROMPT_VERSIONS",
    "ClassificationOutput",
    "LLMError",
    "LLMOutputValidationError",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "LLMTask",
    "LLMUnavailable",
    "MappingProposal",
    "MissingRecordingError",
    "RationaleOutput",
    "RecordedFakeProvider",
    "Recording",
    "RecordingProvider",
    "RecordingSet",
    "TokenUsage",
    "classification_request",
    "mapping_request",
    "rationale_request",
]
