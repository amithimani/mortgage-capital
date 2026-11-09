"""LLM provider over an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Points at the mock today (``mc-recon mock``). Pointing ``MC_LLM_BASE_URL`` at OpenAI or any
compatible gateway needs no code change.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from ..transport import (
    DependencyRejected,
    DependencyUnavailable,
    RetryPolicy,
    post_json,
    reachable,
)
from .prompts import render_messages
from .provider import (
    LLMError,
    LLMOutputValidationError,
    LLMRequest,
    LLMResult,
    LLMUnavailable,
    T,
    TokenUsage,
)


class OpenAICompatibleProvider:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        api_key: str = "",
        retry: RetryPolicy = RetryPolicy(),  # noqa: B008 — immutable
        sleep: Any = None,
    ):
        self.client = client
        self.model_id = model
        self.api_key = api_key
        self.retry = retry
        self._sleep = sleep

    def ready(self) -> bool:
        return reachable(self.client)

    def generate(self, request: LLMRequest, output_schema: type[T]) -> LLMResult[T]:
        body = {
            "model": self.model_id,
            "temperature": request.temperature,
            "messages": render_messages(request),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                },
            },
            "metadata": {"cache_key": request.cache_key},
        }
        kwargs: dict[str, Any] = {"retry": self.retry}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        try:
            data = post_json(
                self.client,
                "/v1/chat/completions",
                body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                **kwargs,
            )
        except DependencyUnavailable as exc:
            raise LLMUnavailable(str(exc)) from exc
        except DependencyRejected as exc:
            raise LLMError(str(exc)) from exc

        try:
            content = data["choices"][0]["message"]["content"]
            output = output_schema.model_validate(json.loads(content))
        except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
            raise LLMOutputValidationError(
                f"response does not match {output_schema.__name__}: {exc}"
            ) from exc
        usage = data.get("usage") or {}
        return LLMResult(
            output=output,
            model_id=data.get("model", self.model_id),
            cache_key=request.cache_key,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            raw_text=content,
        )
