# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model response timeouts, usage observations and Ark reasoning fields."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

from openviking.integrations.langchain.ov_context.token_estimation import (
    PayloadPrefix,
    PayloadUsageCounter,
)


def windowed_model(model: Any, context_window: int) -> Any:
    """Apply a Session window to a local model and reset provider usage anchors."""
    profile = dict(model.profile or {})
    known_limit = profile.get("max_input_tokens")
    profile["max_input_tokens"] = (
        min(context_window, known_limit)
        if isinstance(known_limit, int) and not isinstance(known_limit, bool)
        else context_window
    )
    local = model.model_copy(update={"profile": profile})
    reset = getattr(local, "reset_input_budget", None)
    if callable(reset):
        reset()
    return local


class ModelCallTimeoutError(TimeoutError):
    """A response or the next streaming packet did not arrive in time."""


class TimedChatOpenAI(ChatOpenAI):
    async_call_timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, exclude=True
    )
    # Newer SDKs also have a chunk watchdog. Let this adapter's configured
    # response timer own it, rather than applying the SDK's shorter default.
    stream_chunk_timeout: float | None = Field(default=None, gt=0, exclude=True)
    _input_budget: PayloadUsageCounter = PrivateAttr(default_factory=PayloadUsageCounter)

    @staticmethod
    def _observe_reasoning(source: dict[str, Any], message: BaseMessage) -> None:
        # Observation metadata is excluded from Chat Completions request bodies.
        # Capturing a response must not enable provider-specific history replay.
        reasoning = source.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            message.response_metadata["observed_reasoning_content"] = reasoning
        encrypted = source.get("encrypted_content")
        if isinstance(encrypted, str) and encrypted:
            message.response_metadata["observed_encrypted_content_present"] = True

    def _create_chat_result(
        self, response: Any, generation_info: dict[str, Any] | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for choice, generation in zip(data["choices"], result.generations, strict=True):
            self._observe_reasoning(choice["message"], generation.message)
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices", [])
        if result is not None and isinstance(result.message, AIMessageChunk) and choices:
            self._observe_reasoning(choices[0].get("delta") or {}, result.message)
        return result

    def reset_input_budget(self) -> None:
        self._input_budget = PayloadUsageCounter()

    def estimate_input_tokens(self, payload: dict[str, Any]) -> int:
        return self._input_budget.estimate(payload)

    def input_token_diagnostics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._input_budget.diagnostics(payload)

    def _observe_input(
        self, prefix: PayloadPrefix | None, usage: Any, encrypted: str | None = None
    ) -> None:
        if isinstance(usage, dict):
            self._input_budget.observe_prefix(prefix, usage.get("input_tokens"))
            self._input_budget.observe_reasoning(encrypted, usage)

    @staticmethod
    def _check_deadline(deadline: asyncio.Timeout) -> None:
        # A lower-level wait_for may return a simultaneously completed packet
        # instead of propagating cancellation. Check this packet's wait deadline
        # explicitly, including that cancellation race.
        when = deadline.when()
        if when is not None and asyncio.get_running_loop().time() >= when:
            raise ModelCallTimeoutError("Waiting for the model response timed out")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prefix = self._input_budget.capture(self.preview_payload(messages, stop=stop, **kwargs))
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if result.generations:
            self._observe_input(
                prefix,
                getattr(result.generations[0].message, "usage_metadata", None),
                result.generations[0].message.additional_kwargs.get("encrypted_content"),
            )
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prefix = self._input_budget.capture(self.preview_payload(messages, stop=stop, **kwargs))
        deadline = asyncio.timeout(self.async_call_timeout_seconds)
        try:
            async with deadline:
                result = await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                self._check_deadline(deadline)
        except TimeoutError as exc:
            if deadline.expired():
                raise ModelCallTimeoutError("Waiting for the model response timed out") from exc
            raise
        if result.generations:
            self._observe_input(
                prefix,
                getattr(result.generations[0].message, "usage_metadata", None),
                result.generations[0].message.additional_kwargs.get("encrypted_content"),
            )
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prefix = self._input_budget.capture(self.preview_payload(messages, stop=stop, **kwargs))
        usage = None
        encrypted_parts: list[str] = []
        for chunk in super()._stream(
            messages, stop=stop, run_manager=run_manager, stream_usage=stream_usage, **kwargs
        ):
            encrypted = chunk.message.additional_kwargs.get("encrypted_content")
            if encrypted:
                encrypted_parts.append(encrypted)
            if chunk.message.usage_metadata is not None:
                usage = chunk.message.usage_metadata
                chunk = chunk.model_copy(
                    update={"message": chunk.message.model_copy(update={"usage_metadata": None})}
                )
            yield chunk
        self._observe_input(prefix, usage, "".join(encrypted_parts) or None)
        if usage is not None:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prefix = self._input_budget.capture(self.preview_payload(messages, stop=stop, **kwargs))
        usage = None
        encrypted_parts: list[str] = []
        async with aclosing(
            super()._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                stream_usage=stream_usage,
                **kwargs,
            )
        ) as stream:
            while True:
                # Each SDK packet (text, reasoning, tool arguments or usage)
                # starts a new idle window. Consumer/callback work is outside it.
                deadline = asyncio.timeout(self.async_call_timeout_seconds)
                try:
                    async with deadline:
                        self._check_deadline(deadline)
                        chunk = await anext(stream)
                        self._check_deadline(deadline)
                except StopAsyncIteration:
                    self._check_deadline(deadline)
                    break
                except TimeoutError as exc:
                    if deadline.expired():
                        raise ModelCallTimeoutError(
                            "No model stream packet arrived before the idle timeout"
                        ) from exc
                    raise
                encrypted = chunk.message.additional_kwargs.get("encrypted_content")
                if encrypted:
                    encrypted_parts.append(encrypted)
                if chunk.message.usage_metadata is not None:
                    usage = chunk.message.usage_metadata
                    # OpenAI-compatible endpoints may repeat cumulative usage on
                    # finish and usage-only chunks. Publish the last snapshot once
                    # so message aggregation and the calibration anchor agree.
                    chunk = chunk.model_copy(
                        update={
                            "message": chunk.message.model_copy(update={"usage_metadata": None})
                        }
                    )
                yield chunk
        self._observe_input(prefix, usage, "".join(encrypted_parts) or None)
        if usage is not None:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))

    def preview_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Serialize the provider input for Middleware token estimation."""
        return self._get_request_payload(input_, stop=stop, **kwargs)


class ArkChatOpenAI(TimedChatOpenAI):
    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        wire_messages = payload["messages"]
        if len(messages) != len(wire_messages):
            raise ValueError("Unexpected Ark message mapping")
        for message, wire in zip(messages, wire_messages, strict=True):
            if isinstance(message, AIMessage):
                self._copy_reasoning(message.additional_kwargs, wire)
        return payload

    @staticmethod
    def _copy_reasoning(source: dict[str, Any], target: dict[str, Any]) -> None:
        for field in ("reasoning_content", "encrypted_content"):
            value = source.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"Unexpected Ark {field} field type")
                target[field] = value

    def _create_chat_result(
        self, response: Any, generation_info: dict[str, Any] | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for choice, generation in zip(data["choices"], result.generations, strict=True):
            self._copy_reasoning(choice["message"], generation.message.additional_kwargs)
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices", [])
        if result is not None and isinstance(result.message, AIMessageChunk) and choices:
            delta = choices[0].get("delta") or {}
            self._copy_reasoning(delta, result.message.additional_kwargs)
        return result
