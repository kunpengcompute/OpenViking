# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared request estimates and calibration from completed provider responses."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool

from openviking.utils.token_estimation import estimate_text_tokens

PayloadPrefix = tuple[str, tuple[str, ...]]
_OUTPUT_FIELDS = {
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
}


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _text_only(message: Any) -> bool:
    if not isinstance(message, dict) or message.keys() - {
        "role",
        "content",
        "name",
        "tool_calls",
        "function_call",
        "tool_call_id",
        "reasoning_content",
        "encrypted_content",
        "refusal",
    }:
        return False
    if any(
        message.get(key) is not None and not isinstance(message[key], str)
        for key in ("encrypted_content", "reasoning_content")
    ):
        return False
    content = message.get("content")
    return (
        content is None
        or isinstance(content, str)
        or (
            isinstance(content, list)
            and all(
                isinstance(block, dict)
                and block.get("type") in {"text", "input_text"}
                and isinstance(block.get("text"), str)
                for block in content
            )
        )
    )


def _message_budget(
    message: dict[str, Any], reasoning_costs: Mapping[str, int], tokens_per_image: int = 1024
) -> int:
    if not _text_only(message):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            text_blocks = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in {"image_url", "image", "input_image"}
                )
            ]
            image_count = len(content) - len(text_blocks)
            if image_count:
                # Base64/URL bytes carry an image, not text presented to the
                # model. Price known image blocks by the image allowance while
                # preserving text, framing and reasoning accounting.
                visible = {**message, "content": text_blocks}
                return (
                    _message_budget(visible, reasoning_costs, tokens_per_image)
                    + image_count * tokens_per_image
                )
        # Unknown provider fields and other media keep the wire fallback.
        return len(_encode(message).encode("utf-8")) + tokens_per_image
    visible = dict(message)
    encrypted = visible.pop("encrypted_content", None)
    reasoning = visible.pop("reasoning_content", None)
    thought_tokens = 0
    if encrypted:
        # Ark prioritizes encrypted reasoning over its visible summary. Ciphertext
        # size cannot reveal the decoded token count. Prefer confirmed reasoning
        # usage; cold restored history retains an explicitly conservative fallback.
        key = hashlib.sha256(encrypted.encode()).hexdigest()
        thought_tokens = reasoning_costs.get(key, len(encrypted.encode("utf-8")))
    elif reasoning:
        thought_tokens = estimate_text_tokens(reasoning)
    return estimate_text_tokens(_encode(visible)) + thought_tokens + 64


def payload_text_budget(
    payload: dict[str, Any],
    *,
    reasoning_costs: Mapping[str, int] | None = None,
    tokens_per_image: int = 1024,
) -> int:
    """CJK-aware text estimate, framing reserve and provider reasoning cost.

    Chinese/Japanese/Korean characters weigh 1.5, non-BMP characters 2 and other
    characters 0.25. This is an estimate, not a provider tokenizer or wire size.
    """
    costs = reasoning_costs or {}
    settings = {key: value for key, value in payload.items() if key not in _OUTPUT_FIELDS}
    messages = payload.get("messages", [])
    return (
        estimate_text_tokens(_encode(settings))
        + 256
        + sum(_message_budget(message, costs, tokens_per_image) for message in messages)
    )


def serialized_text_budget(
    messages: Iterable[BaseMessage],
    *,
    tools: list[Any] | None = None,
    tokens_per_image: int = 1024,
    **settings: Any,
) -> int:
    messages = list(messages)
    wire_messages = convert_to_openai_messages(messages)
    for message, wire in zip(messages, wire_messages, strict=True):
        if isinstance(message, AIMessage):
            for field in ("reasoning_content", "encrypted_content"):
                if field in message.additional_kwargs:
                    wire[field] = message.additional_kwargs[field]
    return payload_text_budget(
        {
            **settings,
            "messages": wire_messages,
            "tools": [convert_to_openai_tool(tool) for tool in (tools or [])],
        },
        tokens_per_image=tokens_per_image,
    )


@dataclass(frozen=True)
class _UsageAnchor:
    settings: str
    messages: tuple[str, ...]
    input_tokens: int


class PayloadUsageCounter:
    """Graph-local input anchor and bounded confirmed reasoning ledger.

    Only successful provider responses may train this counter. Message metadata
    from a checkpoint or caller is never accepted as provider input usage.
    """

    def __init__(self) -> None:
        self._anchor: _UsageAnchor | None = None
        self._reasoning_costs: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def capture(payload: dict[str, Any]) -> PayloadPrefix | None:
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or payload.get("n", 1) != 1
            or not all(_text_only(message) for message in messages)
        ):
            return None
        settings = {key: value for key, value in payload.items() if key not in _OUTPUT_FIELDS}
        return _encode(settings), tuple(_encode(message) for message in messages)

    def observe(self, payload: dict[str, Any], input_tokens: Any) -> None:
        self.observe_prefix(self.capture(payload), input_tokens)

    def observe_prefix(self, prefix: PayloadPrefix | None, input_tokens: Any) -> None:
        if type(input_tokens) is int and input_tokens > 0 and prefix is not None:
            self._anchor = _UsageAnchor(*prefix, input_tokens)

    def observe_reasoning(self, encrypted: str | None, usage: Any) -> None:
        """Associate an unchanged encrypted block with the real response usage."""
        if not encrypted or not isinstance(usage, dict):
            return
        details = usage.get("output_token_details") or {}
        count = details.get("reasoning") if isinstance(details, dict) else None
        if type(count) is not int or count < 0:
            # Output total is a confirmed upper bound when reasoning detail is absent.
            count = usage.get("output_tokens")
        if type(count) is not int or count < 0:
            return
        key = hashlib.sha256(encrypted.encode()).hexdigest()
        self._reasoning_costs[key] = count
        self._reasoning_costs.move_to_end(key)
        while len(self._reasoning_costs) > 1024:
            self._reasoning_costs.popitem(last=False)

    def diagnostics(self, payload: dict[str, Any]) -> dict[str, Any]:
        signature = self.capture(payload)
        anchor = self._anchor
        matched = bool(
            signature is not None
            and anchor is not None
            and signature[0] == anchor.settings
            and signature[1][: len(anchor.messages)] == anchor.messages
        )
        # Only appended messages need estimating for a calibrated prefix.
        start = len(anchor.messages) if matched and anchor is not None else 0
        unknown = sum(
            isinstance(m, dict)
            and bool(m.get("encrypted_content"))
            and hashlib.sha256(m["encrypted_content"].encode()).hexdigest()
            not in self._reasoning_costs
            for m in payload.get("messages", [])[start:]
            if isinstance(m, dict) and isinstance(m.get("encrypted_content"), str)
        )
        return {
            "mode": "provider_prefix" if matched else "text_fallback",
            "fallback_reason": (
                None
                if matched
                else "unsupported_payload"
                if signature is None
                else "no_usage"
                if anchor is None
                else "prefix_or_settings_changed"
            ),
            "unknown_reasoning_blocks": unknown,
        }

    def estimate(self, payload: dict[str, Any]) -> int:
        signature = self.capture(payload)
        anchor = self._anchor
        if (
            signature is not None
            and anchor is not None
            and signature[0] == anchor.settings
            and signature[1][: len(anchor.messages)] == anchor.messages
        ):
            # Estimate only the new suffix. Repricing an old encrypted block must
            # never subtract from the provider-confirmed prefix's input usage.
            suffix = payload["messages"][len(anchor.messages) :]
            return (
                anchor.input_tokens
                + sum(_message_budget(message, self._reasoning_costs) for message in suffix)
                + 256
            )
        return payload_text_budget(payload, reasoning_costs=self._reasoning_costs)


def model_text_budget(
    messages: Iterable[BaseMessage],
    *,
    model: Any,
    tools: list[Any] | None = None,
    tokens_per_image: int = 1024,
    **settings: Any,
) -> int:
    messages = list(messages)
    preview = getattr(model, "preview_payload", None)
    if callable(preview):
        if tools:
            settings = model.bind_tools(tools, **settings).kwargs
        payload = preview(messages, **settings)
        estimate = getattr(model, "estimate_input_tokens", None)
        return (
            estimate(payload)
            if callable(estimate)
            else payload_text_budget(payload, tokens_per_image=tokens_per_image)
        )
    return serialized_text_budget(
        messages, tools=tools, tokens_per_image=tokens_per_image, **settings
    )
