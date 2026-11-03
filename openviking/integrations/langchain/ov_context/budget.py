# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model input budgets and approximate request token counting."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any, Callable

from langchain.agents.middleware.types import ModelRequest

from openviking.utils.token_estimation import estimate_serialized_tokens

from .context_guide import BATCH_SEARCH_GUIDANCE as BATCH_SEARCH_GUIDANCE
from .context_guide import GUIDE as GUIDE
from .context_guide import RULES as RULES
from .token_estimation import model_text_budget

TokenCounter = Callable[[ModelRequest[Any]], int]


@dataclass(frozen=True)
class BudgetSpec:
    context_window: int
    output_reserve: int
    safety_margin: int = 512
    max_input_tokens: int | None = None
    base_context_ratio: float = 0.3
    capacity_ratio: float = 0.9

    def __post_init__(self) -> None:
        values = (self.context_window, self.output_reserve, self.safety_margin)
        if any(type(value) is not int for value in values) or (
            self.max_input_tokens is not None and type(self.max_input_tokens) is not int
        ):
            raise ValueError("Budget values must be integers")
        if self.output_reserve < 0 or self.safety_margin < 0 or self.input_limit <= 0:
            raise ValueError("Invalid model budget")
        if not all(isfinite(v) for v in (self.base_context_ratio, self.capacity_ratio)) or not (
            0 < self.base_context_ratio < self.capacity_ratio < 1
        ):
            raise ValueError("Require 0 < base_context_ratio < capacity_ratio < 1")
        if not 0 < self.base_context_limit < self.capacity_limit < self.input_limit:
            raise ValueError("Require base_context_limit < capacity_limit < input_limit")

    @property
    def input_limit(self) -> int:
        available = self.context_window - self.output_reserve - self.safety_margin
        return (
            min(available, self.max_input_tokens)
            if self.max_input_tokens is not None
            else available
        )

    @property
    def base_context_limit(self) -> int:
        return floor(self.context_window * self.base_context_ratio)

    @property
    def capacity_limit(self) -> int:
        return floor(self.context_window * self.capacity_ratio)


class ApproximateTokenCounter:
    """One complete-input estimate for Middleware, summaries and model admission.

    Models exposing provider payload/usage counters use those exact same inputs.
    Generic models use the shared text fallback. Raw state usage is never trusted.
    """

    name = "request_tokens_v2"

    def __init__(self, *, tokens_per_image: int = 1024) -> None:
        self.tokens_per_image = tokens_per_image

    def __call__(self, request: ModelRequest[Any]) -> int:
        messages = ([request.system_message] if request.system_message else []) + request.messages
        settings = dict(request.model_settings)
        if request.tools:
            settings["tool_choice"] = request.tool_choice
        count = model_text_budget(
            messages,
            model=request.model,
            tools=request.tools,
            tokens_per_image=self.tokens_per_image,
            **settings,
        )
        # Structured-output strategies resolve later in create_agent. Reserve the
        # schema here; the final provider payload is checked with the same formula.
        schema = self._response_schema(request.response_format)
        return count + estimate_serialized_tokens({"response_format": schema} if schema else None)

    def diagnostics(self, request: ModelRequest[Any]) -> dict[str, Any]:
        describe = getattr(request.model, "input_token_diagnostics", None)
        preview = getattr(request.model, "preview_payload", None)
        if not callable(describe) or not callable(preview):
            return {"mode": "text_fallback", "fallback_reason": "no_provider_counter"}
        settings = dict(request.model_settings)
        if request.tools:
            settings["tool_choice"] = request.tool_choice
            settings = request.model.bind_tools(request.tools, **settings).kwargs
        messages = ([request.system_message] if request.system_message else []) + request.messages
        return describe(preview(messages, **settings))

    @staticmethod
    def _response_schema(value: Any) -> Any:
        value = getattr(value, "schema", value) if not isinstance(value, type) else value
        if hasattr(value, "model_json_schema"):
            return value.model_json_schema()
        return value


@dataclass(frozen=True)
class HistoryBudget:
    budget: BudgetSpec
    fixed_tokens: int
    history_target_tokens: int


__all__ = ["TokenCounter", "BudgetSpec", "HistoryBudget", "ApproximateTokenCounter"]
