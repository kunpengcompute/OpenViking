# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Validated OVContext execution configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class OVContextConfig:
    """Model assembly, commit and recovery settings; Reader owns retrieval config."""

    context_window: int | None = None
    max_input_tokens: int | None = None
    output_reserve: int = 4096
    safety_margin: int = 512
    context_failure: str = "raise"
    base_context_ratio: float = 0.3
    capacity_ratio: float = 0.9
    commit_wait_timeout: float = 300.0
    poll_interval: float = 0.1
    retry_attempts: int = 0
    tool_results: bool = True
    summary_check_interval: int = 2
    poll_max_interval: float = 2.0

    def __post_init__(self) -> None:
        if type(self.summary_check_interval) is not int or self.summary_check_interval < 1:
            raise ValueError("summary_check_interval must be a positive integer")
        if self.context_failure not in ("raise", "host"):
            raise ValueError("Invalid context failure policy")
        for name in ("base_context_ratio", "capacity_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.base_context_ratio < self.capacity_ratio < 1:
            raise ValueError("Require 0 < base_context_ratio < capacity_ratio < 1")
        for name in ("commit_wait_timeout", "poll_interval", "poll_max_interval"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        for value in (self.context_window, self.max_input_tokens):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Model limits must be positive integers")
        for name in ("output_reserve", "safety_margin"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.retry_attempts, bool)
            or not isinstance(self.retry_attempts, int)
            or self.retry_attempts < 0
        ):
            raise ValueError("retry_attempts must be a non-negative integer")
        if (
            self.context_window is not None
            and self.context_window <= self.output_reserve + self.safety_margin
        ):
            raise ValueError("Context window cannot fit the reserved budget")
