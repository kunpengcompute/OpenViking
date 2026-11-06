# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared Reader/assembler contracts without importing framework orchestration."""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class ContextSnapshot:
    """Model view and its applied terminal archive boundary, including done/failed."""

    messages: list[BaseMessage] = field(default_factory=list)
    # A failed terminal boundary may retain an earlier successful overview.
    archive_id: str | None = None


class ContextReadError(RuntimeError):
    """No verified snapshot was produced; the caller must retain its old cache."""


class ContextHistoryBudgetError(ContextReadError):
    """The complete session summary cannot fit the supplied history budget."""
