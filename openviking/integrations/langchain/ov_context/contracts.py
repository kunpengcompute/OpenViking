# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared snapshot boundary between the assembler and a context reader."""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class ContextSnapshot:
    messages: list[BaseMessage] = field(default_factory=list)
    archive_id: str | None = None
