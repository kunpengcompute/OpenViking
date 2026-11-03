# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Apply batch tool views without changing the original graph messages."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, ToolMessage

BACKTRACKING_TOOLS = (
    "openviking_tool_result_read",
    "openviking_tool_result_search_many",
    "openviking_tool_result_list",
)

logger = logging.getLogger(__name__)


def apply_tool_views(
    model_messages: list[BaseMessage],
    result: Mapping[str, Any],
    positions: Mapping[tuple[int, int], int],
    *,
    session_id: str,
    protected_indices: set[int],
) -> None:
    """Positions use request indices, including when OV splits a tool group.

    Invalid optional views are ignored after a successful write: failing the
    write here would invite a retry of messages already persisted by OV.
    """
    views = result.get("tool_views", [])
    if not isinstance(views, list):
        logger.warning("Ignoring invalid OV tool_views array")
        return
    applied: set[tuple[int, int]] = set()
    for view in views:
        if not isinstance(view, dict):
            logger.warning("Ignoring invalid OV tool view")
            continue
        message_index, part_index = view.get("message_index"), view.get("part_index")
        if type(message_index) is not int or type(part_index) is not int:
            logger.warning("Ignoring OV tool view with invalid request indices")
            continue
        key = (message_index, part_index)
        index = positions.get(key)
        if index is None or key in applied:
            logger.warning("Ignoring OV tool view with unmatched or repeated request position")
            continue
        message = model_messages[index]
        output, ref = view.get("tool_output"), view.get("tool_output_ref")
        if (
            not isinstance(message, ToolMessage)
            or view.get("tool_id") != message.tool_call_id
            or not isinstance(output, str)
            or not isinstance(ref, str)
            or not ref.startswith("viking://")
            or f"/sessions/{session_id}/tool-results/" not in ref
        ):
            logger.warning("Ignoring OV tool view with invalid call ID, output or reference")
            continue
        applied.add(key)
        if index not in protected_indices and output != message.content:
            model_messages[index] = message.model_copy(update={"content": output})
