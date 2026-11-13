# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Batch receipts for native tool externalization; no framework dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openviking.message.part import ToolPart

if TYPE_CHECKING:
    from openviking.session.session import Session


def add_messages_with_tool_views(session: Session, specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep request positions before native aggregation splits tool results.

    Session.add_messages updates the supplied Part objects in place, then
    persists them. Publish only changed outputs with a usable stored reference
    after it returns successfully. Preserve the original batch count fields.
    """
    originals = _original_outputs(specs)
    return _receipt(session, session.add_messages(specs), originals)


async def add_messages_with_tool_views_async(
    session: Session, specs: list[dict[str, Any]]
) -> dict[str, Any]:
    originals = _original_outputs(specs)
    added = await session.add_messages_async(specs)
    return _receipt(session, added, originals)


def _original_outputs(specs):
    return [
        (message_index, part_index, part, part.tool_output)
        for message_index, spec in enumerate(specs)
        if spec["role"] == "user"
        for part_index, part in enumerate(spec["parts"])
        if isinstance(part, ToolPart) and part.tool_status in ("completed", "error")
    ]


def _receipt(session, added, originals):
    return {
        "session_id": session.session_id,
        "message_count": len(session.messages),
        "added": len(added),
        "pending_tokens": int(session.meta.pending_tokens),
        "tool_views": [
            {
                "message_index": message_index,
                "part_index": part_index,
                "tool_id": part.tool_id,
                "tool_output": part.tool_output,
                "tool_output_ref": part.tool_output_ref,
            }
            for message_index, part_index, part, output in originals
            if part.tool_output != output and part.tool_output_ref
        ],
    }
