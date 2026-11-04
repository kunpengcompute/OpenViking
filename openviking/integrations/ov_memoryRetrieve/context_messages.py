# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Restore OV Parts and explicit one-to-one user source identities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from openviking.integrations._message_content import (
    comparable_content as comparable_content,
)
from openviking.integrations._message_content import (
    decode_tool_content,
)

from .types import ContextReadError


def restore_messages(records: list[dict[str, Any]]) -> list[BaseMessage]:
    """Handle both user-role tool results and legacy assistant-role tool results."""
    messages: list[BaseMessage] = []
    for record in records:
        role = record.get("role")
        if role not in {"user", "assistant"}:
            raise ContextReadError(f"Unsupported OV message role: {role!r}")
        ov_id = record.get("id")
        # OV record IDs remain distinct from the caller's source message IDs.
        message_id = ov_id
        origin = {"openviking_message_id": ov_id}
        message_kind = record.get("message_kind")
        source_ids = record.get("source_message_ids")
        if message_kind is not None:
            origin["openviking_message_kind"] = message_kind
        if source_ids is not None:
            if not isinstance(source_ids, list) or any(
                not isinstance(key, str) or not key.strip() for key in source_ids
            ):
                raise ContextReadError("Invalid source_message_ids in OV message")
            origin["openviking_source_message_ids"] = list(source_ids)
        blocks: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        parts = record.get("parts")
        if parts is None:
            parts = [{"type": "text", "text": record.get("content", "")}]
        for part in parts:
            kind = part.get("type", "text")
            if kind == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_url":
                blocks.append({"type": "image_url", "image_url": deepcopy(part["image_url"])})
            elif kind == "context":
                blocks.append({"type": "text", "text": part.get("abstract", "")})
            elif kind == "tool":
                if not part.get("tool_id"):
                    raise ContextReadError("Tool Part is missing its tool_id")
                status = part.get("tool_status", "pending")
                if status in {"completed", "error"}:
                    results.append(part)
                elif role == "assistant" and status in {"pending", "running"}:
                    if not part.get("tool_name"):
                        raise ContextReadError("Assistant tool call is missing its tool_name")
                    calls.append(
                        {
                            "id": part["tool_id"],
                            "name": part["tool_name"],
                            "args": deepcopy(part.get("tool_input") or {}),
                        }
                    )
                else:
                    raise ContextReadError(f"Unsupported tool status {status!r} for {role}")
            else:
                raise ContextReadError(f"Unsupported OV Part type: {kind!r}")
        content: str | list[str | dict[str, Any]] = list(blocks)
        if all(block["type"] == "text" for block in blocks):
            content = "".join(block["text"] for block in blocks)
        if role == "assistant" and (blocks or calls or not results):
            messages.append(
                AIMessage(
                    id=message_id, content=content, tool_calls=calls, additional_kwargs=origin
                )
            )
        elif role == "user" and (blocks or not results):
            user_id = message_id
            if message_kind == "user_query" and source_ids and len(source_ids) == 1:
                user_id = source_ids[0]
            messages.append(HumanMessage(id=user_id, content=content, additional_kwargs=origin))
        for part in results:
            try:
                tool_content = decode_tool_content(part)
            except ValueError as exc:
                raise ContextReadError("Invalid structured tool content") from exc
            result_id = message_id
            if blocks or calls or len(results) > 1:
                result_id = f"{message_id or 'ov_tool'}:{part['tool_id']}"
            messages.append(
                ToolMessage(
                    id=result_id,
                    content=tool_content,
                    tool_call_id=part["tool_id"],
                    name=part.get("tool_name") or None,
                    status="error" if part.get("tool_status") == "error" else "success",
                    additional_kwargs=origin,
                )
            )
    return messages
