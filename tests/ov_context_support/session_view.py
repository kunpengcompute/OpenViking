# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native OV message conversion helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from openviking.integrations._message_content import MessageContent, decode_tool_content

# The native service omits its overview when the supplied budget is too small.
# Fetch complete material within the transport's byte cap; enforce the model
# budget on the assembled request instead of silently losing the overview.
WORKING_VIEW_BUDGET = 2_147_483_647


def history_ids(view: Mapping[str, Any]) -> frozenset[str]:
    records = view.get("messages")
    if not isinstance(records, list):
        raise ValueError("OV context has no message list")
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    if len(ids) != len(records) or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("OV context messages need native IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("OV context has duplicate native IDs")
    stats = view.get("stats") or {}
    if stats.get("failedArchives", 0):
        raise ValueError("OV context has unreadable archives")
    if stats.get("totalArchives", 0) and not view.get("latest_archive_overview"):
        raise ValueError("OV context is missing its archive overview")
    return frozenset(cast(str, key) for key in ids)


def _native_messages(record: dict[str, Any]) -> list[BaseMessage]:
    key = "ov:message:" + record["id"]
    role = record.get("role")
    parts = record.get("parts")
    if parts is None and isinstance(record.get("content"), str):
        parts = [{"type": "text", "text": record["content"]}]
    if role not in ("user", "assistant") or not isinstance(parts, list):
        raise ValueError("Unsupported OV message shape")
    content: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    results: list[ToolMessage] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Unsupported OV content part")
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image_url":
            content.append(deepcopy(part))
        elif kind == "context":
            content.append({"type": "text", "text": str(part.get("abstract", ""))})
        elif kind == "tool":
            call_id, name = part.get("tool_id"), part.get("tool_name")
            if not call_id or not name:
                raise ValueError("OV tool part has no call ID or name")
            if role == "assistant":
                args = part.get("tool_input") or {}
                if not isinstance(args, dict):
                    raise ValueError("OV tool input must be an object")
                calls.append({"id": call_id, "name": name, "args": deepcopy(args)})
            else:
                if part.get("tool_status") not in ("completed", "error"):
                    raise ValueError("OV tool result is unfinished")
                results.append(
                    ToolMessage(
                        id=key + ":tool:" + call_id,
                        tool_call_id=call_id,
                        name=name,
                        content=decode_tool_content(part),
                        status="error" if part.get("tool_status") == "error" else "success",
                        additional_kwargs={"ov_context_history": True, "ov_context_mutable": True},
                    )
                )
        else:
            raise ValueError("Unsupported OV content part")
    if results:
        if content:
            raise ValueError("Mixed OV user text and tool results require an explicit adapter")
        return list(results)
    # Keep a plain string for text-only messages; preserve multimodal blocks.
    body: MessageContent = (
        "\n".join(part["text"] for part in content)
        if all(part.get("type") == "text" for part in content)
        else list(content)
    )
    metadata = {"ov_context_history": True}
    if role == "assistant":
        return [AIMessage(id=key, content=body, tool_calls=calls, additional_kwargs=metadata)]
    return [HumanMessage(id=key, content=body, additional_kwargs=metadata)]
