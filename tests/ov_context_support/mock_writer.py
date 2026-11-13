# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Test-only writer for the offline fake OV server; not a runtime implementation."""

from copy import deepcopy
from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    message_to_dict,
)

from openviking.integrations._message_content import encode_tool_content, normalize_image_block
from openviking.integrations.langchain.ov_context.assembler import INJECTED
from openviking.integrations.langchain.ov_context.errors import AmbiguousWriteError
from openviking.integrations.langchain.ov_context.writer import WriteResult


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") in {"image", "image_url"}:
            parts.append(normalize_image_block(block))
        elif block.get("type") in ("reasoning", "thinking"):
            # Provider-private reasoning is not a user-visible message or memory source.
            continue
        else:
            raise ValueError(
                "Unsupported message block; configure an explicit host content adapter"
            )
    return parts


class MockMessageWriter:
    def __init__(self, client, peer_id=None, metadata=None):
        self.client, self.peer_id = client, peer_id
        self.metadata = metadata if metadata is not None else {}

    async def write(
        self, session_id, raw_messages, write_index, run_id, *, identity=None, peer_id=None
    ):
        peer_id = peer_id or self.peer_id
        end = len(raw_messages)
        if not 0 <= write_index <= end:
            raise ValueError("Raw message cursor regressed; restore the original graph history")
        if write_index == end:
            return WriteResult([], end, 0)
        calls = {
            call["id"]: call
            for message in raw_messages[:end]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        specs, model_messages, groups, native_groups = [], [], [], []
        delta = list(raw_messages[write_index:end])
        index = 0
        while index < len(delta):
            message = delta[index]
            if isinstance(message, AIMessageChunk):
                raise ValueError("Write only finalized assistant messages")
            if message.additional_kwargs.get(INJECTED):
                raise ValueError("Injected model views must not enter original graph messages")
            if not isinstance(message, (HumanMessage, AIMessage, ToolMessage)):
                index += 1
                continue
            if not message.id:
                raise ValueError("Messages require stable source IDs")
            group = [message]
            role, parts = "user", []
            if isinstance(message, ToolMessage):
                while index + len(group) < len(delta) and isinstance(
                    delta[index + len(group)], ToolMessage
                ):
                    group.append(delta[index + len(group)])
                for item in group:
                    if not item.id:
                        raise ValueError("Tool messages require stable source IDs")
                    call = calls.get(item.tool_call_id)
                    if call is None:
                        raise ValueError("Tool result has no original assistant call")
                    output, mime = encode_tool_content(item.content)
                    part = {
                        "type": "tool",
                        "tool_id": item.tool_call_id,
                        "tool_name": call["name"],
                        "tool_input": call["args"],
                        "tool_output": output,
                        "tool_status": "error" if item.status == "error" else "completed",
                    }
                    if mime != "text/plain":
                        part["tool_output_mime_type"] = mime
                    parts.append(part)
            else:
                parts = _parts(message.content)
                if not parts and not getattr(message, "tool_calls", None):
                    parts = [{"type": "text", "text": ""}]
                if isinstance(message, AIMessage):
                    role = "assistant"
                    parts.extend(
                        {
                            "type": "tool",
                            "tool_id": call["id"],
                            "tool_name": call["name"],
                            "tool_input": call["args"],
                            "tool_status": "pending",
                        }
                        for call in message.tool_calls
                    )
            if parts:
                payload = {"role": role, "parts": parts}
                if role == "assistant" or isinstance(message, ToolMessage):
                    if peer_id:
                        payload["peer_id"] = peer_id
                specs.append(payload)
                groups.append([item.id for item in group])
                native_groups.append([message_to_dict(item) for item in group])
            model_messages.extend(group)
            index += len(group)
        for offset in range(0, len(specs), 100):
            batch = specs[offset : offset + 100]
            try:
                result = await self.client.call(
                    "batch_add_messages", identity=identity, session_id=session_id, messages=batch
                )
                # Test-only OV persistence stores full serialized native messages.
                # It is independent of Middleware's local cache; production I/O
                # and its storage representation are supplied by the integrator.
                start = result["message_count"] - len(batch)
                for index, ids in enumerate(groups[offset : offset + 100]):
                    record_id = f"native-{start + index}"
                    self.metadata[record_id] = ids
                    self.metadata.setdefault("_native_payloads", {})[record_id] = deepcopy(
                        native_groups[offset + index]
                    )
            except httpx.TransportError as exc:
                raise AmbiguousWriteError("Batch outcome unknown; no automatic retry") from exc
        return WriteResult(model_messages, end, len(model_messages))
