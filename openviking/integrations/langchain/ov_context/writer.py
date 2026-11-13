# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Incremental OV message writing contracts and implementation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, cast

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from openviking.integrations._message_content import (
    MessageContent,
    encode_tool_content,
    has_image_blocks,
    normalize_image_block,
)
from openviking_cli.client.identity import OVIdentity

from .assembler import is_user_anchor
from .errors import AmbiguousWriteError, IncompleteWriteError
from .tool_views import BACKTRACKING_TOOLS, apply_tool_views


@dataclass(frozen=True)
class WriteResult:
    """Returned only after every message in the requested delta is persisted.

    model_messages keeps native LangChain types and metadata; a tool's content
    may be replaced by a short view supplied by the external implementation.
    Image-bearing tool results retain the complete native content-block list.
    write_index is the next index in the original graph list, not an OV count.
    """

    model_messages: list[BaseMessage]
    write_index: int
    written_count: int
    pending_tokens: int | None = None


class MessageWriterBase(Protocol):
    """Convert and synchronously persist a graph delta; implemented by the host.

    Borrow the host-provided async client; the host owns its lifetime.
    Write OV content and tool fields, then return the local native delta.
    A compressed tool result copies the original message and changes only content.
    Do not mutate raw_messages. Preserve native metadata and tool-call pairing.
    On failure or an unknown write outcome, raise instead of advancing the
    cursor. Retries and OV payload conversion belong to the implementation.

    Tool-cycle batching additionally requires an optional synchronous
    ``validate(messages, *, identity, peer_id) -> None`` method. It must reuse
    the writer's local encoding/serialization checks, raise on invalid input,
    and perform no I/O or mutation. This is a capability, not a required
    Protocol member: writers without it keep immediate writes. Validation
    cannot guarantee later server acceptance and must run again during write.
    """

    async def write(
        self,
        session_id: str,
        raw_messages: list[BaseMessage],
        write_index: int,
        run_id: str,
        *,
        identity: OVIdentity | None = None,
        peer_id: str | None = None,
    ) -> WriteResult: ...


def ov_record_ranges(messages: Sequence[BaseMessage], start: int = 0) -> list[tuple[int, int]]:
    """Batch payload ranges; OV persists each aggregated tool result separately."""
    groups = []
    index = start
    while index < len(messages):
        message = messages[index]
        if isinstance(message, SystemMessage) and not isinstance(message, BaseMessageChunk):
            index += 1
            continue
        end = index + 1
        if isinstance(message, ToolMessage):
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                end += 1
        groups.append((index, end))
        index = end
    return groups


@dataclass(frozen=True)
class WriterConfig:
    batch_size: int = 100
    peer_id: str | None = None
    backtracking_tools: tuple[str, ...] = BACKTRACKING_TOOLS

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")


def _content_parts(
    content: MessageContent, *, tool_calls: Sequence[ToolCall] = ()
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    parts = []
    calls = {call["id"]: call for call in tool_calls}
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") in ("image_url", "image"):
            parts.append(normalize_image_block(block))
        elif block.get("type") in ("tool_call", "tool_use"):
            # Native providers may repeat parsed tool_calls in content. Emit
            # ToolParts once from AIMessage.tool_calls, after verifying the block.
            call = calls.get(block.get("id"))
            args = block.get("input") if block["type"] == "tool_use" else block.get("args")
            if call is None or block.get("name") != call["name"] or args != call["args"]:
                raise ValueError("Tool content block does not match parsed assistant tool_calls")
        elif block.get("type") in ("reasoning", "thinking"):
            continue
        else:
            raise ValueError("Unsupported message content block")
    return parts


class BatchMessageClient(Protocol):
    """The borrowed client's existing batch write entry point."""

    async def batch_add_messages(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class IdentityBatchMessageClient(Protocol):
    """Batch transport receiving the caller identity explicitly for each request."""

    async def batch_add_messages(
        self, session_id: str, messages: list[dict[str, Any]], *, identity: OVIdentity
    ) -> dict[str, Any]: ...


@dataclass
class _PreparedWrite:
    end: int
    specs: list[dict[str, Any]]
    model_messages: list[BaseMessage]
    positions: dict[tuple[int, int], int]
    protected: set[int]
    spec_ends: list[int]
    model_ends: list[int]


class OVMessageWriter(MessageWriterBase):
    """Borrow an initialized async OV client; keep no per-session progress.

    DeepAgent serializes writes for a session and checkpoints WriteResult's
    cursor with the returned messages. Client initialization/closing, cache
    updates, recall and commit remain DeepAgent's responsibility.
    """

    def __init__(self, client: BatchMessageClient, config: WriterConfig | None = None) -> None:
        self.client = client
        self.config = config or WriterConfig()

    def validate(
        self,
        messages: Sequence[BaseMessage],
        *,
        identity: OVIdentity | None = None,
        peer_id: str | None = None,
    ) -> None:
        """Check local encoding without I/O or changing messages and progress."""
        self._prepare(messages, 0, identity=identity, peer_id=peer_id)

    def _prepare(
        self,
        raw_messages: Sequence[BaseMessage],
        write_index: int,
        *,
        identity: OVIdentity | None = None,
        peer_id: str | None = None,
    ) -> _PreparedWrite:
        if identity is not None:
            if not isinstance(identity, OVIdentity):
                raise TypeError("identity must be an OVIdentity")
            identity.__post_init__()
        end = len(raw_messages)
        if type(write_index) is not int or not 0 <= write_index <= end:
            raise ValueError("Invalid raw message cursor; restore the original graph history")
        if write_index == end:
            return _PreparedWrite(end, [], [], {}, set(), [], [])
        snapshot = list(raw_messages[:end])
        calls = {
            call["id"]: call
            for message in snapshot
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        specs: list[dict[str, Any]] = []
        model_messages: list[BaseMessage] = []
        positions: dict[tuple[int, int], int] = {}
        protected: set[int] = set()
        spec_ends: list[int] = []
        model_ends: list[int] = []
        peer_id = peer_id if peer_id is not None else self.config.peer_id
        if peer_id is None and identity is not None:
            peer_id = identity.actor_peer_id
        # Convert the whole delta before the first write, so unsupported input
        # cannot leave an earlier batch persisted.
        for index, group_end in ov_record_ranges(snapshot, write_index):
            message = snapshot[index]
            if isinstance(message, BaseMessageChunk):
                raise ValueError("Write only finalized messages, not streaming chunks")
            if (
                message.additional_kwargs.get("ov_context_injected")
                or message.additional_kwargs.get("ov_context_history")
                or is_user_anchor(message)
            ):
                raise ValueError("Write original graph messages, not injected or restored views")
            role = "user"
            parts: list[dict[str, Any]] = []
            if isinstance(message, ToolMessage):
                while index < group_end:
                    item = snapshot[index]
                    assert isinstance(item, ToolMessage)
                    if (
                        isinstance(item, BaseMessageChunk)
                        or item.additional_kwargs.get("ov_context_injected")
                        or item.additional_kwargs.get("ov_context_history")
                    ):
                        raise ValueError("Write only original finalized tool messages")
                    call = calls.get(item.tool_call_id)
                    if call is None:
                        raise ValueError("Tool result has no matching assistant call")
                    output, mime = encode_tool_content(item.content)
                    positions[(len(specs), len(parts))] = len(model_messages)
                    if call["name"] in self.config.backtracking_tools or has_image_blocks(
                        item.content
                    ):
                        protected.add(len(model_messages))
                    parts.append(
                        {
                            "type": "tool",
                            "tool_id": item.tool_call_id,
                            "tool_name": call["name"],
                            "tool_input": deepcopy(call["args"]),
                            "tool_output": output,
                            "tool_output_mime_type": mime,
                            "tool_status": "error" if item.status == "error" else "completed",
                        }
                    )
                    model_messages.append(item)
                    index += 1
            elif isinstance(message, (HumanMessage, AIMessage)):
                parts = _content_parts(
                    message.content,
                    tool_calls=message.tool_calls if isinstance(message, AIMessage) else (),
                )
                if isinstance(message, AIMessage):
                    role = "assistant"
                    parts.extend(
                        {
                            "type": "tool",
                            "tool_id": call["id"],
                            "tool_name": call["name"],
                            "tool_input": deepcopy(call["args"]),
                            "tool_status": "pending",
                        }
                        for call in message.tool_calls
                    )
                if not parts:
                    parts = [{"type": "text", "text": ""}]
                model_messages.append(message)
                index += 1
            else:
                raise ValueError(f"Unsupported original message type: {message.type}")
            spec: dict[str, Any] = {"role": role, "parts": parts}
            if isinstance(message, HumanMessage):
                spec["message_kind"] = "user_query"
                if message.id:
                    spec["source_message_ids"] = [message.id]
            if peer_id is not None and (role == "assistant" or isinstance(message, ToolMessage)):
                spec["peer_id"] = peer_id
            specs.append(spec)
            spec_ends.append(index)
            model_ends.append(len(model_messages))

        # Validate every outgoing spec before any batch is persisted. User source
        # IDs cross the boundary; other native metadata remains local.
        json.dumps({"messages": specs}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return _PreparedWrite(
            end, specs, model_messages, positions, protected, spec_ends, model_ends
        )

    async def write(
        self,
        session_id: str,
        raw_messages: Sequence[BaseMessage],
        write_index: int,
        run_id: str = "session",
        *,
        identity: OVIdentity | None = None,
        peer_id: str | None = None,
    ) -> WriteResult:
        prepared = self._prepare(raw_messages, write_index, identity=identity, peer_id=peer_id)
        end, specs, model_messages = prepared.end, prepared.specs, prepared.model_messages
        positions, protected = prepared.positions, prepared.protected
        spec_ends, model_ends = prepared.spec_ends, prepared.model_ends

        confirmed_index, confirmed_models = write_index, 0
        pending_tokens = None
        for offset in range(0, len(specs), self.config.batch_size):
            batch = specs[offset : offset + self.config.batch_size]
            try:
                if identity is None:
                    result = await self.client.batch_add_messages(
                        session_id=session_id, messages=batch
                    )
                else:
                    result = await cast(IdentityBatchMessageClient, self.client).batch_add_messages(
                        session_id=session_id, messages=batch, identity=identity
                    )
                if (
                    not isinstance(result, dict)
                    or result.get("session_id") != session_id
                    or type(result.get("added")) is not int
                    or result["added"] < len(batch)
                    or type(result.get("message_count")) is not int
                    or result["message_count"] < result["added"]
                ):
                    raise AmbiguousWriteError("OV did not confirm the complete batch")
            except Exception as exc:
                if confirmed_models or isinstance(exc, (httpx.TransportError, AmbiguousWriteError)):
                    raise IncompleteWriteError(
                        WriteResult(
                            model_messages[:confirmed_models],
                            confirmed_index,
                            confirmed_models,
                            pending_tokens,
                        )
                    ) from exc
                raise
            # Missing/invalid receipts from older servers do not invalidate a
            # successful write. Never retain an earlier batch's token count.
            receipt = result.get("pending_tokens")
            pending_tokens = receipt if type(receipt) is int and receipt >= 0 else None
            batch_positions = {
                (spec_index - offset, part_index): model_index
                for (spec_index, part_index), model_index in positions.items()
                if offset <= spec_index < offset + len(batch)
            }
            apply_tool_views(
                model_messages,
                result,
                batch_positions,
                session_id=session_id,
                protected_indices=protected,
            )
            confirmed_index = spec_ends[offset + len(batch) - 1]
            confirmed_models = model_ends[offset + len(batch) - 1]
        return WriteResult(model_messages, end, len(model_messages), pending_tokens)
