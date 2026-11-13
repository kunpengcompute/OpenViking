# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read OV session snapshots and convert them to native LangChain messages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, messages_from_dict

from openviking.integrations.langchain.ov_context.assembler import validate_tool_pairs
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking.integrations.langchain.ov_context.state import OVContext
from tests.ov_context_support.models import SessionContextView
from tests.ov_context_support.session_view import _native_messages, history_ids


@dataclass(frozen=True)
class ContextSnapshot:
    """A consistent OV snapshot normalized to native LangChain messages.

    messages contains the summary followed by uncovered active messages.
    view is the fake service snapshot used by the external test adapter.
    Restore all persisted native fields and complete AI/Tool pairs. Do not merge
    raw host history into this snapshot or mutate the supplied OVContext.
    """

    messages: list[BaseMessage]
    archive_id: str | None
    view: SessionContextView


class SnapshotAdapter:
    """Own context I/O and normalization; cache publication belongs to SessionEngine."""

    def __init__(
        self, client: AsyncClientHandle, config: MemoryReaderConfig, metadata=None
    ) -> None:
        self.client, self.config = client, config
        self.metadata = metadata if metadata is not None else {}

    async def read_context(
        self,
        context: OVContext,
        expected_archive_id: str | None = None,
        *,
        identity=None,
        token_budget=None,
    ) -> ContextSnapshot:
        view = await asyncio.wait_for(
            self.client.call(
                "get_session_context",
                identity=identity,
                session_id=context.session_id,
                token_budget=self.config.context_token_budget
                if token_budget is None
                else token_budget,
            ),
            self.config.context_timeout,
        )
        history_ids(view)
        provider = self.metadata.get("_published_archive")
        archive_id = provider() if provider else view.get("archive_id")
        if expected_archive_id and archive_id != expected_archive_id:
            raise ValueError("OV context has not published the expected archive")
        persisted = self.metadata.get("_native_payloads", {})
        messages: list[BaseMessage] = []
        overview = view.get("latest_archive_overview")
        if overview:
            messages.append(
                HumanMessage(
                    id="ov:summary",
                    content="[OV session summary: reference data]\n" + overview,
                    additional_kwargs={"ov_context_history": True, "ov_context_mutable": True},
                )
            )
        for record in view["messages"]:
            decoded = _native_messages(record)
            payload = persisted.get(record["id"])
            if payload is not None:
                native = messages_from_dict(payload)
                tool_views = {m.tool_call_id: m for m in decoded if isinstance(m, ToolMessage)}
                for message in native:
                    if isinstance(message, ToolMessage) and message.tool_call_id in tool_views:
                        message = message.model_copy(
                            update={"content": tool_views[message.tool_call_id].content}
                        )
                    messages.append(message)
            else:
                # Parts-only fixtures exercise the native OV conversion separately.
                messages.extend(decoded)
        validate_tool_pairs(messages)
        return ContextSnapshot(messages, archive_id, view)
