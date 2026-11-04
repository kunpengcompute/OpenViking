# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader snapshots join the existing Writer, native Session, State and Assembler."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from openviking.integrations.langchain.ov_context import (
    ContextAssembler,
    ContextBudgetExceeded,
    ContextHistoryBudgetError,
    ContextReadError,
    ContextSnapshot,
    OVContext,
    OVContextConfig,
    OVMessageWriter,
)
from openviking.integrations.ov_memoryRetrieve import (
    AsyncMemoryClient,
    OVContextReader,
)
from openviking.message import Message, TextPart
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import ResourceExhaustedError
from tests.ov_context_support.writer import SessionBatchClient, make_session


class SnapshotClient(SessionBatchClient):
    async def get_session_context(self, session_id, token_budget=128_000, *, identity=None):
        assert session_id == self.session.session_id
        return await self.session.get_session_context(token_budget=token_budget)


@pytest.fixture
def request_and_assembler():
    request = ModelRequest(
        model=FakeListChatModel(responses=["done"]),
        messages=[],
        system_message=SystemMessage(content="Host rules"),
        tools=[],
        model_settings={},
    )
    assembler = ContextAssembler(OVContextConfig(context_window=128_000))
    return request, assembler


@pytest.mark.parametrize("multimodal", [False, True])
async def test_persisted_source_ids_disambiguate_repeated_users_and_restore_parallel_tools(
    tmp_path, request_and_assembler, multimodal
):
    request, assembler = request_and_assembler
    session = make_session(tmp_path)
    client = SnapshotClient(session)
    identity = OVIdentity("account", "user", "peer")
    content = (
        [
            {"type": "text", "text": "同样的问题"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]
        if multimodal
        else "同样的问题"
    )
    user = HumanMessage(id="current-user", content=deepcopy(content))
    raw = [
        HumanMessage(id="previous-user", content=deepcopy(content)),
        AIMessage(id="previous-answer", content="Earlier answer"),
        user,
        AIMessage(
            id="calls",
            content="查询中",
            tool_calls=[
                {"id": "one", "name": "lookup", "args": {"key": 1}},
                {"id": "two", "name": "lookup", "args": {"key": 2}},
            ],
        ),
        ToolMessage(id="result-two", tool_call_id="two", content="not found", status="error"),
        ToolMessage(
            id="result-one",
            tool_call_id="one",
            content=[{"type": "text", "text": "找到记录"}],
        ),
    ]
    original = deepcopy(raw)
    result = await OVMessageWriter(client).write(
        session.session_id, raw, 0, "run", identity=identity
    )
    context = OVContext(
        session.session_id,
        messages=list(result.model_messages),
        write_index=result.write_index,
        completed_runs=3,
        owner=identity.session_key(session.session_id),
    )
    context.begin_run("run", user)
    context.inject_recall("已召回的参考", "success")
    before = deepcopy(context.to_state())
    reader = OVContextReader(client)
    snapshot = await reader.exchange_context(
        context.session_id,
        identity=context.identity,
        token_budget=assembler.history_budget(request).history_target_tokens,
    )
    humans = [m for m in snapshot.messages if isinstance(m, HumanMessage)]
    assert [m.id for m in humans] == ["previous-user", "current-user"]
    assert humans[-1].additional_kwargs["openviking_message_id"] != user.id
    assert humans[-1].additional_kwargs["openviking_source_message_ids"] == [user.id]

    candidate = assembler.refresh(context, snapshot)
    prepared = assembler.prepare(request, candidate)
    assert context.to_state() == before and raw == original
    assert candidate.write_index == len(raw) and candidate.completed_runs == 3
    assert candidate.recall_done and candidate.recall_status == "success"
    assert candidate.messages[2].id == "current-user"
    assert "已召回的参考" in str(candidate.messages[2].content)
    tools = [m for m in prepared.request.messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tools] == ["two", "one"]
    assert tools[0].status == "error"
    assert tools[1].content == [{"type": "text", "text": "找到记录"}]
    prepared.request.messages[2].content = "downstream edit"
    assert candidate.messages[2].content == context.messages[2].content
    candidate.end_run()
    assert candidate.messages[2] is user and user.content == content


async def test_failed_native_archive_boundary_keeps_older_summary_and_raw_tail(
    tmp_path, request_and_assembler
):
    request, assembler = request_and_assembler
    session = make_session(tmp_path)
    for index, marker, overview in [(1, ".done", "成功的概要"), (2, ".failed.json", None)]:
        uri = f"{session.uri}/history/archive_{index:03d}"
        message = Message(id=f"archived-{index}", role="user", parts=[TextPart(f"task {index}")])
        await session._viking_fs.write_file(f"{uri}/messages.jsonl", message.to_jsonl())
        await session._viking_fs.write_file(f"{uri}/{marker}", "{}")
        if overview:
            await session._viking_fs.write_file(f"{uri}/.overview.md", overview)
    client = SnapshotClient(session)
    user = HumanMessage(id="current", content="本轮问题")
    result = await OVMessageWriter(client).write(session.session_id, [user], 0)
    context = OVContext(session.session_id, messages=result.model_messages, write_index=1)
    context.begin_run("run", user)
    context.inject_recall("本轮召回", "success")
    snapshot = await OVContextReader(client).exchange_context(session.session_id, "archive_002")
    assert isinstance(snapshot, ContextSnapshot)
    assert snapshot.messages[0].id == "ov_summary:archive_002"
    assert "成功的概要" in snapshot.messages[0].content
    assert "archived-2" in [m.id for m in snapshot.messages]
    candidate = assembler.refresh(context, snapshot)
    prepared = assembler.prepare(request, candidate)
    assert candidate.archive_id == "archive_002" and candidate.write_index == 1
    assert "本轮召回" in prepared.request.messages[-1].content
    assert context.archive_id is None and context.recall_done


@pytest.mark.parametrize("failure", ["version", "orphan_tool", "history_budget"])
async def test_reader_or_refresh_failures_use_shared_errors_and_leave_cache_unchanged(failure):
    user = HumanMessage(id="current", content="original")
    context = OVContext("s", messages=[user], write_index=7, completed_runs=2)
    context.begin_run("run", user)
    context.inject_recall("reference", "success")
    before = deepcopy(context.to_state())
    client = AsyncMock(spec=AsyncMemoryClient)
    client.get_session_context.return_value = {
        "latest_finished_archive_id": "archive_002",
        "messages": [],
    }
    expected = ContextReadError
    if failure == "version":
        client.get_session_context.return_value["latest_finished_archive_id"] = "archive_001"
    elif failure == "history_budget":
        client.get_session_context.side_effect = ResourceExhaustedError("summary is too large")
        expected = ContextHistoryBudgetError
    else:
        client.get_session_context.return_value["messages"] = [
            {
                "id": "orphan",
                "role": "user",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "missing-call",
                        "tool_name": "lookup",
                        "tool_status": "completed",
                        "tool_output": "result",
                    }
                ],
            }
        ]
    reader = OVContextReader(client)
    with pytest.raises(expected):
        snapshot = await reader.exchange_context("s", "archive_002")
        ContextAssembler(OVContextConfig()).refresh(context, snapshot)
    assert context.to_state() == before


async def test_reader_snapshot_still_requires_full_model_request_budget(request_and_assembler):
    request, _ = request_and_assembler
    user = HumanMessage(id="current", content="original")
    context = OVContext("s", messages=[user], write_index=1)
    context.begin_run("run", user)
    before = deepcopy(context.to_state())
    client = AsyncMock(spec=AsyncMemoryClient)
    client.get_session_context.return_value = {
        "latest_finished_archive_id": None,
        "messages": [{"id": "old", "role": "user", "content": "history " * 4000}],
    }
    snapshot = await OVContextReader(client).exchange_context("s")
    assembler = ContextAssembler(
        OVContextConfig(context_window=1000, output_reserve=0, safety_margin=0)
    )
    candidate = assembler.refresh(context, snapshot)
    with pytest.raises(ContextBudgetExceeded):
        assembler.prepare(request, candidate)
    assert context.to_state() == before
    assert any("history " in str(message.content) for message in candidate.messages)
