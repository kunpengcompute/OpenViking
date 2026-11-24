# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Failed preparation must leave the same Session available for a new Run."""

import asyncio
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from examples.deepagents_ov_context.chat.config import ChatOVPolicy
from openviking.integrations.langchain.ov_context.assembler import ContextAssembler
from openviking.integrations.langchain.ov_context.errors import ContextBudgetExceeded

from .test_chat_api import new_session, parse_events
from .test_chat_resilience import new


@pytest.mark.parametrize("failed_batch", [1, 2])
async def test_presend_write_failure_allows_new_run_on_same_session(resilient_chat, failed_batch):
    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(fail_open=False))
    batches = 0
    blocked = True

    async def fault(request):
        nonlocal batches
        if request.url.path.endswith("/messages/batch"):
            batches += 1
            if blocked and batches >= failed_batch:
                raise httpx.ConnectError("batch could not be sent", request=request)

    server.fault = fault
    sid = await new(client)
    first = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "first"}), 5
    )
    first_events = parse_events(first.text)
    assert first_events[-1][0] == "error", first.text
    assert first_events[-1][1]["can_continue"]
    assert first_events[-1][1]["statistics"]["model_calls"] == failed_batch - 1
    assert sid not in runtime.active
    blocked = False
    second = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "continue"}), 5
    )
    assert second.status_code == 200, second.text
    second_events = parse_events(second.text)
    assert second_events[-1][0] == "done", second.text
    assert second_events[-1][1]["statistics"]["model_calls"] == 1
    assert second_events[-1][1]["context"]["run_id"] is None
    assert not runtime.store.get(sid).needs_restore
    assert sid not in runtime.active


async def test_budget_failure_cleanup_releases_run_for_next_request(resilient_chat, monkeypatch):
    client, runtime, _ = resilient_chat
    blocked = True
    original = ContextAssembler.prepare

    def prepare(assembler, *args, **kwargs):
        if blocked:
            raise ContextBudgetExceeded("synthetic model-input budget failure")
        return original(assembler, *args, **kwargs)

    monkeypatch.setattr(ContextAssembler, "prepare", prepare)
    sid = await new(client)
    first = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "first"}), 5
    )
    events = parse_events(first.text)
    assert events[-1][0] == "error", first.text
    assert events[-1][1]["error_type"] == "ContextBudgetExceeded"
    assert events[-1][1]["statistics"]["model_calls"] == 0
    assert events[-1][1]["can_continue"]
    assert events[-1][1]["stage"] == "budget"
    assert sid not in runtime.active
    blocked = False
    second = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "continue"}), 5
    )
    assert second.status_code == 200, second.text
    assert parse_events(second.text)[-1][0] == "done", second.text
    assert not runtime.store.get(sid).needs_restore
    assert sid not in runtime.active


@pytest.mark.parametrize("existing_history", [False, True])
@pytest.mark.parametrize("mode", ["current", "context_only", "ordinary", "experience"])
async def test_oversized_input_preserves_session_and_allows_short_followup(
    resilient_chat, existing_history, mode
):
    client, runtime, server = resilient_chat
    sid = await new_session(
        client,
        execution={
            "mode": mode,
            "context_window": 12000,
            "max_tokens": 1024,
            "enable_todos": False,
        },
    )
    if existing_history:
        response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
        assert parse_events(response.text)[-1][0] == "done"
    session = runtime.store.get(sid)
    before_state = deepcopy(await runtime.state(session))
    before_checkpoint = await runtime.saver.aget_tuple(session.graph_config)
    before_ov = deepcopy(server.session_messages)
    before_requests = server.counts.copy()

    response = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "预算测试" * 7400}), 5
    )
    assert response.status_code == 200
    event, result = parse_events(response.text)[-1]
    assert event == "error"
    assert result["error_type"] == "ContextBudgetExceeded"
    assert result["statistics"]["model_calls"] == 0
    assert result["stage"] == "budget"
    assert result["can_continue"]
    assert sid not in runtime.active
    assert not runtime.store.get(sid).needs_restore
    assert await runtime.state(session) == before_state
    assert await runtime.saver.aget_tuple(session.graph_config) == before_checkpoint
    assert server.session_messages == before_ov
    assert server.counts == before_requests

    response = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "continue"}), 5
    )
    event, result = parse_events(response.text)[-1]
    assert event == "done", response.text
    assert result["statistics"]["model_calls"] == 1
    assert result["context"]["write_index"] == (4 if existing_history else 2)
    state = await runtime.state(session)
    assert all("预算测试" not in str(message.content) for message in state["messages"])


async def test_input_preflight_includes_frozen_system_prompt(resilient_chat):
    client, runtime, server = resilient_chat
    sid = await new_session(
        client,
        execution={
            "mode": "context_only",
            "context_window": 12000,
            "max_tokens": 1024,
            "enable_todos": False,
            "system_prompt": "系统说明" * 200,
        },
    )
    # Each part fits alone, but fixed instructions plus the incoming input do not.
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "预算测试" * 1500}
    )
    event, result = parse_events(response.text)[-1]
    assert event == "error", response.text
    assert result["error_type"] == "ContextBudgetExceeded"
    assert result["can_continue"]
    assert not server.counts
    assert await runtime.state(runtime.store.get(sid)) == {}
    response = await client.post("/api/chat", json={"session_id": sid, "message": "short"})
    event, result = parse_events(response.text)[-1]
    assert event == "done", result.get("error_type")


async def test_lost_write_receipt_preserves_explicit_restore_boundary(resilient_chat):
    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(fail_open=False))

    async def fault(request):
        if request.url.path.endswith("/messages/batch"):
            raise httpx.ReadTimeout("write outcome is unknown", request=request)

    server.fault = fault
    sid = await new(client)
    first = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "first"}), 5
    )
    assert parse_events(first.text)[-1][0] == "error", first.text
    assert runtime.store.get(sid).needs_restore
    assert sid not in runtime.active
    server.fault = None
    response = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "continue"}), 5
    )
    assert response.status_code == 409


async def test_budget_cleanup_failure_keeps_restore_boundary(resilient_chat, monkeypatch):
    from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware

    client, runtime, _ = resilient_chat

    def prepare(*args, **kwargs):
        raise ContextBudgetExceeded("synthetic model-input budget failure")

    async def failed_cleanup(*args, **kwargs):
        raise OSError("checkpoint could not be saved")

    monkeypatch.setattr(ContextAssembler, "prepare", prepare)
    monkeypatch.setattr(OpenVikingContextMiddleware, "acleanup_run", failed_cleanup)
    sid = await new(client)
    response = await asyncio.wait_for(
        client.post("/api/chat", json={"session_id": sid, "message": "first"}), 5
    )
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert not events[-1][1].get("can_continue", False)
    assert runtime.store.get(sid).needs_restore
    assert sid not in runtime.active
