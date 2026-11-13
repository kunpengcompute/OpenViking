# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Failed preparation must leave the same Session available for a new Run."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from examples.deepagents_ov_context.chat.config import ChatOVPolicy
from openviking.integrations.langchain.ov_context.budget import ApproximateTokenCounter

from .test_chat_api import parse_events
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


async def test_budget_failure_allows_new_run_after_cleanup(resilient_chat, monkeypatch):
    client, runtime, _ = resilient_chat
    blocked = True
    original = ApproximateTokenCounter.__call__

    def count(counter, request):
        return 1_000_000 if blocked and request.messages else original(counter, request)

    monkeypatch.setattr(ApproximateTokenCounter, "__call__", count)
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
    original = ApproximateTokenCounter.__call__

    def count(counter, request):
        return 1_000_000 if request.messages else original(counter, request)

    async def failed_cleanup(*args, **kwargs):
        raise OSError("checkpoint could not be saved")

    monkeypatch.setattr(ApproximateTokenCounter, "__call__", count)
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
