"""Faults cross real Chat/Agent hooks without repeating business execution."""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.deepagents_ov_context.chat.config import ChatOVPolicy
from examples.deepagents_ov_context.chat.ov_resilience import ResilientOVMiddleware
from openviking.integrations.langchain.ov_context import RunIdentity

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events


async def new(client):
    r = await client.post("/api/sessions", json={"execution": {"mode": "current"}})
    assert r.status_code == 201
    return r.json()["session_id"]


@pytest.mark.parametrize("failure", ["unreachable", "slow", "batch", "malformed"])
async def test_ov_failure_keeps_answer_and_next_turn_local(resilient_chat, failure):
    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(io_timeout=0.05))
    calls = []

    async def fault(request):
        calls.append(request.url.path)
        if failure == "slow":
            await asyncio.sleep(10)
        if failure == "unreachable":
            raise httpx.ConnectError("synthetic refusal", request=request)
        if failure == "batch" and request.url.path.endswith("/messages/batch"):
            raise httpx.ReadTimeout("synthetic uncertain write", request=request)
        if failure == "malformed":
            return httpx.Response(200, content=b"invalid JSON")

    server.fault = fault
    sid = await new(client)
    t = time.monotonic()
    response = await client.post("/api/chat", json={"session_id": sid, "message": "保留本地对话"})
    assert time.monotonic() - t < 1.5
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert events[-1][1]["statistics"]["model_calls"] == 1
    view = events[-1][1]["context"]
    assert view["ov_degraded"] and view["ov_sync_pending"]
    assert any(e == "stage" and d.get("name") == "ov.degraded" for e, d in events)
    n = len(calls)
    second = await client.post("/api/chat", json={"session_id": sid, "message": "继续本地对话"})
    assert parse_events(second.text)[-1][0] == "done"
    assert len(calls) == n  # No repeated five-second waits or uncertain writes.
    detail = (await client.get("/api/sessions/" + sid)).json()
    assert not detail["session"]["needs_restore"]
    assert len(detail["context"]["messages"]) == 4
    status = (await client.get(f"/api/sessions/{sid}/memory")).json()
    assert status["status"] == "unknown" and status["stage"] == "ov_degraded"


async def test_after_agent_exception_keeps_completed_answer(resilient_chat, monkeypatch):
    from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware

    client, _, _ = resilient_chat

    async def broken(*args):
        raise RuntimeError("synthetic final hook error")

    monkeypatch.setattr(OpenVikingContextMiddleware, "aafter_agent", broken)
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done"
    assert events[-1][1]["statistics"]["model_calls"] == 1
    assert events[-1][1]["context"]["ov_degraded"]
    assert events[-1][1]["context"]["messages"][-1]["content"] == "done"


async def test_ov_health_is_degraded_but_chat_alive(resilient_chat):
    client, _, server = resilient_chat

    async def fault(request):
        raise httpx.ConnectError("synthetic refusal", request=request)

    server.fault = fault
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ov"]["healthy"] is False
    assert (await client.get("/api/execution-profiles")).status_code == 200
    sid = await new(client)
    detail = await client.get("/api/sessions/" + sid)
    assert detail.status_code == 200 and detail.json()["context"]["ov_degraded"]


@pytest.mark.parametrize("mode", ["current", "context_only"])
@pytest.mark.parametrize("failure", ["disconnect", "unavailable"])
async def test_session_preparation_failure_keeps_budget_checked_local_run(
    resilient_chat, mode, failure
):
    client, _, server = resilient_chat

    async def fault(request):
        if failure == "unavailable":
            return httpx.Response(
                503,
                json={"status": "error", "error": {"code": "UNAVAILABLE", "message": "offline"}},
            )
        raise httpx.ReadError("synthetic disconnected peer", request=request)

    server.fault = fault
    created = await client.post(
        "/api/sessions", json={"execution": {"mode": mode, "memory_extraction": False}}
    )
    sid = created.json()["session_id"]
    for prompt in ("第一次本地请求", "继续本地请求"):
        response = await client.post("/api/chat", json={"session_id": sid, "message": prompt})
        events = parse_events(response.text)
        assert events[-1][0] == "done", events[-1]
        assert events[-1][1]["context"]["ov_degraded"]
        inputs = [data for event, data in events if event == "model_input"]
        assert len(inputs) == 1
        assert inputs[0]["estimated_input_tokens"] <= inputs[0]["input_limit"]


@pytest.mark.parametrize("failure", ["permission", "policy"])
async def test_session_preparation_does_not_hide_permission_or_policy_errors(
    resilient_chat, failure
):
    client, _, server = resilient_chat

    async def fault(request):
        if failure == "policy":
            raise ValueError("Existing session must have extraction disabled")
        return httpx.Response(
            403,
            json={"status": "error", "error": {"code": "PERMISSION_DENIED", "message": "denied"}},
        )

    server.fault = fault
    created = await client.post("/api/sessions", json={"execution": {"mode": "context_only"}})
    response = await client.post(
        "/api/chat", json={"session_id": created.json()["session_id"], "message": "hello"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["statistics"]["model_calls"] == 0
    assert not any(event == "stage" and data.get("name") == "ov.degraded" for event, data in events)


async def test_ov_timeout_does_not_limit_model_duration(resilient_chat):
    client, runtime, _ = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(io_timeout=0.05))

    class SlowModel(NoToolModel):
        def _generate(self, *args, **kwargs):
            time.sleep(0.15)
            return super()._generate(*args, **kwargs)

    runtime.model = SlowModel()
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert not events[-1][1]["context"]["ov_degraded"]


async def test_model_exception_not_replayed_or_hidden(resilient_chat):
    client, runtime, _ = resilient_chat

    class BrokenModel(NoToolModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("synthetic model error")

    runtime.model = BrokenModel()
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["statistics"]["model_calls"] == 1
    assert not any(e == "stage" and d.get("name") == "ov.degraded" for e, d in events)


async def test_hook_cancellation_propagates():
    middleware = ResilientOVMiddleware(commit_config=None)
    identity = RunIdentity("a", "u", "p", "s", "r")
    runtime = SimpleNamespace(context=SimpleNamespace(identity=identity))
    state = {"messages": [HumanMessage(content="hello")]}
    started = asyncio.Event()

    async def pending(*args):
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(middleware._hook("probe", pending, state, runtime))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert middleware.failure is None


async def test_only_ov_tools_have_the_io_deadline():
    middleware = ResilientOVMiddleware(commit_config=None, io_timeout=0.03)
    count = 0

    async def slow(request):
        nonlocal count
        count += 1
        await asyncio.sleep(0.08)
        return "business tool finished"

    browser = SimpleNamespace(tool_call={"name": "browser", "id": "b"}, state={})
    assert await middleware.awrap_tool_call(browser, slow) == "business tool finished"
    ov = SimpleNamespace(tool_call={"name": "openviking_tool_result_read", "id": "o"}, state={})
    result = await middleware.awrap_tool_call(ov, slow)
    assert result.update["messages"][0].tool_call_id == "o"
    assert result.update["messages"][0].status == "error"
    assert result.update["ov_failure"]["error_type"] == "TimeoutError"
    assert count == 2


async def test_timed_out_ov_tool_keeps_tool_pairs_and_finishes(resilient_chat):
    client, runtime, server = resilient_chat
    # Allow graph preparation to finish; only the ten-second tool read should time out.
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(io_timeout=0.2))

    class ReadModel(NoToolModel):
        def _generate(self, messages, *args, **kwargs):
            if any(isinstance(m, ToolMessage) for m in messages):
                return super()._generate(messages, *args, **kwargs)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"id": "ov-list", "name": "openviking_tool_result_list", "args": {}}
                            ],
                        )
                    )
                ]
            )

    runtime.model = ReadModel()
    attempted = []

    async def fault(request):
        if "/tool-results" in request.url.path:
            attempted.append(request.url.path)
            await asyncio.sleep(10)

    server.fault = fault
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "检查工具结果"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert events[-1][1]["statistics"]["model_calls"] == 2
    view = events[-1][1]["context"]
    assert view["ov_degraded"]
    tools = [m for m in view["messages"] if m["type"] == "tool"]
    assert len(tools) == 1 and tools[0]["tool_call_id"] == "ov-list"
    assert tools[0]["status"] == "error" and len(attempted) == 1
    assert any(
        d.get("error_type") == "TimeoutError"
        for e, d in events
        if e == "stage" and d.get("name") == "ov.degraded"
    )


@pytest.mark.parametrize("timeout", [0, -1, 5.01, float("nan")])
def test_timeout_cannot_exceed_five_seconds(timeout):
    with pytest.raises(ValueError):
        ChatOVPolicy(io_timeout=timeout)


async def test_commit_failure_keeps_answer_and_is_not_replayed(resilient_chat):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        commit_config=replace(
            runtime.settings.commit_config, completed_run_threshold=1, keep_recent_count=2
        ),
    )
    server.session_meta_hook = lambda sid: {"pending_tokens": 128}
    attempts = []

    async def fault(request):
        if request.url.path.endswith("/commit"):
            attempts.append(request.url.path)
            raise httpx.ReadTimeout("commit receipt lost", request=request)

    server.fault = fault
    sid = await new(client)
    # Earlier messages provide a safe history to archive if the current Run is kept.
    for prompt in ("first", "second", "after lost receipt"):
        response = await client.post("/api/chat", json={"session_id": sid, "message": prompt})
        assert parse_events(response.text)[-1][0] == "done", response.text
    assert len(attempts) == 1
    detail = (await client.get(f"/api/sessions/{sid}")).json()
    assert detail["context"]["ov_degraded"]
    assert detail["context"]["messages"][-1]["content"] == "done"


async def test_strict_policy_still_propagates_dependency_failure(resilient_chat):
    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(fail_open=False))

    async def fault(request):
        raise httpx.ConnectError("synthetic refusal", request=request)

    server.fault = fault
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    assert parse_events(response.text)[-1][0] == "error"


async def test_context_preparation_failure_calls_raw_model_once(resilient_chat, monkeypatch):
    from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware

    client, _, _ = resilient_chat

    async def broken(*args):
        raise RuntimeError("synthetic context assembly failure")

    monkeypatch.setattr(OpenVikingContextMiddleware, "awrap_model_call", broken)
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert events[-1][1]["statistics"]["model_calls"] == 1
    assert events[-1][1]["context"]["ov_degraded"]


async def test_chat_ov_outage_returns_budget_error_before_model_generation(resilient_chat):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        max_tokens=1000,
        context_config=replace(
            runtime.settings.context_config,
            context_window=6000,
            output_reserve=1000,
            safety_margin=100,
            capacity_ratio=0.7,
        ),
    )

    async def fault(request):
        raise httpx.ConnectError("OV unavailable", request=request)

    server.fault = fault
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "x" * 24000})
    assert response.status_code == 200, response.text
    events = parse_events(response.text)
    assert events[-1][0] == "error", response.text
    assert events[-1][1]["error_type"] == "ContextBudgetExceeded"
    assert events[-1][1]["statistics"]["model_calls"] == 0


async def test_bad_local_ref_keeps_writes_and_automatic_commit(resilient_chat):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        commit_config=replace(
            runtime.settings.commit_config, completed_run_threshold=1, keep_recent_count=2
        ),
    )
    server.session_meta_hook = lambda sid: {"pending_tokens": 128}

    class LocalRefModel(NoToolModel):
        def _generate(self, messages, *args, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                return super()._generate(messages, *args, **kwargs)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "bad-ref",
                                    "name": "openviking_tool_result_search_many",
                                    "args": {
                                        "ref": "tmp/market_metric.json",
                                        "queries": ["excluded_intents"],
                                        "limit": 20,
                                    },
                                }
                            ],
                        )
                    )
                ]
            )

    runtime.model = LocalRefModel()
    commits = []

    async def record(request):
        if request.url.path.endswith("/commit"):
            commits.append(request.url.path)

    server.fault = record
    sid = await new(client)
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "检查本地计算结果"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert not events[-1][1]["context"]["ov_degraded"]
    assert not any(e == "stage" and d.get("name") == "ov.degraded" for e, d in events)
    assert commits, "Correctable tool arguments must not disable automatic after-agent commit"
    detail = (await client.get(f"/api/sessions/{sid}")).json()
    assert len(server.session_messages[sid]) == 4
    assert detail["context"]["write_index"] == 4
    tool_messages = [m for m in detail["context"]["messages"] if m["type"] == "tool"]
    assert tool_messages and all(m["status"] == "error" for m in tool_messages)
    assert all("openviking_tool_result_list" in m["content"] for m in tool_messages)
