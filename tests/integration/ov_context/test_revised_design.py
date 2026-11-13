"""Executable contracts for the reviewed session, budgeting and observation design."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from examples.deepagents_ov_context.chat.event_log import RunEventLog
from examples.deepagents_ov_context.chat.history import RunStatistics
from examples.deepagents_ov_context.chat.ov_resilience import ResilientOVMiddleware
from examples.deepagents_ov_context.chat.session import SessionInfo, SessionStore
from examples.deepagents_ov_context.chat.stream import ActiveRun, InputTrace
from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ContextBudgetExceeded,
    OVContext,
    OVContextConfig,
    RunIdentity,
)
from openviking.integrations.langchain.ov_context.budget import ApproximateTokenCounter
from openviking.integrations.langchain.ov_context.errors import IncompleteWriteError
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking.message import Message, TextPart
from openviking.message.part import ToolPart
from openviking.session import Session
from openviking_cli.exceptions import ResourceExhaustedError


class SDK:
    _initialized = True

    def __init__(self) -> None:
        self.rows = []
        self.archive = None
        self.commits = 0
        self.polls = 0
        self.budgets = []
        self.fail_write = False

    async def get_session(self, session_id, *, identity, **kwargs):
        return {
            "session_id": session_id,
            "pending_tokens": 10000,
            "commit_count": self.commits,
            "latest_finished_archive_id": self.archive,
        }

    async def list_tasks(self, **kwargs):
        return []

    async def batch_add_messages(self, session_id, messages, *, identity):
        for spec in messages:
            self.rows.append({**spec, "id": f"msg-{len(self.rows)}"})
        if self.fail_write:
            self.fail_write = False
            raise httpx.ReadTimeout("write stored; response lost")
        return {"session_id": session_id, "added": len(messages), "message_count": len(self.rows)}

    async def get_session_context(self, session_id, token_budget=128000, *, identity):
        self.budgets.append(token_budget)
        return {
            "latest_finished_archive_id": self.archive,
            "latest_archive_overview": "s" * 2200 if self.archive else "",
            "messages": list(self.rows),
        }

    async def commit_session(self, session_id, *, keep_recent_count, identity):
        self.commits += 1
        self.rows = self.rows[-keep_recent_count:] if keep_recent_count else []
        return {
            "archived": True,
            "task_id": "task-1",
            "archive_uri": "viking://session/s/archive_001",
        }

    async def get_task(self, task_id, *, identity):
        self.polls += 1
        if self.polls == 1:
            return {"status": "running"}
        self.archive = "archive_001"
        return {"status": "completed", "result": {"archive_id": self.archive}}


def make(sdk, *, resilient=False, io_timeout=5.0, **kwargs):
    middleware = ResilientOVMiddleware if resilient else OpenVikingContextMiddleware
    return middleware(
        client=sdk,
        message_writer=OVMessageWriter(sdk),
        context_reader=OVContextReader(sdk, MemoryReaderConfig(enabled=False)),
        context_config=OVContextConfig(
            context_window=10000,
            output_reserve=1000,
            safety_margin=100,
            capacity_ratio=0.7,
            poll_interval=0.001,
            **kwargs,
        ),
        token_counter=lambda req: 100 + sum(len(m.content) for m in req.messages),
        **({"io_timeout": io_timeout} if resilient else {}),
    )


def runtime(run_id="r"):
    return SimpleNamespace(
        context=SimpleNamespace(identity=RunIdentity("a", "u", "p", "s", run_id))
    )


def test_history_budget_is_available_before_an_over_limit_request(env):
    assembler = make(SDK())._ov_session_engine.assembler
    context = OVContext("s", messages=[HumanMessage(id="current", content="x" * 10000)])
    history = assembler.history_budget(env[4])
    assert history.fixed_tokens == 100 and history.history_target_tokens == 2900
    assert history.budget.capacity_limit == 7000 and history.budget.input_limit == 8900
    with pytest.raises(ContextBudgetExceeded):
        assembler.prepare(env[4], context)


async def test_capacity_waits_for_archive_and_repairs_only_once(env):
    sdk = SDK()
    mw = make(sdk)
    state = {
        "messages": [
            HumanMessage(id="old", content="h" * 2300),
            AIMessage(id="answer", content="a" * 100),
            HumanMessage(id="current", content="u" * 4800),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime()))
    before = deepcopy(state)

    async def handler(request):
        assert sdk.archive == "archive_001" and sdk.polls >= 2
        assert sdk.commits == 1
        # Still above the trigger, but admissible after the one allowed repair.
        assert 7000 <= mw._ov_session_engine.assembler.token_counter(request) < 8900
        return AIMessage(id="final", content="done")

    result = await mw.awrap_model_call(env[4].override(state=state, runtime=runtime()), handler)
    assert state == before
    assert result.command.update["ov_context"]["archive_id"] == "archive_001"
    assert sdk.budgets[-1] == 2900
    assert not hasattr(mw._ov_session_engine, "locks")


async def capacity_request(mw, request):
    state = {
        "messages": [
            HumanMessage(id="old", content="h" * 2300),
            AIMessage(id="answer", content="a" * 100),
            HumanMessage(id="current", content="u" * 4800),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime()))
    return request.override(state=state, runtime=runtime(), messages=state["messages"])


@pytest.mark.parametrize("resilient", [False, True])
@pytest.mark.parametrize("failure", ["commit", "wait_timeout", "refresh"])
async def test_required_compression_failure_is_a_budget_error(env, resilient, failure):
    sdk = SDK()
    mw = make(sdk, resilient=resilient, commit_wait_timeout=0.02)
    request = await capacity_request(mw, env[4])
    if failure == "commit":
        sdk.commit_session = AsyncMock(side_effect=RuntimeError("commit unavailable"))
    elif failure == "wait_timeout":
        sdk.get_task = AsyncMock(return_value={"status": "running"})
    else:
        sdk.get_session_context = AsyncMock(side_effect=RuntimeError("archive read failed"))
    handler = AsyncMock(return_value=AIMessage(content="done"))
    before = deepcopy(request.state)
    with pytest.raises(ContextBudgetExceeded, match="Capacity compression failed") as raised:
        await mw.awrap_model_call(request, handler)
    assert isinstance(raised.value.__cause__, (RuntimeError, TimeoutError))
    handler.assert_not_awaited()
    assert request.state == before
    if resilient:
        assert mw.failure is None


async def test_required_compression_wait_deadline_is_a_budget_error(env):
    sdk = SDK()
    mw = make(sdk, resilient=True, io_timeout=0.05, commit_wait_timeout=0.1)
    request = await capacity_request(mw, env[4])
    sdk.get_task = AsyncMock(return_value={"status": "running"})
    handler = AsyncMock(return_value=AIMessage(content="done"))
    with pytest.raises(ContextBudgetExceeded, match="Capacity compression failed"):
        await mw.awrap_model_call(request, handler)
    assert sdk.commits == 1
    handler.assert_not_awaited()
    assert mw.failure is None
    assert mw._preparation_deadline.get() is None


async def test_archive_wait_can_exceed_short_optional_io_deadline(env):
    sdk = SDK()
    mw = make(sdk, resilient=True, io_timeout=0.05, commit_wait_timeout=0.5)
    request = await capacity_request(mw, env[4])
    original = sdk.get_task

    async def slow_archive(*args, **kwargs):
        await asyncio.sleep(0.06)
        return await original(*args, **kwargs)

    sdk.get_task = slow_archive
    handler = AsyncMock(return_value=AIMessage(content="done"))
    result = await mw.awrap_model_call(request, handler)
    assert result.command.update["ov_context"]["archive_id"] == "archive_001"
    handler.assert_awaited_once()
    assert mw.failure is None
    assert mw._preparation_deadline.get() is None


async def test_caller_cancellation_during_compression_still_propagates(env):
    sdk = SDK()
    mw = make(sdk, resilient=True, commit_wait_timeout=10)
    request = await capacity_request(mw, env[4])
    started = asyncio.Event()

    async def pending(*args, **kwargs):
        started.set()
        await asyncio.sleep(10)

    sdk.commit_session = pending
    handler = AsyncMock(return_value=AIMessage(content="done"))
    task = asyncio.create_task(mw.awrap_model_call(request, handler))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    handler.assert_not_awaited()
    assert mw.failure is None


@pytest.mark.parametrize("disabled", ["instance", "checkpoint"])
@pytest.mark.parametrize("over_limit", [False, True])
async def test_disabled_ov_checks_the_actual_fallback_request_budget(env, disabled, over_limit):
    mw = make(SDK(), resilient=True)
    messages = [HumanMessage(id="u", content="x" * (8801 if over_limit else 8800))]
    state = {"messages": messages}
    failure = {"stage": "before_agent", "error_type": "TimeoutError", "persistence": "unconfirmed"}
    if disabled == "instance":
        mw.failure = failure
    else:
        state["ov_failure"] = failure
    request = env[4].override(state=state, runtime=runtime(), messages=messages)
    before = deepcopy(state)
    handler = AsyncMock(return_value=AIMessage(content="done"))
    if over_limit:
        with pytest.raises(ContextBudgetExceeded, match="8901 exceeds 8900"):
            await mw.awrap_model_call(request, handler)
        handler.assert_not_awaited()
    else:
        from langchain_core.runnables.config import ensure_config

        seen = []

        async def callback(prepared):
            seen.append(ensure_config()["metadata"]["ov_input_budget"].copy())
            return AIMessage(content="done")

        handler.side_effect = callback
        await mw.awrap_model_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert seen[0]["estimated_input_tokens"] == seen[0]["input_limit"] == 8900
        assert seen[0]["source"] == "local_fallback"
        assert "ov_input_budget" not in ensure_config()["metadata"]
    assert state == before


@pytest.mark.parametrize("over_limit", [False, True])
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_optional_ov_failure_checks_the_fallback_budget(
    env, monkeypatch, over_limit, failure
):
    mw = make(SDK(), resilient=True, io_timeout=0.03)

    async def broken(*args):
        if failure == "timeout":
            await asyncio.sleep(10)
        raise RuntimeError("optional OV preparation failed")

    monkeypatch.setattr(OpenVikingContextMiddleware, "awrap_model_call", broken)
    messages = [HumanMessage(id="u", content="x" * (8801 if over_limit else 8800))]
    request = env[4].override(state={"messages": messages}, runtime=runtime(), messages=messages)
    handler = AsyncMock(return_value=AIMessage(content="done"))
    if over_limit:
        with pytest.raises(ContextBudgetExceeded, match="8901 exceeds 8900"):
            await mw.awrap_model_call(request, handler)
        handler.assert_not_awaited()
    else:
        result = await mw.awrap_model_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result.command.update["ov_failure"] == mw.failure
    assert mw.failure["error_type"] == ("TimeoutError" if failure == "timeout" else "RuntimeError")


@pytest.mark.parametrize("overhead", ["system", "tools"])
async def test_fallback_budget_includes_system_and_tool_schemas(env, overhead):
    mw = make(SDK(), resilient=True)
    mw._ov_session_engine.assembler.token_counter = ApproximateTokenCounter()
    messages = [HumanMessage(id="u", content="hello")]
    state = {"messages": messages, "ov_failure": {"error_type": "TimeoutError"}}
    request = env[4].override(
        state=state, runtime=runtime(), messages=messages, system_message=None, tools=[]
    )
    if overhead == "system":
        request = request.override(system_message=SystemMessage(content="x" * 50000))
    else:
        request = request.override(
            tools=[
                {"name": "browser", "description": "x" * 50000, "parameters": {"type": "object"}}
            ]
        )
    handler = AsyncMock(return_value=AIMessage(content="done"))
    with pytest.raises(ContextBudgetExceeded):
        await mw.awrap_model_call(request, handler)
    handler.assert_not_awaited()


async def test_new_run_replays_saved_cursor_after_a_lost_write_response():
    sdk = SDK()
    sdk.fail_write = True
    state = {
        "messages": [HumanMessage(id="first", content="first")],
        "ov_context": OVContext("s", context_status="ready", owner=("a", "u", "p", "s")).to_state(),
    }
    first = make(sdk)
    with pytest.raises(IncompleteWriteError):
        await first.abefore_agent(state, runtime("old-run"))
    assert state["ov_context"]["write_index"] == 0
    state["messages"].append(HumanMessage(id="second", content="second"))
    rebuilt = make(sdk)
    result = await rebuilt.abefore_agent(state, runtime("new-run"))
    assert result["ov_context"]["write_index"] == 2
    assert result["ov_context"]["run_id"] == "new-run"
    assert [row["parts"][0]["text"] for row in sdk.rows] == ["first", "first", "second"]


async def test_summary_has_priority_and_oversized_summary_is_an_error():
    session = Session.__new__(Session)
    session.session_id = "s"
    messages = [
        Message(id="old", role="user", parts=[TextPart("old" * 100)]),
        Message(id="new", role="user", parts=[TextPart("new")]),
    ]
    session._collect_session_context_components = AsyncMock(
        return_value={
            "messages": messages,
            "latest_archive": {"overview": "full summary", "overview_tokens": 20},
            "latest_finished_archive_id": "archive_001",
            "total_archives": 1,
            "failed_archives": 0,
        }
    )
    result = await session.get_session_context(20 + messages[-1].estimated_tokens)
    assert result["latest_archive_overview"] == "full summary"
    assert [m["id"] for m in result["messages"]] == ["new"]
    with pytest.raises(ResourceExhaustedError) as raised:
        await session.get_session_context(19)
    assert raised.value.details == {"token_budget": 19, "summary_tokens": 20}


def test_parallel_tool_results_are_selected_as_a_complete_group():
    rows = [
        Message(id="q", role="user", parts=[TextPart("question")]),
        Message(id="a", role="assistant", parts=[ToolPart(tool_id="c1"), ToolPart(tool_id="c2")]),
        Message(id="t2", role="user", parts=[ToolPart(tool_id="c2", tool_output="two")]),
        Message(id="t1", role="user", parts=[ToolPart(tool_id="c1", tool_output="one")]),
    ]
    group_tokens = sum(m.estimated_tokens for m in rows[1:])
    assert Session._budget_recent_messages(rows, group_tokens) == rows[1:]
    assert Session._budget_recent_messages(rows, group_tokens - 1) == []


async def test_real_graph_callback_correlates_input_and_budget_in_jsonl(tmp_path: Path):
    class Model(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    sdk = SDK()
    mw = make(sdk)
    store = SessionStore(tmp_path / "sessions.db")
    log = RunEventLog(tmp_path / "runs.jsonl")
    session = SessionInfo(
        session_id="s",
        account_id="a",
        user_id="u",
        agent_id="p",
        created_at=0,
        last_run=RunStatistics(run_id="r"),
    )
    store.save(session)
    store.history.begin("s", "r", "hello")
    run = ActiveRun(run_id="r", session_id="s", history=store.history)
    run.bind_log(session, log)
    graph = create_agent(
        Model(responses=["hello"]),
        middleware=[mw],
        checkpointer=InMemorySaver(),
        context_schema=SimpleNamespace,
    )
    try:
        await graph.ainvoke(
            {"messages": [HumanMessage(id="u", content="hello")]},
            config={
                "configurable": {"thread_id": "thread-s"},
                "callbacks": [InputTrace(run, session, store)],
            },
            context=runtime().context,
        )
        log.flush()
        records = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
        start = next(r for r in records if r["event"] == "model.start")
        model_input = next(r for r in records if r["event"] == "model_input")
        usage = next(r for r in records if r["event"] == "model_usage")
        assert start["model_call_id"] == model_input["model_call_id"] == usage["model_call_id"]
        assert start["run_id"] == "r" and start["model_call_id"] != "r"
        assert model_input["data"]["estimated_input_tokens"] == 105
        assert model_input["data"]["messages"][0]["type"] == "system"
        assert store.get("s").last_run.model_calls == 1
        assert store.get("s").last_run.total_tokens is None
    finally:
        log.close()
        store.close()


async def test_session_freezes_inherited_config_across_service_rebuild(chat):
    from examples.deepagents_ov_context.chat.server import ChatRuntime
    from openviking.integrations.langchain.ov_context.commit import CommitConfig

    client, first = chat
    first.settings = replace(
        first.settings,
        reader_config=replace(first.settings.reader_config, recall_limit=2, max_injected_chars=777),
        commit_config=CommitConfig(9000, 0.4, 6, 0),
    )
    response = await client.post("/api/sessions", json={"execution": {"base_context_ratio": 0.2}})
    assert response.status_code == 201
    sid = response.json()["session_id"]
    original = first.store.get(sid).config_snapshot
    changed_settings = replace(
        first.settings,
        reader_config=replace(first.settings.reader_config, recall_limit=7, max_injected_chars=999),
        commit_config=CommitConfig(12000, 0.5, 9, 5),
        context_config=replace(first.settings.context_config, base_context_ratio=0.4),
    )
    rebuilt = ChatRuntime(changed_settings, first.model_config, checkpointer=first.saver)
    try:
        session = rebuilt.store.get(sid)
        assert session.config_snapshot == original
        sdk = rebuilt.sdk()
        _, middleware = rebuilt.session_graph(session, sdk, ActiveRun())
        assert middleware._ov_session_engine.config.base_context_ratio == 0.2
        assert middleware._ov_session_engine.context_reader.config.recall_limit == 2
        assert middleware._ov_session_engine.session_commit_policy.config.keep_recent_count == 0
        await middleware.aclose()
        await sdk.close()
        first.settings = changed_settings
        new_response = await client.post("/api/sessions", json={})
        fresh = first.store.get(new_response.json()["session_id"]).config_snapshot
        assert fresh.reader_config.recall_limit == 7
        assert fresh.context_config.base_context_ratio == 0.4
    finally:
        await rebuilt.close()


@pytest.mark.parametrize("mode", ["current", "ordinary", "baseline"])
async def test_rebuilt_session_keeps_summary_policy_for_its_snapshot_model(chat, monkeypatch, mode):
    import deepagents.graph as deepagents_graph
    from deepagents.profiles.harness import harness_profiles

    from examples.deepagents_ov_context.chat.server import ChatRuntime

    client, first = chat
    response = await client.post("/api/sessions", json={"execution": {"mode": mode}})
    assert response.status_code == 201
    sid = response.json()["session_id"]
    snapshot = first.store.get(sid).config_snapshot
    original_model = snapshot.execution["model"]
    new_model = original_model + "-new-default"
    # A fresh process has no model registrations left by the previous runtime.
    monkeypatch.setattr(
        harness_profiles,
        "_HARNESS_PROFILES",
        {
            key: value
            for key, value in harness_profiles._HARNESS_PROFILES.items()
            if key not in {f"openai:{original_model}", f"openai:{new_model}"}
        },
    )
    built = []
    create_agent = deepagents_graph.create_agent

    def capture(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        built.append((model, list(kwargs.get("middleware", ()))))
        return create_agent(*args, **kwargs)

    monkeypatch.setattr(deepagents_graph, "create_agent", capture)
    rebuilt = ChatRuntime(
        first.settings, replace(first.model_config, chat_model=new_model), checkpointer=first.saver
    )
    sdk = rebuilt.sdk()
    middleware = None
    try:
        session = rebuilt.store.get(sid)
        _, middleware = rebuilt.session_graph(session, sdk, ActiveRun())
        model, components = built[-1]
        assert rebuilt.model_config.chat_model == new_model
        assert model.model_name == original_model
        has_framework_summary = any(
            "SummarizationMiddleware" in type(component).__name__ for component in components
        )
        assert has_framework_summary is (mode == "baseline")
        assert rebuilt.store.get(sid).config_snapshot == snapshot
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()
        await rebuilt.close()


async def test_owned_sdk_uses_trusted_identity_without_closing_the_borrowed_pool():
    import asyncio

    from openviking.integrations.langchain.client import OpenVikingConnection
    from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "ok", "result": {"session_id": "s"}})

    pool = httpx.AsyncClient(transport=httpx.MockTransport(respond), headers={"X-Pool": "shared"})
    before = dict(pool.headers)
    handle = AsyncClientHandle(
        OpenVikingConnection(
            url="http://ov.test",
            api_key="test-service-credential",
            http_client=pool,
            timeout=3,
        )
    )
    try:
        one, two = await asyncio.gather(handle.get(), handle.get())
        assert one is two
        identities = [
            RunIdentity("a", user, "p", "s", "r").sdk_identity for user in ("alice", "bob")
        ]
        await asyncio.gather(
            *(handle.call("get_session", session_id="s", identity=i) for i in identities)
        )
        assert {r.headers["X-OpenViking-User"] for r in requests} == {"alice", "bob"}
        assert all(r.headers["X-API-Key"] == "test-service-credential" for r in requests)
        assert dict(pool.headers) == before
        await handle.close()
        assert not pool.is_closed and one._http is None
    finally:
        await pool.aclose()


async def test_log_failure_marks_observation_incomplete_without_failing_delivery(tmp_path):
    class UnavailableLog:
        def append(self, record):
            raise OSError("disk unavailable")

    session = SessionInfo(
        session_id="s",
        account_id="a",
        user_id="u",
        agent_id="p",
        created_at=0,
        last_run=RunStatistics(run_id="r"),
    )
    run = ActiveRun(run_id="r")
    run.bind_log(session, UnavailableLog())
    run.statistics = session.last_run
    await run.emit("done", statistics=session.last_run.model_dump())
    event = await run.queue.get()
    assert event.event == "done" and event.data["statistics"]["observations_complete"] is False
    assert session.last_run.observations_complete is False


def test_event_log_appends_across_process_instances_and_escapes_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    for run_id in ("r1", "r2"):
        log = RunEventLog(path)
        log.append({"run_id": run_id, "data": {"text": "one\ntwo"}})
        log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["run_id"] for row in rows] == ["r1", "r2"]
    assert rows[0]["data"]["text"] == "one\ntwo"
