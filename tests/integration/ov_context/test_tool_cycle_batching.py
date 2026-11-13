"""Opt-in batching uses the graph checkpoint, never a shared pending queue."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain.ov_context import OVContextConfig
from openviking.integrations.langchain.ov_context.writer import (
    IncompleteWriteError,
    OVMessageWriter,
)
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_graph import CallContext, ToolLoopModel, read_mail
from .test_session_commit_lifecycle import build_middleware, initial
from .test_session_hooks import call, middleware, runtime

pytestmark = pytest.mark.asyncio


def build(env, enabled=True, **kwargs):
    mw = middleware(env, batch_tool_messages=enabled, **kwargs)
    mw._ov_session_engine.writer = OVMessageWriter(TransportSDKAdapter(env[1]))
    return mw


def assistant():
    return AIMessage(
        id="call",
        content="Checking mail",
        tool_calls=[{"id": "c1", "name": "read_mail", "args": {}}],
    )


async def started(env, **kwargs):
    mw = build(env, **kwargs)
    state = {"messages": [HumanMessage(id="u1", content="Read mail")]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    return mw, state


@pytest.mark.parametrize("enabled", [False, True])
async def test_cursor_view_and_assembled_input_follow_confirmed_batch(env, enabled):
    mw, state = await started(env, enabled=enabled)
    state["messages"].append(assistant())
    original = deepcopy(state)
    state.update(await mw.aafter_model(state, runtime(env)))
    assert env[0].counts["batch"] == (1 if enabled else 2)
    if enabled:
        assert state == original
    state["messages"].append(ToolMessage(id="t1", tool_call_id="c1", content="mail body"))
    state.update(await mw.abefore_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 3
    assert env[0].counts["batch"] == (2 if enabled else 3)
    captured = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in captured][-2:] == ["call", "t1"]
    assert captured[-1].content == "mail body"
    state["messages"].append(AIMessage(id="final", content="Done"))
    state.update(await mw.aafter_model(state, runtime(env)))
    state.update(await mw.aafter_agent(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 4
    assert env[0].counts["batch"] == (3 if enabled else 4)
    assert await mw.aafter_agent(state, runtime(env)) == {}


@pytest.mark.parametrize("factory", ["langchain", "deepagents"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_graph_keeps_model_inputs_and_reduces_writes(env, factory, enabled):
    if factory == "deepagents":
        from deepagents import create_deep_agent

        factory_fn = create_deep_agent
    else:
        factory_fn = create_agent
    mw, model = build(env, enabled=enabled), ToolLoopModel()
    graph = factory_fn(
        model=model,
        tools=[read_mail],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(id="u1", content="Read mail")]},
        {"configurable": {"thread_id": "s"}},
        context=CallContext(env[3]),
    )
    assert result["messages"][-1].content == "Mail summarized"
    assert len(model.inputs) == 2
    assert any(isinstance(m, ToolMessage) for m in model.inputs[-1])
    saved = (await graph.aget_state({"configurable": {"thread_id": "s"}})).values
    assert saved["ov_context"]["write_index"] == len(result["messages"])
    assert env[0].counts["batch"] == (3 if enabled else 4)
    assert not mw._ov_session_engine.active


async def test_parallel_tool_results_and_error_result_keep_pairing(env):
    mw, state = await started(env)
    state["messages"].append(
        AIMessage(
            id="calls",
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_mail", "args": {}},
                {"id": "c2", "name": "read_mail", "args": {}},
            ],
        )
    )
    state.update(await mw.aafter_model(state, runtime(env)))
    state["messages"].extend(
        [
            ToolMessage(id="t2", tool_call_id="c2", content="failure", status="error"),
            ToolMessage(id="t1", tool_call_id="c1", content="success"),
        ]
    )
    state.update(await mw.abefore_model(state, runtime(env)))
    assert env[0].counts["batch"] == 2
    assert state["ov_context"]["write_index"] == 4
    tools = [m for m in state["ov_context"]["messages"] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.status) for m in tools] == [("c2", "error"), ("c1", "success")]


@pytest.mark.parametrize("factory", ["langchain", "deepagents"])
@pytest.mark.parametrize("handled_error", [False, True])
async def test_real_parallel_tools_wait_for_all_results_before_flushing(
    env, factory, handled_error
):
    both_started, fast_finished, release_slow = asyncio.Event(), asyncio.Event(), asyncio.Event()
    active = peak = 0
    completed = []

    @tool
    async def parallel_read(slot: str) -> str:
        """Read one of two independent sources."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            both_started.set()
        try:
            # A sequential tool runner cannot satisfy this barrier.
            await asyncio.wait_for(both_started.wait(), 5)
            if slot == "slow":
                await asyncio.wait_for(release_slow.wait(), 5)
            completed.append(slot)
            if slot == "fast":
                fast_finished.set()
                if handled_error:
                    raise ToolException("Fast source unavailable")
            return f"Result from {slot}"
        finally:
            active -= 1

    parallel_read.handle_tool_error = True

    class ParallelModel(ToolLoopModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.inputs.append(messages)
            if any(isinstance(m, ToolMessage) for m in messages):
                response = AIMessage(id="final", content="Both sources processed")
            else:
                response = AIMessage(
                    id="calls",
                    content="",
                    tool_calls=[
                        {"id": f"c-{slot}", "name": "parallel_read", "args": {"slot": slot}}
                        for slot in ("slow", "fast")
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=response)])

    if factory == "deepagents":
        from deepagents import create_deep_agent

        factory_fn = create_deep_agent
    else:
        factory_fn = create_agent
    mw, model = build(env), ParallelModel()
    graph = factory_fn(
        model=model,
        tools=[parallel_read],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "s"}}
    task = asyncio.create_task(
        graph.ainvoke(
            {"messages": [HumanMessage(id="u1", content="Read both sources")]},
            cfg,
            context=CallContext(env[3]),
        )
    )
    try:
        await asyncio.wait_for(fast_finished.wait(), 5)
        assert peak == 2 and completed == ["fast"]
        assert not task.done() and len(model.inputs) == 1
        assert env[0].counts["batch"] == 1  # Only the initial user is in OV.
        checkpoint = (await graph.aget_state(cfg)).values
        assert checkpoint["ov_context"]["write_index"] == 1
        release_slow.set()
        result = await asyncio.wait_for(task, 10)
        assert result["messages"][-1].content == "Both sources processed"
        assert completed == ["fast", "slow"] and active == 0
        assert len(model.inputs) == 2 and env[0].counts["batch"] == 3
        results = [m for m in model.inputs[-1] if isinstance(m, ToolMessage)]
        assert len(results) == 2
        by_id = {m.tool_call_id: m for m in results}
        assert set(by_id) == {"c-slow", "c-fast"}
        assert by_id["c-slow"].content == "Result from slow"
        assert by_id["c-fast"].status == ("error" if handled_error else "success")
        assert by_id["c-fast"].content == (
            "Fast source unavailable" if handled_error else "Result from fast"
        )
        checkpoint = (await graph.aget_state(cfg)).values
        assert checkpoint["ov_context"]["write_index"] == 5
        batches = [
            body["messages"] for req, body in env[0].requests if req.url.path.endswith("/batch")
        ]
        outputs = [p for m in batches[1] for p in m["parts"] if "tool_output" in p]
        assert len(outputs) == 2 and {p["tool_id"] for p in outputs} == {"c-slow", "c-fast"}
    finally:
        release_slow.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_return_direct_flushes_in_real_graph(env):
    @tool("read_mail", return_direct=True)
    def direct_mail() -> str:
        """Read mail and deliver directly."""
        return "Direct result"

    mw, model = build(env), ToolLoopModel()
    graph = create_agent(
        model=model,
        tools=[direct_mail],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "s"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(id="u1", content="Read mail")]},
        cfg,
        context=CallContext(env[3]),
    )
    assert len(model.inputs) == 1
    assert result["messages"][-1].content == "Direct result"
    assert (await graph.aget_state(cfg)).values["ov_context"]["write_index"] == 3
    assert env[0].counts["batch"] == 2


async def test_interrupted_graph_recreated_from_checkpoint_flushes_once(env):
    saver = InMemorySaver()
    cfg = {"configurable": {"thread_id": "s"}}

    def graph(mw):
        return create_agent(
            model=ToolLoopModel(),
            tools=[read_mail],
            middleware=[mw],
            context_schema=CallContext,
            checkpointer=saver,
            interrupt_before=["tools"],
        )

    first = graph(build(env))
    await first.ainvoke(
        {"messages": [HumanMessage(id="u1", content="Read mail")]},
        cfg,
        context=CallContext(env[3]),
    )
    checkpoint = (await first.aget_state(cfg)).values
    assert len(checkpoint["messages"]) == 2
    assert checkpoint["ov_context"]["write_index"] == 1
    assert env[0].counts["batch"] == 1
    second = graph(build(env))
    await second.ainvoke(None, cfg, context=CallContext(env[3]))
    assert (await second.aget_state(cfg)).values["ov_context"]["write_index"] == 4
    assert env[0].counts["batch"] == 3


@pytest.mark.parametrize("cancelled", [False, True])
async def test_tool_failure_or_cancellation_can_resume_deferred_messages(env, cancelled):
    entered = asyncio.Event()
    fail = True

    @tool("read_mail")
    async def flaky_mail() -> str:
        """Read mail, with a retryable test interruption before any side effect."""
        entered.set()
        if fail:
            if cancelled:
                await asyncio.Event().wait()
            raise RuntimeError("test tool failure")
        return "Recovered mail"

    mw = build(env)
    graph = create_agent(
        model=ToolLoopModel(),
        tools=[flaky_mail],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "s"}}
    task = asyncio.create_task(
        graph.ainvoke(
            {"messages": [HumanMessage(id="u1", content="Read mail")]},
            cfg,
            context=CallContext(env[3]),
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    if cancelled:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        await task
    checkpoint = (await graph.aget_state(cfg)).values
    assert checkpoint["ov_context"]["write_index"] == 1
    assert env[0].counts["batch"] == 1
    fail = False
    await graph.ainvoke(None, cfg, context=CallContext(env[3]))
    assert (await graph.aget_state(cfg)).values["ov_context"]["write_index"] == 4
    assert env[0].counts["batch"] == 3


async def test_cleanup_keeps_unsent_messages_without_retrying_ambiguous_batch(env):
    mw, state = await started(env, retry_attempts=2)
    state["messages"].append(assistant())
    state.update(await mw.aafter_model(state, runtime(env)))
    state["messages"].append(ToolMessage(id="t1", tool_call_id="c1", content="body"))

    async def fail():
        raise httpx.ReadTimeout("response lost")

    env[0].batch_hook = fail
    before = deepcopy(state)
    with pytest.raises(IncompleteWriteError):
        await mw.abefore_model(state, runtime(env))
    assert state == before
    assert env[0].counts["batch"] == 2

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values=state)

        async def aupdate_state(self, config, update):
            state.update(update)

    await mw.acleanup_run(Graph(), {})
    assert env[0].counts["batch"] == 2
    assert state["messages"] == before["messages"]
    assert state["ov_context"]["write_index"] == 1
    assert state["ov_context"]["run_id"] is None


async def test_other_pending_messages_are_not_silently_deferred(env):
    mw, state = await started(env)
    state["messages"].extend([AIMessage(id="note", content="Interim note"), assistant()])
    state.update(await mw.aafter_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 3
    assert env[0].counts["batch"] == 2


async def test_batching_preserves_the_model_visible_payload(env):
    payloads = []
    for enabled in (False, True):
        mw, model = build(env, enabled=enabled), ToolLoopModel()
        identity = replace(env[3], session_id=f"compare-{enabled}")
        graph = create_agent(
            model=model,
            tools=[read_mail],
            middleware=[mw],
            context_schema=CallContext,
        )
        await graph.ainvoke(
            {"messages": [HumanMessage(id="u1", content="Read mail")]},
            context=CallContext(identity),
        )
        payloads.append(
            [
                [
                    (
                        m.type,
                        m.content,
                        getattr(m, "tool_calls", None),
                        getattr(m, "tool_call_id", None),
                    )
                    for m in messages
                ]
                for messages in model.inputs
            ]
        )
    assert payloads[0] == payloads[1]


async def test_wrong_run_is_rejected_before_defer(env):
    mw, state = await started(env)
    state["messages"].append(assistant())
    with pytest.raises(ValueError, match="different Run"):
        await mw.aafter_model(state, runtime(env, "different"))


async def test_commit_runs_after_combined_write_not_pending_call(env):
    mw, writer, client = build_middleware(env)
    # This host stub opts into the same local encoding contract as the native writer.
    writer.validate = OVMessageWriter(TransportSDKAdapter(env[1])).validate
    mw._ov_session_engine.config = replace(mw._ov_session_engine.config, batch_tool_messages=True)
    state = initial()
    state.update(await mw.abefore_agent(state, runtime(env)))
    state["messages"].append(assistant())
    state.update(await mw.aafter_model(state, runtime(env)))
    assert len(writer.calls) == 1 and client.calls == []
    state["messages"].append(ToolMessage(id="t1", tool_call_id="c1", content="body"))
    state.update(await mw.abefore_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 5
    assert len(writer.calls) == 2 and len(client.calls) == 1


async def test_background_boundary_change_keeps_flushed_tool_cycle(env):
    mw, state = await started(env)
    state["messages"].append(assistant())
    state.update(await mw.aafter_model(state, runtime(env)))
    # Simulate archive movement while the tool is running. The next refresh
    # returns archived history plus the newly flushed live pair.
    server = env[0]
    server.history = deepcopy(server.session_messages["s"])
    server.session_messages["s"] = []
    server.archive_id = "archive_1"
    server.overview = "Earlier work"
    state["messages"].append(ToolMessage(id="t1", tool_call_id="c1", content="new tool result"))
    state.update(await mw.abefore_model(state, runtime(env)))
    captured = await call(mw, state, env[4], runtime(env))
    results = [m for m in captured if isinstance(m, ToolMessage)]
    assert len(results) == 1 and results[0].content == "new tool result"
    assert state["ov_context"]["archive_id"] == "archive_1"


async def test_shared_middleware_keeps_concurrent_sessions_separate(env):
    mw = build(env)
    graph = create_agent(
        model=ToolLoopModel(),
        tools=[read_mail],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )

    async def invoke(i):
        identity = replace(env[3], session_id=f"session{i}", run_id=f"run{i}")
        cfg = {"configurable": {"thread_id": identity.session_id}}
        await graph.ainvoke(
            {"messages": [HumanMessage(id=f"u{i}", content=f"Private request {i}")]},
            cfg,
            context=CallContext(identity),
        )
        saved = (await graph.aget_state(cfg)).values
        assert saved["ov_context"]["write_index"] == 4
        users = [m for m in saved["ov_context"]["messages"] if isinstance(m, HumanMessage)]
        assert [m.id for m in users] == [f"u{i}"]
        assert tuple(saved["ov_context"]["owner"]) == identity.session_key

    await asyncio.gather(*(invoke(i) for i in range(4)))
    assert env[0].counts["batch"] == 12
    assert not mw._ov_session_engine.active


async def test_shared_sdk_without_preflight_keeps_same_session_users_separate(env):
    from .test_request_identity import FakeOVService
    from .test_request_identity import build as identity_build
    from .test_request_identity import runtime as identity_runtime

    sdk = FakeOVService()
    mw, writer, _ = identity_build(sdk)
    mw._ov_session_engine.config = replace(mw._ov_session_engine.config, batch_tool_messages=True)
    identities = [replace(env[3], user_id=f"user{i}") for i in range(3)]

    async def invoke(identity):
        rt = identity_runtime(identity)
        state = {"messages": [HumanMessage(id="u1", content=identity.user_id)]}
        state.update(await mw.abefore_agent(state, rt))
        state["messages"].append(assistant())
        state.update(await mw.aafter_model(state, rt))
        assert state["ov_context"]["write_index"] == 2
        state["messages"].append(ToolMessage(id="t1", tool_call_id="c1", content=identity.user_id))
        state.update(await mw.abefore_model(state, rt))
        state.update(await mw.aafter_agent(state, rt))
        assert state["ov_context"]["messages"][-1].content == identity.user_id

    try:
        await asyncio.gather(*(invoke(identity) for identity in identities))
        assert len(writer.identities) == 9
        assert set(writer.identities) == {i.sdk_identity for i in identities}
        assert len(sdk.messages) == 3 and all(len(v) == 3 for v in sdk.messages.values())
    finally:
        await sdk.http.aclose()


@pytest.mark.parametrize("value", [0, 1, None, "true"])
async def test_batching_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        OVContextConfig(batch_tool_messages=value)
