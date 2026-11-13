"""Deterministic write validation must precede side-effecting tool execution."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from tests.ov_context_support.transport import OVHTTPError

from .test_graph import CallContext, ToolLoopModel
from .test_session_hooks import runtime
from .test_tool_cycle_batching import assistant, build, started

pytestmark = pytest.mark.asyncio


def invalid_message(kind):
    message = AIMessage(
        id="bad-call",
        content="",
        tool_calls=[{"name": "record_event", "id": "c1", "args": {}}],
    )
    if kind == "unsupported_block":
        message.content = [{"type": "redacted_thinking", "data": "opaque"}]
    elif kind == "mismatched_tool_use":
        message.content = [{"type": "tool_use", "id": "c1", "name": "wrong", "input": {}}]
    elif kind == "not_json":
        message.tool_calls[0]["args"] = {"value": {1, 2}}
    elif kind == "nan":
        message.tool_calls[0]["args"] = {"value": float("nan")}
    elif kind == "injected_view":
        message.additional_kwargs["ov_context_injected"] = True
    return message


@pytest.mark.parametrize("factory", ["langchain", "deepagents"])
@pytest.mark.parametrize(
    "kind", ["unsupported_block", "mismatched_tool_use", "not_json", "nan", "injected_view"]
)
@pytest.mark.parametrize("enabled", [False, True])
async def test_invalid_ai_blocks_tools_even_after_resume(env, factory, kind, enabled):
    effects = []

    @tool
    def record_event(value: Any = None) -> str:
        """Perform a side effect recorded by the regression test."""
        effects.append("executed")
        return "Recorded"

    class InvalidModel(ToolLoopModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.inputs.append(messages)
            message = invalid_message(kind)
            return ChatResult(generations=[ChatGeneration(message=message)])

    mw, model = build(env, enabled=enabled), InvalidModel()
    if factory == "deepagents":
        from deepagents import create_deep_agent

        factory_fn = create_deep_agent
    else:
        factory_fn = create_agent
    graph = factory_fn(
        model=model,
        tools=[record_event],
        middleware=[mw],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "s"}}
    for attempt in range(2):
        with pytest.raises((ValueError, TypeError)):
            await graph.ainvoke(
                {"messages": [HumanMessage(id="u1", content="Record an event")]}
                if attempt == 0
                else None,
                cfg,
                context=CallContext(env[3]),
            )
    assert effects == []
    assert len(model.inputs) == 1
    assert env[0].counts["batch"] == 1
    assert (await graph.aget_state(cfg)).values["ov_context"]["write_index"] == 1


async def test_preflight_uses_write_encoder_without_network_or_mutation(env):
    sdk = AsyncMock()
    writer = OVMessageWriter(sdk)
    messages = [assistant()]
    original = deepcopy(messages)
    assert writer.validate(messages, identity=env[3].sdk_identity, peer_id="p") is None
    assert sdk.mock_calls == [] and messages == original
    for kind in ("unsupported_block", "mismatched_tool_use", "not_json", "nan", "injected_view"):
        bad = invalid_message(kind)
        with pytest.raises((ValueError, TypeError)) as preflight:
            writer.validate([bad], identity=env[3].sdk_identity)
        with pytest.raises(type(preflight.value)) as sending:
            await writer.write("s", [bad], 0, identity=env[3].sdk_identity)
        assert str(preflight.value) == str(sending.value)
    assert sdk.mock_calls == []


@pytest.mark.parametrize("capability", ["missing", "async", "dynamic"])
async def test_host_without_declared_sync_validation_keeps_immediate_write(env, capability):
    mw, state = await started(env)
    native = mw._ov_session_engine.writer

    class Host:
        async def write(self, *args, **kwargs):
            return await native.write(*args, **kwargs)

    class AsyncHost(Host):
        async def validate(self, *args, **kwargs):
            raise AssertionError("An asynchronous validator must not be used for deferral")

    class DynamicHost(Host):
        def __getattr__(self, name):
            if name == "validate":
                raise AssertionError("Dynamic proxies must not fabricate batching capabilities")
            raise AttributeError(name)

    mw._ov_session_engine.writer = {"missing": Host, "async": AsyncHost, "dynamic": DynamicHost}[
        capability
    ]()
    state["messages"].append(assistant())
    state.update(await mw.aafter_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 2
    assert env[0].counts["batch"] == 2


async def test_invalid_host_message_fails_on_original_write_path(env):
    mw, state = await started(env)
    native = mw._ov_session_engine.writer
    mw._ov_session_engine.writer = SimpleNamespace(write=native.write)
    state["messages"].append(invalid_message("unsupported_block"))
    with pytest.raises(ValueError, match="Unsupported message content block"):
        await mw.aafter_model(state, runtime(env))
    assert state["ov_context"]["write_index"] == 1 and env[0].counts["batch"] == 1


@pytest.mark.parametrize("enabled", [False, True])
async def test_server_refusal_is_not_guaranteed_by_local_validation(env, enabled):
    effects = []

    @tool("read_mail")
    def side_effect() -> str:
        """Record a side effect before a later server-only rejection."""
        effects.append("executed")
        return "Done"

    class RejectedModel(ToolLoopModel):
        def _generate(self, *args, **kwargs):
            # Initial user is already stored; OV now rejects subsequent writes.
            env[0].error["batch"] = 400
            return super()._generate(*args, **kwargs)

    graph = create_agent(
        model=RejectedModel(),
        tools=[side_effect],
        middleware=[build(env, enabled=enabled)],
        context_schema=CallContext,
    )
    with pytest.raises(OVHTTPError):
        await graph.ainvoke(
            {"messages": [HumanMessage(id="u1", content="Read mail")]},
            context=CallContext(env[3]),
        )
    assert effects == (["executed"] if enabled else [])
    assert len(env[0].messages) == 1
