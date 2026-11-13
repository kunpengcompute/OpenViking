from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from .test_graph import CallContext, build_middleware

pytestmark = pytest.mark.asyncio


async def test_resumed_turn_uses_persisted_cursor_without_rewriting(env):
    server, _, _, identity, request, _ = env
    middleware = build_middleware(env)
    state = {"messages": [*request.messages, AIMessage(id="prior-call", content="already started")]}
    runtime = SimpleNamespace(context=CallContext(identity))
    state.update(await middleware.abefore_agent(state, runtime))
    assert state["ov_context"]["write_index"] == 2
    fresh = build_middleware(env)
    state.update(await fresh.abefore_agent(state, runtime))
    assert len(server.messages) == 2
    assert server.counts["batch"] == 1
    assert [m.id for m in state["ov_context"]["messages"]] == ["m1", "prior-call"]


async def test_middleware_imports_initial_history_via_synchronous_writer(env):
    server, _, _, identity, request, _ = env
    middleware = build_middleware(env)
    state = {"messages": [HumanMessage(id="old", content="unmigrated"), *request.messages]}
    runtime = SimpleNamespace(context=CallContext(identity))
    state.update(await middleware.abefore_agent(state, runtime))
    assert len(server.messages) == 2
    assert [m.id for m in state["ov_context"]["messages"]] == ["old", "m1"]
    assert state["ov_context"]["write_index"] == 2
