"""Run the public initialization example through a real offline DeepAgents graph."""

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from deepagents import graph as deepagents_graph
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.profiles.harness import harness_profiles
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from examples.deepagents_ov_context.chat.agent import (
    RequestContext,
    create_demo_agent,
)
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import OVContextConfig
from tests.ov_context_support.fixtures import mock_io
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_graph import ToolLoopModel, read_mail


@pytest.mark.asyncio
async def test_demo_builds_ov_owned_context_and_runs_tool_loop(
    env: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, transport, _, identity, _, _ = env
    model = ToolLoopModel()
    monkeypatch.setattr(
        harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES)
    )
    # Keep the provider:model profile lookup while replacing external model I/O.
    monkeypatch.setattr(deepagents_graph, "resolve_model", lambda value: model)
    stack: list[AgentMiddleware[Any, Any]] = []
    real_create_agent = deepagents_graph.create_agent

    def capture_stack(*args: Any, **kwargs: Any) -> Any:
        stack[:] = kwargs["middleware"]
        return real_create_agent(*args, **kwargs)

    monkeypatch.setattr(deepagents_graph, "create_agent", capture_stack)
    client = TransportSDKAdapter(transport)
    config = OVContextConfig(context_window=100000)
    checkpointer = InMemorySaver()
    supplied_io = mock_io(client)
    agent, middleware = create_demo_agent(
        model="ov-demo:tool-loop",
        connection=OpenVikingConnection(client=client),
        context_config=config,
        checkpointer=checkpointer,
        tools=[read_mail],
        **supplied_io,
    )
    assert middleware._ov_session_engine.client.client is client
    for implementation in supplied_io.values():
        assert await implementation.client.get() is client
    assert server.requests == []  # No OV I/O or state initialization at construction.
    assert agent.checkpointer is checkpointer
    assert middleware in stack
    assert not any(item.name == "SummarizationMiddleware" for item in stack)
    filesystem = next(item for item in stack if isinstance(item, FilesystemMiddleware))
    assert filesystem._tool_token_limit_before_evict is None
    assert filesystem._human_message_token_limit_before_evict is None
    runnable_config: RunnableConfig = {"configurable": {"thread_id": identity.session_id}}
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(id="m1", content="Read the mail")]},
            config=runnable_config,
            context=RequestContext(identity),
        )
        assert result["messages"][-1].content == "Mail summarized"
        assert len(model.inputs) == 2
        assert any(isinstance(item, ToolMessage) for item in model.inputs[-1])
        state = (await agent.aget_state(runnable_config)).values
        context = state["ov_context"]
        assert context["session_id"] == identity.session_id
        assert context["last_completed_run"] == identity.run_id
        assert context["run_id"] is None
        assert context["archive_id"] is None
        assert server.counts["context"] == 1
        assert server.counts["find"] == 1
        assert any(tool.name == "read_file" for tool in filesystem.tools)
        assert not middleware._ov_session_engine.active
    except BaseException:
        await middleware.acleanup_run(agent, runnable_config)
        raise
    finally:
        await middleware.aclose()


def test_demo_requires_host_shared_client_before_building_graph(env: tuple[Any, ...]) -> None:
    client = TransportSDKAdapter(env[1])
    config = OVContextConfig(context_window=100000)
    with pytest.raises(ValueError, match="host-created client"):
        create_demo_agent(
            checkpointer=InMemorySaver(),
            model="ov-demo:tool-loop",
            connection=OpenVikingConnection(url="http://unused.invalid"),
            context_config=config,
            **mock_io(client),
        )
    assert not env[0].requests


@pytest.mark.asyncio
async def test_two_agents_two_sessions_each_keep_history_client_and_recall_isolated(
    env: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES)
    )
    monkeypatch.setattr(deepagents_graph, "resolve_model", lambda value: ToolLoopModel())
    from .test_hook_design import Reader
    from .test_io_contracts import SuppliedWriter

    agents = []
    for peer in ("mail-agent", "doc-agent"):
        identity = replace(env[3], actor_peer_id=peer)
        client = TransportSDKAdapter(env[1], peer)
        reader = Reader()
        agent, middleware = create_demo_agent(
            checkpointer=InMemorySaver(),
            model=f"ov-demo:{peer}",
            connection=OpenVikingConnection(client=client),
            context_config=OVContextConfig(context_window=100000),
            context_reader=reader,
            message_writer=SuppliedWriter(),
            tools=[read_mail],
        )
        agents.append((agent, middleware, identity, client, reader))

    async def invoke(index: int, session: int) -> None:
        agent, middleware, base, client, reader = agents[index]
        sid = f"{base.actor_peer_id}-{session}"
        identity = replace(base, session_id=sid, run_id=f"{sid}-run")
        config: RunnableConfig = {"configurable": {"thread_id": sid}}
        await agent.ainvoke(
            {"messages": [HumanMessage(id="same-user-id", content=f"Read {sid}")]},
            config=config,
            context=RequestContext(identity),
        )
        snapshot = (await agent.aget_state(config)).values["ov_context"]
        assert snapshot["session_id"] == sid and snapshot["owner"][2] == base.actor_peer_id
        assert snapshot["messages"][0].content == f"Read {sid}"
        assert snapshot["run_id"] is None and snapshot["write_index"] == 4
        assert snapshot["completed_runs"] == 1
        assert middleware._ov_session_engine.client.client is client
        assert (sid, f"Read {sid}") in reader.queries

    await asyncio.gather(*(invoke(index, session) for index in range(2) for session in range(2)))
    for agent, middleware, base, _client, reader in agents:
        assert len(reader.queries) == 2 and not middleware._ov_session_engine.active
        # Changing Runtime identity must not reuse another Agent's checkpoint.
        with pytest.raises(ValueError, match="different session/identity"):
            await agent.ainvoke(
                {"messages": [HumanMessage(id="bad", content="ignored")]},
                config={"configurable": {"thread_id": f"{base.actor_peer_id}-1"}},
                context=RequestContext(
                    replace(base, session_id=f"{base.actor_peer_id}-1", actor_peer_id="other-agent")
                ),
            )
        await middleware.aclose()
    assert agents[0][3] is not agents[1][3]
