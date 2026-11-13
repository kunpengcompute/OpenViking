import asyncio
from dataclasses import dataclass

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    OVContextConfig,
    RunIdentity,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from tests.ov_context_support.fixtures import BUDGET, POLICY, mock_io
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

pytestmark = pytest.mark.asyncio


@dataclass
class CallContext:
    identity: RunIdentity


class ToolLoopModel(BaseChatModel):
    inputs: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-tool-loop"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.inputs.append(messages)
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(
                id="final",
                content="Mail summarized",
                usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            )
        else:
            message = AIMessage(
                id="call", content="", tool_calls=[{"name": "read_mail", "id": "c1", "args": {}}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_middleware(env):
    _, transport, options, _, _, _ = env
    client = TransportSDKAdapter(transport)
    config = OVContextConfig(
        context_window=BUDGET.context_window,
        output_reserve=BUDGET.output_reserve,
        safety_margin=BUDGET.safety_margin,
        context_failure=options.context_failure,
    )
    reader_config = MemoryReaderConfig(
        search_mode="find",
        target_uris=POLICY.target_uri or MemoryReaderConfig().target_uris,
        recall_limit=POLICY.limit,
        candidate_limit=max(10, POLICY.limit),
        score_threshold=POLICY.score_threshold or 0,
        prefer_abstract=POLICY.content_mode != "full",
    )
    return OpenVikingContextMiddleware(
        client=client,
        peer_id_resolver=lambda state, runtime: runtime.context.identity.actor_peer_id,
        context_config=config,
        target_uri=list(POLICY.target_uri),
        token_counter=ApproximateTokenCounter(),
        commit_config=CommitConfig(100000, 0.7, 1, 10),
        metrics=options.metrics,
        **mock_io(client, reader_config=reader_config),
    )


@tool
def read_mail() -> str:
    """Read an example mail."""
    return "Final mail body after tool processing"


async def check_loop(factory, env):
    server, transport, options, identity, _, samples = env
    model = ToolLoopModel()
    middleware = build_middleware(env)
    agent = factory(
        model=model, tools=[read_mail], middleware=[middleware], context_schema=CallContext
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(id="m1", content="Read the mail")]},
        context=CallContext(identity),
    )
    assert result["messages"][-1].content == "Mail summarized"
    assert len(model.inputs) == 2
    assert server.counts["find"] == 1 and server.counts["context"] == 1
    assert server.counts["commit"] == 0
    assert any(
        part.get("tool_output") == "Final mail body after tool processing"
        for message in server.messages
        for part in message["parts"]
    )
    assert not any(
        message.additional_kwargs.get("ov_context_injected") for message in result["messages"]
    )
    assert not middleware._ov_session_engine.active
    assert len([event for event, _ in samples if event == "context.estimate"]) == 2
    await transport.http.aclose()


async def test_real_langchain_tool_loop(env):
    await check_loop(create_agent, env)


async def test_real_deepagents_tool_loop(env):
    deepagents = pytest.importorskip("deepagents")
    await check_loop(deepagents.create_deep_agent, env)


async def test_cancel_during_tool_cleans_up_graph_state(env):
    _, transport, options, identity, _, _ = env
    entered = asyncio.Event()

    @tool
    async def read_mail() -> str:
        """Wait for a cancellable mail read."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    middleware = build_middleware(env)
    agent = create_agent(
        model=ToolLoopModel(),
        tools=[read_mail],
        middleware=[middleware],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": identity.session_id}}

    async def invoke():
        try:
            await agent.ainvoke(
                {"messages": [HumanMessage(id="m1", content="Read the mail")]},
                context=CallContext(identity),
                config=config,
            )
        finally:
            await middleware.acleanup_run(agent, config)

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not middleware._ov_session_engine.active
    assert (await agent.aget_state(config)).values["ov_context"]["run_id"] is None
    await transport.http.aclose()


async def test_real_deepagents_stream_and_model_failure_cleanup(env):
    from deepagents import create_deep_agent

    server, transport, options, identity, _, samples = env
    middleware = build_middleware(env)
    agent = create_deep_agent(
        model=ToolLoopModel(),
        tools=[read_mail],
        middleware=[middleware],
        context_schema=CallContext,
    )
    events = []
    async for event in agent.astream(
        {"messages": [HumanMessage(id="m1", content="Read the mail")]},
        context=CallContext(identity),
        stream_mode=["messages", "updates"],
    ):
        events.append(event)
    assert events and server.counts["commit"] == 0
    assert any(event == "context.estimate" for event, _ in samples)

    class BrokenModel(ToolLoopModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("test model failure")

    middleware2 = build_middleware(env)
    failed_agent = create_agent(
        model=BrokenModel(),
        middleware=[middleware2],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    from dataclasses import replace

    second = replace(identity, run_id="r2", session_id="s2")
    config = {"configurable": {"thread_id": second.session_id}}
    with pytest.raises(RuntimeError, match="model failure"):
        try:
            await failed_agent.ainvoke(
                {"messages": [HumanMessage(id="new", content="Read the mail")]},
                context=CallContext(second),
                config=config,
            )
        finally:
            await middleware2.acleanup_run(failed_agent, config)
    assert not middleware2._ov_session_engine.active
    assert (await failed_agent.aget_state(config)).values["ov_context"]["run_id"] is None
    await transport.http.aclose()
