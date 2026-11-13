import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    OVContextConfig,
    RunIdentity,
)
from openviking.integrations.langchain.ov_context.reader import OVContextReader as NativeReader
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from tests.ov_context_support.writer import SessionBatchClient, make_session

from .test_graph import CallContext, ToolLoopModel
from .test_io_contracts import SuppliedReader


@pytest.mark.parametrize("backend", ["local", "http"])
@pytest.mark.parametrize("framework", ["langchain", "deepagents"])
async def test_real_framework_loop_uses_short_view_and_checkpoints_raw_messages(
    tmp_path, framework, backend, native_api
):
    if framework == "deepagents":
        from deepagents import create_deep_agent

        factory = create_deep_agent
    else:
        factory = create_agent
    if backend == "http":
        session, client = native_api
        reader = NativeReader(client)
    else:
        session = make_session(tmp_path)
        client = SessionBatchClient(session)
        reader = SuppliedReader()
    model = ToolLoopModel()
    raw_output = "订单已发货，预计明天送达。\n" * 800

    @tool
    async def read_mail() -> str:
        """Read the full order notice."""
        return raw_output

    middleware = OpenVikingContextMiddleware(
        client=client,
        message_writer=OVMessageWriter(client),
        context_reader=reader,
        context_config=OVContextConfig(context_window=100000),
        session_id_resolver=lambda state, runtime: session.session_id,
        peer_id="test-peer",
        token_counter=ApproximateTokenCounter(),
    )
    saver = InMemorySaver()
    graph = factory(
        model=model,
        tools=[read_mail],
        middleware=[middleware],
        checkpointer=saver,
        context_schema=CallContext,
    )
    config = {"configurable": {"thread_id": "writer-loop"}}
    identity = RunIdentity("a", "u", "test-peer", session.session_id, "writer-loop")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="查询订单通知")]}, config, context=CallContext(identity)
    )
    assert result["messages"][-1].content == "Mail summarized"
    assert len(model.inputs) == 2
    model_tool = next(m for m in model.inputs[-1] if isinstance(m, ToolMessage))
    graph_tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert graph_tool.content == raw_output
    assert model_tool.content != raw_output and "viking://" in model_tool.content
    assert model_tool.model_dump(exclude={"content"}) == graph_tool.model_dump(exclude={"content"})
    saved = (await graph.aget_state(config)).values
    assert next(m for m in saved["messages"] if isinstance(m, ToolMessage)).content == raw_output
    assert saved["ov_context"]["write_index"] == len(saved["messages"])
    assert len(session.messages) == 4
    if backend == "local":
        assert len(client.requests) == 4  # Four writing hooks, no duplicates.
