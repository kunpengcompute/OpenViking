"""Chat delegates budgeting to OV assembly or DeepAgents native summarization."""

from dataclasses import replace

import pytest
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.deepagents_ov_context.chat.model import windowed_model
from examples.deepagents_ov_context.chat.native_summary import native_summary
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile
from openviking.integrations.langchain.ov_context import ApproximateTokenCounter


@pytest.mark.parametrize(
    "mode,enabled,eviction_limit",
    [("baseline", False, None), ("baseline", True, 5000)],
)
async def test_native_modes_use_sdk_summary_defaults(
    chat, monkeypatch, mode, enabled, eviction_limit
):
    from examples.deepagents_ov_context.chat import server as chat_server
    from examples.deepagents_ov_context.chat.server import ActiveRun, SessionInfo

    _, rt = chat
    captured = {}
    monkeypatch.setattr(chat_server, "create_deep_agent", lambda **kw: captured.update(kw))
    session = SessionInfo(
        session_id="native-budget",
        account_id=rt.settings.account_id,
        user_id="u",
        agent_id="p",
        created_at=0,
        execution=ExecutionProfile(
            mode=mode,
            context_window=256000,
            max_tokens=16384,
            native_tool_compression=enabled,
        ),
    )
    session.config_snapshot = rt.snapshot(session)
    sdk = rt.sdk()
    try:
        rt.graph(sdk, ActiveRun(), session=session)
        files = [m for m in captured["middleware"] if isinstance(m, FilesystemMiddleware)]
        assert len(files) == 1
        assert files[0]._tool_token_limit_before_evict == eviction_limit
        summaries = [m for m in captured["middleware"] if isinstance(m, SummarizationMiddleware)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.token_counter.func is count_tokens_approximately
        assert summary.token_counter.keywords == {"use_usage_metadata_scaling": True}
        assert summary._lc_helper.trigger == ("fraction", 0.85)
        assert summary._lc_helper.keep == ("fraction", 0.1)
        assert summary._lc_helper.trim_tokens_to_summarize is None
        assert captured["model"].profile["max_input_tokens"] == 256000
        assert rt.model.profile is None
    finally:
        await sdk.close()


@pytest.mark.parametrize("mode", ["current", "context_only", "ordinary", "experience"])
async def test_ov_modes_use_middleware_counter_and_configured_margin(chat, monkeypatch, mode):
    from examples.deepagents_ov_context.chat import server as chat_server
    from examples.deepagents_ov_context.chat.server import ActiveRun, SessionInfo

    _, rt = chat
    rt.settings = replace(
        rt.settings, context_config=replace(rt.settings.context_config, safety_margin=777)
    )
    captured = {}
    monkeypatch.setattr(chat_server, "create_deep_agent", lambda **kw: captured.update(kw))
    session = SessionInfo(
        session_id="ov-budget",
        account_id=rt.settings.account_id,
        user_id="u",
        agent_id="p",
        created_at=0,
        execution=ExecutionProfile(mode=mode, context_window=256000, max_tokens=16384),
    )
    session.config_snapshot = rt.snapshot(session)
    assert session.config_snapshot.context_config.safety_margin == 777
    sdk = rt.sdk()
    try:
        _, middleware = rt.graph(sdk, ActiveRun(), session=session)
        assembler = middleware._ov_session_engine.assembler
        assert type(assembler.token_counter) is ApproximateTokenCounter
        assert assembler.config.safety_margin == 777
        assert assembler.config.context_window == 256000
        assert assembler.config.output_reserve == 16384
        assert (
            assembler.config.capacity_ratio == session.config_snapshot.context_config.capacity_ratio
        )
        assert (
            assembler.config.base_context_ratio
            == session.config_snapshot.context_config.base_context_ratio
        )
        assert not any(isinstance(m, SummarizationMiddleware) for m in captured["middleware"])
        assert sum(m.name == "SummarizationMiddleware" for m in captured["middleware"]) == 1
    finally:
        await sdk.close()


def test_native_tool_compression_switch_is_explicit_and_defaults_on():
    assert ExecutionProfile(mode="baseline").native_tool_compression is True
    assert (
        ExecutionProfile(mode="baseline", native_tool_compression=False).native_tool_compression
        is False
    )
    with pytest.raises(ValueError):
        ExecutionProfile(mode="baseline", native_tool_compression="false")


class SummaryModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "native-summary-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="condensed result"))]
        )


class InputRecorder(AsyncCallbackHandler):
    def __init__(self):
        self.inputs = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.inputs.extend(messages)


async def test_native_graph_compacts_large_tool_history_with_sdk_counter():
    model = windowed_model(SummaryModel(), 32000)
    messages = [HumanMessage(content="Find the daily report")]
    for i in range(3):
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "grep", "args": {"pattern": "report"}, "id": str(i)}],
                ),
                ToolMessage(content="report data " * 15000, tool_call_id=str(i), name="grep"),
            ]
        )
    assert count_tokens_approximately(messages) > 32000 * 0.85
    graph = create_deep_agent(
        model=model,
        middleware=[
            FilesystemMiddleware(tool_token_limit_before_evict=None),
            native_summary(model),
        ],
    )
    recorder = InputRecorder()
    result = await graph.ainvoke({"messages": messages}, config={"callbacks": [recorder]})
    assembled = [
        batch
        for batch in recorder.inputs
        if any("conversation that has been summarized" in str(m.content) for m in batch)
    ]
    assert assembled
    assert count_tokens_approximately(assembled[-1]) < count_tokens_approximately(messages)
    assert any("conversation_history" in path for path in result.get("files", {}))
    assert result["messages"][-1].content == "condensed result"
    assert len(result["messages"]) == len(messages) + 1
