"""Middleware estimation uses the actual model payload and local Session profile."""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI, windowed_model
from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    ContextBudgetExceeded,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.assembler import ContextAssembler
from openviking.integrations.langchain.ov_context.token_estimation import (
    payload_text_budget,
)


@tool
def echo_budget(text: str) -> str:
    """Echo text for budget verification."""
    return text


def request(messages, *, model=None, tools=None):
    return ModelRequest(
        model=model or ArkChatOpenAI(api_key="offline", max_tokens=1024),
        messages=messages,
        system_message=SystemMessage(content="保留系统提示及工具规则"),
        tools=tools or [],
        state={"messages": messages},
        runtime=SimpleNamespace(),
    )


def test_counter_matches_provider_preview_including_reasoning_and_tools():
    messages = [
        HumanMessage(content="检查中文预算"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "opaque中文" * 200},
            tool_calls=[{"id": "c", "name": "echo_budget", "args": {"text": "原文"}}],
        ),
        ToolMessage(content="原文", tool_call_id="c", artifact={"local-only": "x" * 100000}),
    ]
    r = request(messages, tools=[echo_budget])
    settings = r.model.bind_tools(r.tools, tool_choice=r.tool_choice).kwargs
    payload = r.model.preview_payload([r.system_message, *messages], **settings)
    assert payload["messages"][2]["reasoning_content"] == "opaque中文" * 200
    assert ApproximateTokenCounter()(r) == payload_text_budget(payload)
    assert "local-only" not in str(payload)
    messages[-1].artifact = None
    assert ApproximateTokenCounter()(r) == payload_text_budget(payload)


def test_planning_count_is_available_even_when_actual_send_is_over_budget():
    model = windowed_model(ArkChatOpenAI(api_key="offline", max_tokens=1024), 6000)
    r = request([HumanMessage(content="中文" * 10000)], model=model)
    assembler = ContextAssembler(
        OVContextConfig(
            context_window=6000, output_reserve=1024, base_context_ratio=0.2, capacity_ratio=0.5
        )
    )
    assert ApproximateTokenCounter()(r) > assembler.budget(r).input_limit
    with pytest.raises(ContextBudgetExceeded):
        assembler.check_input_budget(r)


def test_middleware_budget_includes_opaque_reasoning_added_to_ark_payload():
    model = windowed_model(ArkChatOpenAI(api_key="offline", max_tokens=1024), 6000)
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="ok", additional_kwargs={"reasoning_content": "x" * 30000}),
    ]
    assembler = ContextAssembler(
        OVContextConfig(
            context_window=6000, output_reserve=1024, base_context_ratio=0.2, capacity_ratio=0.5
        )
    )
    with pytest.raises(ContextBudgetExceeded):
        assembler.check_input_budget(request(messages, model=model))


def test_counter_does_not_treat_large_unsent_metadata_as_model_input():
    message = AIMessage(content="same", response_metadata={"local": "x" * 100000})
    large = ApproximateTokenCounter()(request([message]))
    message.response_metadata = {}
    assert ApproximateTokenCounter()(request([message])) == large


def test_window_profile_is_local_and_does_not_enlarge_known_limit():
    shared = ArkChatOpenAI(api_key="offline", profile={"max_input_tokens": 128000})
    new = windowed_model(shared, 256000)
    assert shared.profile["max_input_tokens"] == 128000
    assert new.profile["max_input_tokens"] == 128000
    assert new is not shared
