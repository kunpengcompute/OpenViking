"""Requirements exercised against the assembler used by the production hooks."""

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    ContextAssembler,
    ContextBudgetExceeded,
    OVContext,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.assembler import BATCH_SEARCH_GUIDANCE


def assembler(counter):
    return ContextAssembler(
        OVContextConfig(
            context_window=200, output_reserve=100, safety_margin=0, capacity_ratio=0.4
        ),
        counter,
    )


def test_batch_guidance_follows_registered_tools_and_is_not_duplicated(env):
    @tool("openviking_tool_result_search_many", description="Test batch search")
    def lookup() -> str:
        return "unused"

    context = OVContext("s", messages=[HumanMessage(id="u", content="Read original")])
    request = env[4].override(system_message=SystemMessage(content="Host rules"), tools=[lookup])
    instance = assembler(lambda request: 10)
    first = instance.assemble(request, context)
    second = instance.assemble(first, context)
    assert second.system_message.content.count(BATCH_SEARCH_GUIDANCE) == 1
    assert second.system_message.content.startswith("Host rules")
    without_tool = instance.assemble(second.override(tools=[]), context)
    assert BATCH_SEARCH_GUIDANCE not in without_tool.system_message.content
    assert request.system_message.content == "Host rules"


def test_outgoing_multimodal_messages_and_system_are_isolated(env):
    request = env[4]
    user = HumanMessage(
        id="u",
        content=[
            {"type": "text", "text": "question"},
            {"type": "image_url", "image_url": {"url": "image"}},
        ],
        additional_kwargs={"source": {"name": "host"}},
    )
    system = SystemMessage(content=[{"type": "text", "text": "Original system"}])
    context = OVContext("s", messages=[user])
    before = deepcopy(context.to_state())
    updated = (
        assembler(lambda request: 10)
        .prepare(request.override(system_message=system), context)
        .request
    )

    updated.messages[0].content[0]["text"] = "downstream mutation"
    updated.messages[0].additional_kwargs["source"]["name"] = "changed"
    updated.system_message.content[0]["text"] = "changed"
    assert context.to_state() == before
    assert system.content == [{"type": "text", "text": "Original system"}]


def test_recall_is_preserved_and_full_input_is_rejected_when_over_budget(env):
    user = HumanMessage(id="u", content="Current task")
    context = OVContext("s", messages=[user])
    context.begin_run("r", user)
    context.inject_recall("Large optional reference", "success")
    before = deepcopy(context.to_state())

    def count(request):
        return 200 if request.messages else 10

    instance = assembler(count)
    assert instance.history_budget(env[4]).history_target_tokens == 50
    with pytest.raises(ContextBudgetExceeded):
        instance.prepare(env[4], context)
    assert context.to_state() == before
    assert "[参考记忆]" in context.messages[0].content


def test_required_history_is_never_silently_trimmed(env):
    user = HumanMessage(id="u", content="Current task")
    context = OVContext("s", messages=[HumanMessage(id="old", content="History"), user])
    context.begin_run("r", user)
    before = deepcopy(context.to_state())
    with pytest.raises(ContextBudgetExceeded):
        assembler(lambda request: 200).prepare(env[4], context)
    assert context.to_state() == before


def test_model_messages_come_only_from_ov_context(env):
    user = HumanMessage(id="u", content="Cached task")
    context = OVContext("s", messages=[user])
    request = env[4].override(messages=[HumanMessage(id="foreign", content="Pollution")])
    prepared = assembler(lambda request: 10).prepare(request, context)
    assert [message.id for message in prepared.request.messages] == ["u"]
    assert request.messages[0].content == "Pollution"


def test_tool_fields_and_artifact_survive_isolated_assembly(env):
    messages = [
        HumanMessage(id="u", content="Read"),
        AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {"x": 1}}]),
        ToolMessage(
            id="t",
            tool_call_id="c",
            content=[{"type": "text", "text": "Result"}],
            artifact={"original": ["full result"]},
            status="success",
        ),
    ]
    context = OVContext("s", messages=messages)
    prepared = assembler(lambda request: 10).prepare(env[4], context)
    assert prepared.request.messages == messages
    prepared.request.messages[-1].artifact["original"].append("Changed")
    prepared.request.messages[1].tool_calls[0]["args"]["x"] = 2
    assert messages[-1].artifact == {"original": ["full result"]}
    assert messages[1].tool_calls[0]["args"] == {"x": 1}


@pytest.mark.parametrize(
    "messages,reason",
    [
        ([HumanMessage(id="u", content="a"), HumanMessage(id="u", content="b")], "unique"),
        ([HumanMessage(content="Missing ID")], "unique"),
        ([ToolMessage(id="t", content="orphan", tool_call_id="c")], "Orphan"),
        (
            [AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}])],
            "unfinished",
        ),
    ],
)
def test_invalid_messages_are_rejected_without_mutating_cache(env, messages, reason):
    context = OVContext("s", messages=messages)
    before = deepcopy(context.to_state())
    with pytest.raises(ValueError, match=reason):
        assembler(lambda request: 10).prepare(env[4], context)
    assert context.to_state() == before


def test_full_request_counter_includes_system_and_tool_schema(env):
    counter = ApproximateTokenCounter()
    request = env[4]
    base = counter(request)
    larger = request.override(
        system_message=SystemMessage(content="System " * 100),
        tools=[{"type": "function", "function": {"name": "read", "description": "Tool " * 100}}],
    )
    assert counter(larger) > base
