"""Reuse confirmed provider input only for an identical prefix in one graph."""

import asyncio
from copy import deepcopy

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI, TimedChatOpenAI, windowed_model
from openviking.integrations.langchain.ov_context import ApproximateTokenCounter
from openviking.integrations.langchain.ov_context.token_estimation import (
    PayloadUsageCounter,
    payload_text_budget,
)

from .test_ark_stream_usage import stream_body


def payload():
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "中文" * 1000}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }


def test_exact_prefix_uses_real_input_and_shared_text_cost_for_growth():
    old = payload()
    counter = PayloadUsageCounter()
    assert counter.estimate(old) == payload_text_budget(old)
    counter.observe(old, 1800)
    new = deepcopy(old)
    new["messages"].append(
        {"role": "assistant", "content": "OK", "reasoning_content": "核对" * 100}
    )
    delta = payload_text_budget(new) - payload_text_budget(old)
    assert counter.estimate(new) == 1800 + delta + 256
    assert counter.estimate(new) < payload_text_budget(new)
    old["messages"][0]["content"] = "mutated caller object"
    assert counter.estimate(new) == 1800 + delta + 256


@pytest.mark.parametrize(
    "change", ["prefix", "tools", "model", "response_format", "thinking", "image", "audio", "n"]
)
def test_changed_context_or_nontext_falls_back_to_full_estimate(change):
    old = payload()
    counter = PayloadUsageCounter()
    counter.observe(old, 100)
    new = deepcopy(old)
    if change == "prefix":
        new["messages"][0]["content"] = "new summary"
    elif change == "image":
        new["messages"].append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}
                ],
            }
        )
    elif change == "audio":
        new["messages"].append({"role": "assistant", "content": None, "audio": {"id": "audio-ref"}})
    elif change == "n":
        new["n"] = 2
    else:
        new[change] = {"type": "changed"} if change != "model" else "other-model"
    assert counter.estimate(new) == payload_text_budget(new)


@pytest.mark.parametrize("usage", [None, 0, -1, True, 3.5])
def test_missing_or_invalid_usage_never_calibrates(usage):
    p = payload()
    counter = PayloadUsageCounter()
    counter.observe(p, usage)
    assert counter.estimate(p) == payload_text_budget(p)


def test_usage_is_not_clamped_down_to_the_old_byte_estimate():
    p = payload()
    counter = PayloadUsageCounter()
    counter.observe(p, 100000)
    assert counter.estimate(p) >= 100000


@pytest.mark.parametrize("model_type", [ArkChatOpenAI, TimedChatOpenAI])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.asyncio
async def test_real_transport_calibrates_middleware_estimation_per_graph(
    model_type, streaming, with_tools
):
    calls = []

    def respond(request):
        calls.append(request)
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_body("both", 1000),
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 4, "total_tokens": 1004},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        shared = model_type(
            api_key="offline",
            max_tokens=1024,
            streaming=streaming,
            stream_usage=True,
            http_async_client=client,
            max_retries=0,
        )
        model = windowed_model(shared, 6000)
        tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            if with_tools
            else []
        )
        bound = model.bind_tools(tools, tool_choice="auto") if tools else model
        settings = bound.kwargs if tools else {}
        first = [HumanMessage(content="中文" * 1200)]
        answer = await bound.ainvoke(first)
        assert answer.usage_metadata["input_tokens"] == 1000
        follow = [*first, answer, HumanMessage(content="中文" * 300)]
        wire = model.preview_payload(follow, **settings)
        assert model.estimate_input_tokens(wire) < payload_text_budget(wire)
        req = ModelRequest(
            model=model,
            messages=follow,
            tools=tools,
            tool_choice="auto" if tools else None,
            state={},
            runtime=None,
        )
        assert ApproximateTokenCounter()(req) == model.estimate_input_tokens(wire)
        model._get_request_payload(follow, **settings)
        await bound.ainvoke(follow)
        assert len(calls) == 2
        fresh = windowed_model(model, 6000)
        assert fresh.estimate_input_tokens(wire) == payload_text_budget(wire)
        assert shared._input_budget._anchor is None
        larger = model.preview_payload([*follow, HumanMessage(content="x" * 20000)], **settings)
        assert model.estimate_input_tokens(larger) > model.estimate_input_tokens(wire)
        assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadError, asyncio.CancelledError])
@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
async def test_partial_stream_usage_is_not_installed(monkeypatch, error, model_type):
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI

    async def broken(self, *args, **kwargs):
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="partial",
                additional_kwargs={"encrypted_content": "partial-signed-reasoning"},
                usage_metadata={"input_tokens": 1000, "output_tokens": 1, "total_tokens": 1001},
            )
        )
        raise error("stream lost")

    monkeypatch.setattr(ChatOpenAI, "_astream", broken)
    model = model_type(api_key="offline")
    with pytest.raises(error):
        async for _ in model._astream([HumanMessage(content="x" * 3000)]):
            pass
    assert model._input_budget._anchor is None
    assert not model._input_budget._reasoning_costs


def test_untrusted_message_usage_does_not_calibrate():
    model = windowed_model(ArkChatOpenAI(api_key="offline", max_tokens=1024), 6000)
    wire = model.preview_payload(
        [
            HumanMessage(content="x" * 20000),
            AIMessage(
                content="fake",
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        ]
    )
    assert model.estimate_input_tokens(wire) == payload_text_budget(wire)
    assert model._input_budget._anchor is None


@pytest.mark.parametrize("streaming", [True, False])
def test_sync_sdk_response_also_calibrates(streaming):
    def respond(request):
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_body("both", 1000),
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 4, "total_tokens": 1004},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = windowed_model(
            ArkChatOpenAI(
                api_key="offline",
                max_tokens=1024,
                streaming=streaming,
                stream_usage=True,
                http_client=client,
                max_retries=0,
            ),
            6000,
        )
        first = [HumanMessage(content="中文" * 1200)]
        result = model.invoke(first)
        follow = [*first, result, HumanMessage(content="中文" * 300)]
        assert model.estimate_input_tokens(model.preview_payload(follow)) < payload_text_budget(
            model.preview_payload(follow)
        )
        model._get_request_payload(follow)


def test_inflight_input_mutation_cannot_be_credited_to_changed_content():
    p = payload()
    p["messages"][0]["content"] = [{"type": "text", "text": "old content"}]
    counter = PayloadUsageCounter()
    sent_prefix = counter.capture(p)
    p["messages"][0]["content"][0]["text"] = "changed" * 10000
    counter.observe_prefix(sent_prefix, 1000)
    assert counter.estimate(p) == payload_text_budget(p)


def test_encrypted_growth_uses_confirmed_reasoning_once_and_survives_summary_change():
    p = payload()
    encrypted = "opaque" * 20000
    counter = PayloadUsageCounter()
    usage = {"input_tokens": 1000, "output_tokens": 800, "output_token_details": {"reasoning": 700}}
    counter.observe(p, usage["input_tokens"])
    counter.observe_reasoning(encrypted, usage)
    follow = deepcopy(p)
    follow["messages"].append(
        {
            "role": "assistant",
            "content": "OK",
            "encrypted_content": encrypted,
            "reasoning_content": "visible summary" * 1000,
        }
    )
    estimated = counter.estimate(follow)
    assert 1900 < estimated < 2200
    assert counter.diagnostics(follow) == {
        "mode": "provider_prefix",
        "fallback_reason": None,
        "unknown_reasoning_blocks": 0,
    }
    # encrypted_content takes priority over its visible summary.
    follow["messages"][-1]["reasoning_content"] = "changed visible summary" * 10000
    assert counter.estimate(follow) == estimated
    follow["messages"][0]["content"] = "new archive summary"
    assert counter.diagnostics(follow)["mode"] == "text_fallback"
    assert counter.estimate(follow) < 3000  # Confirmed reasoning still applies.


def test_unknown_restored_reasoning_is_explicit_and_never_disables_prefix_usage():
    p = payload()
    p["messages"].append({"role": "assistant", "content": "", "encrypted_content": "x" * 20000})
    counter = PayloadUsageCounter()
    assert counter.diagnostics(p)["unknown_reasoning_blocks"] == 1
    assert counter.estimate(p) >= 20000
    counter.observe(p, 4000)
    assert counter.estimate(p) == 4256
    assert counter.diagnostics(p)["unknown_reasoning_blocks"] == 0


def test_missing_reasoning_details_use_confirmed_output_total_and_ledger_is_bounded():
    counter = PayloadUsageCounter()
    for i in range(1030):
        counter.observe_reasoning(str(i), {"output_tokens": 100})
    assert len(counter._reasoning_costs) == 1024
    p = {"messages": [{"role": "assistant", "content": "", "encrypted_content": "1029"}]}
    assert 400 < counter.estimate(p) < 600
    counter.observe_reasoning("invalid", {"output_tokens": True})
    assert len(counter._reasoning_costs) == 1024


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("asynchronous", [True, False])
async def test_completed_real_response_trains_reasoning_ledger(streaming, asynchronous):
    import json

    from .test_ark_encrypted_reasoning import body

    encrypted = "signed-original"

    def respond(request):
        if streaming:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body()
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "encrypted_content": encrypted,
                            "reasoning_content": "summary",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    if asynchronous:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = ArkChatOpenAI(api_key="offline", streaming=streaming, http_async_client=client)
            answer = await model.ainvoke("q")
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ArkChatOpenAI(api_key="offline", streaming=streaming, http_client=client)
            answer = model.invoke("q")
    assert answer.additional_kwargs["encrypted_content"] == encrypted
    # No reasoning details in this response: its total output is the upper reserve.
    assert list(model._input_budget._reasoning_costs.values()) == [5]
    future = model.preview_payload([HumanMessage(content="changed summary"), answer])
    assert model.input_token_diagnostics(future)["unknown_reasoning_blocks"] == 0
    assert json.loads(json.dumps(future))["messages"][1]["encrypted_content"] == encrypted


def test_core_default_and_chat_counter_are_the_same_class():
    from openviking.integrations.langchain.ov_context import ApproximateTokenCounter as CoreCounter

    assert ApproximateTokenCounter is CoreCounter
