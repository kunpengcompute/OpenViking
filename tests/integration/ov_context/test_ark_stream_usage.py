"""Cumulative SSE usage snapshots are retained once for every Chat adapter."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI, TimedChatOpenAI


def stream_body(mode, prompt=73):
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": 4,
        "total_tokens": prompt + 4,
        "prompt_tokens_details": {"cached_tokens": 32},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    base = {
        "id": "chat-test",
        "model": "doubao-seed-2-1-lite-260915",
        "object": "chat.completion.chunk",
    }
    chunks = [
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if mode in {"finish", "both", "updated"}:
        chunks[-1]["usage"] = (
            {"prompt_tokens": prompt, "completion_tokens": 2, "total_tokens": prompt + 2}
            if mode == "updated"
            else usage
        )
    if mode in {"usage", "both", "updated"}:
        chunks.append({**base, "choices": [], "usage": usage})
    return (
        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    ).encode()


def response(mode, request):
    body = json.loads(request.content)
    prompt = 91 if body["messages"][-1]["content"] == "second" else 73
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=stream_body(mode, prompt)
    )


@pytest.mark.parametrize("mode", ["finish", "usage", "both", "updated", "none"])
@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
def test_sync_usage_is_final_snapshot_not_sum(mode, model_type):
    with httpx.Client(transport=httpx.MockTransport(lambda r: response(mode, r))) as client:
        model = model_type(
            api_key="offline",
            model="doubao-seed-2-1-lite-260915",
            http_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        message = model.invoke([HumanMessage(content="first")])
    assert message.content == "OK"
    if mode == "none":
        assert message.usage_metadata is None
    else:
        assert message.usage_metadata["input_tokens"] == 73
        assert message.usage_metadata["output_tokens"] == 4
        assert message.usage_metadata["total_tokens"] == 77
        assert message.usage_metadata["input_token_details"]["cache_read"] == 32
        assert message.usage_metadata["output_token_details"]["reasoning"] == 2
        assert model._input_budget._anchor.input_tokens == 73
    assert message.response_metadata["finish_reason"] == "stop"


@pytest.mark.parametrize("mode", ["finish", "usage", "both", "updated", "none"])
@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
async def test_async_usage_is_final_snapshot_not_sum(mode, model_type):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: response(mode, r))
    ) as client:
        model = model_type(
            api_key="offline",
            model="doubao-seed-2-1-lite-260915",
            http_async_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        message = await model.ainvoke([HumanMessage(content="first")])
    assert message.content == "OK"
    if mode == "none":
        assert message.usage_metadata is None
    else:
        assert message.usage_metadata["input_tokens"] == 73
        assert message.usage_metadata["output_tokens"] == 4
        assert message.usage_metadata["total_tokens"] == 77
        assert message.usage_metadata["input_token_details"]["cache_read"] == 32
        assert message.usage_metadata["output_token_details"]["reasoning"] == 2
        assert model._input_budget._anchor.input_tokens == 73


@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
async def test_parallel_model_requests_keep_usage_separate(model_type):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: response("both", r))
    ) as client:
        model = model_type(
            api_key="offline",
            model="doubao-seed-2-1-lite-260915",
            http_async_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        messages = await asyncio.gather(
            *(model.ainvoke([HumanMessage(content=word)]) for word in ("first", "second"))
        )
    assert [m.usage_metadata["input_tokens"] for m in messages] == [73, 91]


@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
async def test_usage_normalization_preserves_parallel_tool_calls(model_type):
    usage = {"prompt_tokens": 73, "completion_tokens": 12, "total_tokens": 85}
    chunks = [
        {
            "id": "parallel",
            "model": "doubao-seed-2-1-lite-260915",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": i,
                                "id": f"c-{i}",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": json.dumps({"text": str(i)}),
                                },
                            }
                            for i in range(2)
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "parallel",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": usage,
        },
        {"id": "parallel", "choices": [], "usage": usage},
    ]
    body = (
        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    ).encode()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )
        )
    ) as client:
        model = model_type(
            api_key="offline",
            model="doubao-seed-2-1-lite-260915",
            http_async_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        message = await model.ainvoke([HumanMessage(content="two independent calls")])
    assert [c["id"] for c in message.tool_calls] == ["c-0", "c-1"]
    assert not message.invalid_tool_calls
    assert message.response_metadata["finish_reason"] == "tool_calls"
    assert message.usage_metadata["total_tokens"] == 85
