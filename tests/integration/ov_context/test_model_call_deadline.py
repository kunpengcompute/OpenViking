"""Model streams stay alive while data arrives and expire on idle gaps."""

import asyncio
import json
import time

import anyio
import httpx
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from examples.deepagents_ov_context.chat.model import (
    ArkChatOpenAI,
    ModelCallTimeoutError,
    TimedChatOpenAI,
)


class ReasoningStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.packets = 0

    async def __aiter__(self):
        while True:
            await asyncio.sleep(0.01)
            self.packets += 1
            delta = {"role": "assistant", "reasoning_content": "still thinking"}
            yield (
                "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"
            ).encode()

    async def aclose(self):
        self.closed = True


class ActivityStream(ReasoningStream):
    def __init__(self, delays, activity="reasoning"):
        super().__init__()
        self.delays, self.activity = delays, activity

    async def __aiter__(self):
        arguments = ["{", '"text"', ":", '"', "hel", "lo", '"', "}"]
        for i, delay in enumerate(self.delays):
            await asyncio.sleep(delay)
            self.packets += 1
            if self.activity == "tools":
                call = {"index": 0, "function": {"arguments": arguments[i]}}
                if i == 0:
                    call.update(id="call-test", type="function")
                    call["function"]["name"] = "echo"
                delta = {"tool_calls": [call]}
            else:
                field = "content" if self.activity == "text" else "reasoning_content"
                delta = {"role": "assistant", field: "still responding"}
            yield (
                "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"
            ).encode()
        finish = "tool_calls" if self.activity == "tools" else "stop"
        for packet in (
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        ):
            yield ("data: " + json.dumps(packet) + "\n\n").encode()
        yield b"data: [DONE]\n\n"


class StalledReasoningStream(ActivityStream):
    def __init__(self):
        super().__init__([0.01, 10])


@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
@pytest.mark.parametrize("method", ["astream", "ainvoke"])
@pytest.mark.parametrize("activity", ["text", "reasoning", "tools"])
async def test_active_stream_outlives_timeout_and_preserves_usage(model_type, method, activity):
    stream = ActivityStream([0.04] * 8, activity)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), timeout=90) as http:
        model = model_type(
            api_key="offline",
            http_async_client=http,
            streaming=True,
            async_call_timeout_seconds=0.2,
            max_retries=0,
        )
        started = time.monotonic()
        async with asyncio.timeout(2):
            if method == "astream":
                chunks = [item async for item in model.astream("question")]
                result = chunks[0]
                for chunk in chunks[1:]:
                    result += chunk
            else:
                result = await model.ainvoke("question")
        assert time.monotonic() - started > 0.2
        assert stream.closed and stream.packets == 8
        assert result.usage_metadata["total_tokens"] == 11
        if activity == "tools":
            assert result.tool_calls[0]["args"] == {"text": "hello"}
        assert len(requests) == 1
        assert "async_call_timeout_seconds" not in requests[0]
        assert "stream_chunk_timeout" not in requests[0]


@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
@pytest.mark.parametrize("method", ["astream", "ainvoke"])
@pytest.mark.parametrize("delays,packets", [([10], 0), ([0.01, 10], 1)])
async def test_first_packet_or_midstream_idle_times_out(model_type, method, delays, packets):
    stream = ActivityStream(delays)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
    ) as http:
        model = model_type(
            api_key="offline",
            http_async_client=http,
            streaming=True,
            async_call_timeout_seconds=0.08,
            max_retries=0,
        )
        with pytest.raises(ModelCallTimeoutError):
            async with asyncio.timeout(1):
                if method == "astream":
                    _ = [chunk async for chunk in model.astream("q")]
                else:
                    await model.ainvoke("q")
        assert stream.closed and stream.packets == packets
        assert model._input_budget._anchor is None


async def test_nonstream_first_response_is_also_bounded():
    async def respond(request):
        await asyncio.sleep(10)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = ArkChatOpenAI(
            api_key="offline",
            http_async_client=http,
            streaming=False,
            async_call_timeout_seconds=0.05,
            max_retries=0,
        )
        with pytest.raises(ModelCallTimeoutError):
            async with asyncio.timeout(1):
                await model.ainvoke("question")


async def test_external_cancellation_remains_cancelled_and_closes_stream():
    stream = ReasoningStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
    ) as http:
        model = ArkChatOpenAI(
            api_key="offline",
            http_async_client=http,
            streaming=True,
            async_call_timeout_seconds=5,
            max_retries=0,
        )
        task = asyncio.create_task(model.ainvoke([HumanMessage(content="q")]))
        while stream.packets == 0:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed


async def test_outer_timeout_stops_continuous_reasoning_and_closes_http():
    stream = ReasoningStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
    ) as http:
        model = ArkChatOpenAI(
            api_key="offline",
            http_async_client=http,
            streaming=True,
            async_call_timeout_seconds=0.08,
            max_retries=0,
        )
        with pytest.raises(TimeoutError) as error:
            with anyio.fail_after(0.2):
                await model.ainvoke("q")
        assert not isinstance(error.value, ModelCallTimeoutError)
        assert stream.packets > 1 and stream.closed


async def test_fast_nonstream_request_completes_and_keeps_encrypted_reasoning():
    response = {
        "id": "chat-test",
        "model": "doubao-seed-2-1-lite-260915",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "OK",
                    "reasoning_content": "summary",
                    "encrypted_content": "signed-original",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    ) as http:
        model = ArkChatOpenAI(
            api_key="offline",
            http_async_client=http,
            streaming=False,
            async_call_timeout_seconds=1,
            max_retries=0,
        )
        result = await model.ainvoke("q")
        assert result.content == "OK"
        assert result.additional_kwargs["encrypted_content"] == "signed-original"
        assert result.usage_metadata["total_tokens"] == 13


async def test_packet_winning_cancellation_race_cannot_bypass_idle_timeout(monkeypatch):
    closed = []

    async def lower_stream(self, *args, **kwargs):
        try:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Simulate wait_for returning a packet that completed at the
                # same instant as the deadline cancelled its caller.
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="", additional_kwargs={"reasoning_content": "late packet"}
                    )
                )
        finally:
            closed.append(True)

    monkeypatch.setattr(ChatOpenAI, "_astream", lower_stream)
    model = ArkChatOpenAI(api_key="offline", async_call_timeout_seconds=0.02)
    with pytest.raises(ModelCallTimeoutError):
        _ = [chunk async for chunk in model.astream("q")]
    assert closed == [True]


async def test_stream_idle_timeout_does_not_cancel_work_in_consumer(monkeypatch):
    async def lower_stream(self, *args, **kwargs):
        for _ in range(3):
            yield ChatGenerationChunk(message=AIMessageChunk(content="packet"))

    monkeypatch.setattr(ChatOpenAI, "_astream", lower_stream)
    model = ArkChatOpenAI(api_key="offline", async_call_timeout_seconds=0.02)
    consumed = []
    async for chunk in model.astream("q"):
        await asyncio.sleep(0.04)
        if chunk.text:
            consumed.append(chunk.text)
    assert consumed == ["packet"] * 3
