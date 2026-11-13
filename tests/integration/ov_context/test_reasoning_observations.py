"""Capture provider reasoning even when the Run fails, without replaying it."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from examples.deepagents_ov_context.chat.model import TimedChatOpenAI
from tests.integration.ov_context.test_chat_api import parse_events


def stream_response(chunks):
    text = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=text + "data: [DONE]\n\n",
    )


def model(client, *, streaming=True):
    return TimedChatOpenAI(
        model="deepseek-v4-flash",
        api_key="test",
        base_url="http://model.invalid/v1",
        http_async_client=client,
        max_completion_tokens=16384,
        streaming=streaming,
        stream_usage=True,
        max_retries=0,
    )


def test_stream_parser_captures_in_metadata_without_changing_wire():
    instance = TimedChatOpenAI(model="deepseek-v4-flash", api_key="test")
    parts = []
    for delta in (
        {"role": "assistant", "reasoning_content": "check "},
        {"reasoning_content": "the inputs", "encrypted_content": "opaque"},
        {"content": "answer"},
    ):
        chunk = instance._convert_chunk_to_generation_chunk(
            {"choices": [{"delta": delta}]}, AIMessageChunk, None
        )
        assert chunk is not None
        parts.append(chunk.message)
    merged = parts[0] + parts[1] + parts[2]
    assert merged.content == "answer"
    assert merged.response_metadata["observed_reasoning_content"] == "check the inputs"
    assert merged.response_metadata["observed_encrypted_content_present"] is True
    assert "reasoning_content" not in merged.additional_kwargs
    observed = AIMessage(content="answer", response_metadata=merged.response_metadata)
    clean = AIMessage(content="answer")
    assert instance.preview_payload(
        [HumanMessage(content="q"), observed]
    ) == instance.preview_payload([HumanMessage(content="q"), clean])


async def test_non_streaming_reasoning_is_observation_only():
    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "test",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "check inputs",
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        instance = model(client, streaming=False)
        result = await instance.ainvoke([HumanMessage(content="q")])
    assert result.response_metadata["observed_reasoning_content"] == "check inputs"
    assert result.content == "answer" and "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize("ending", ["length", "timeout"])
async def test_failed_chat_persists_reasoning_by_model_call_id(chat, ending):
    client, runtime = chat

    class StalledReasoning(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"partial thought"}}]}\n\n'
            await asyncio.sleep(5)

    def respond(request):
        if ending == "timeout":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=StalledReasoning()
            )
        return stream_response(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "reasoning_content": "first thought; "},
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {"reasoning_content": "second thought"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 16384,
                        "total_tokens": 16404,
                        "completion_tokens_details": {"reasoning_tokens": 16384},
                    },
                },
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        runtime.model = model(http)
        response = await client.post(
            "/api/sessions",
            json={
                "execution": {
                    "mode": "baseline",
                    "memory_extraction": False,
                    "max_tokens": 16384,
                    "model_call_timeout_seconds": 0.05 if ending == "timeout" else 5,
                }
            },
        )
        response.raise_for_status()
        sid = response.json()["session_id"]
        response = await client.post("/api/chat", json={"session_id": sid, "message": "q"})
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error_type"] == (
        "ModelCallTimeoutError" if ending == "timeout" else "ModelOutputLimitError"
    )
    rid = next(data["run_id"] for event, data in events if event == "start")
    trace = (await client.get(f"/api/sessions/{sid}/runs/{rid}")).json()
    observed = [
        event["data"]
        for event in trace["events"]
        if event["event"] == "stage" and event["data"].get("name") == "model.reasoning"
    ]
    assert len(observed) == 1
    expected = "partial thought" if ending == "timeout" else "first thought; second thought"
    assert observed[0]["reasoning_content"] == expected
    assert observed[0]["status"] == ("partial" if ending == "timeout" else "complete")
    call = next(data["model_call_id"] for event, data in events if event == "model_input")
    assert observed[0]["model_call_id"] == call
    journal = runtime.event_log.path.read_text()
    records = [json.loads(line) for line in journal.splitlines()]
    assert any(
        row["event"] == "model.reasoning" and row["data"]["reasoning_content"] == expected
        for row in records
    )
    assert events[-1][1]["statistics"]["first_token_seconds"] is None
