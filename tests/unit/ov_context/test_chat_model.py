"""Verify the actual Ark payload and streaming reasoning round trip offline."""

import json

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI


def test_ark_wire_keeps_total_limit_and_replays_opaque_reasoning() -> None:
    model = ArkChatOpenAI(model="deepseek-v4-flash", api_key="test", max_completion_tokens=16384)
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "opaque-provider-field"},
            tool_calls=[{"id": "c1", "name": "browser", "args": {"argv": ["links"]}}],
        ),
        ToolMessage(content="observed page", tool_call_id="c1"),
    ]
    payload = model._get_request_payload(messages, stream=True)
    assert payload["max_completion_tokens"] == 16384 and "max_tokens" not in payload
    assert payload["messages"][1]["reasoning_content"] == "opaque-provider-field"
    assert payload["messages"][1]["content"] is None
    assert payload["messages"][2]["tool_call_id"] == "c1"


def test_streaming_reasoning_stays_out_of_answer_content() -> None:
    model = ArkChatOpenAI(model="deepseek-v4-flash", api_key="test")
    chunks = []
    for delta in (
        {"role": "assistant", "reasoning_content": "opaque-"},
        {"reasoning_content": "field"},
        {"content": "public answer"},
    ):
        chunk = model._convert_chunk_to_generation_chunk(
            {"choices": [{"delta": delta}]}, AIMessageChunk, None
        )
        assert chunk is not None
        chunks.append(chunk.message)
    merged = chunks[0] + chunks[1] + chunks[2]
    assert merged.content == "public answer"
    assert merged.additional_kwargs["reasoning_content"] == "opaque-field"


async def test_sdk_http_body_contains_enabled_thinking_and_ark_limit() -> None:
    captured: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "opaque"}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body + "data: [DONE]\n\n"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ArkChatOpenAI(
            model="deepseek-v4-flash",
            api_key="test",
            base_url="http://model.invalid/v1",
            http_async_client=client,
            max_completion_tokens=16384,
            extra_body={"thinking": {"type": "enabled"}},
        )
        chunks = [chunk async for chunk in model.astream([HumanMessage(content="question")])]
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["max_completion_tokens"] == 16384 and "max_tokens" not in captured[0]
    assert "".join(chunk.text for chunk in chunks) == "answer"
    assert any(chunk.additional_kwargs.get("reasoning_content") == "opaque" for chunk in chunks)
