"""Signed reasoning survives provider streaming, cache/checkpoint and replay."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from openviking.integrations.langchain.ov_context import OVContext
from openviking.integrations.langchain.ov_context.token_estimation import (
    PayloadUsageCounter,
    serialized_text_budget,
)


def body():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "summary",
                    },
                }
            ]
        },
        # Ark sends one complete opaque block, separately from public text.
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "",
                        "reasoning_content": "",
                        "encrypted_content": "signed-original",
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "browser",
                                    "arguments": '{"argv":["snapshot","-i"]}',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        },
    ]
    return (
        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    ).encode()


def response(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_provider_stream_preserves_complete_ciphertext_once(asynchronous):
    if asynchronous:
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
            model = ArkChatOpenAI(api_key="offline", http_async_client=http, streaming=True)
            message = await model.ainvoke("q")
    else:
        with httpx.Client(transport=httpx.MockTransport(response)) as http:
            model = ArkChatOpenAI(api_key="offline", http_client=http, streaming=True)
            message = model.invoke("q")
    assert message.additional_kwargs == {
        "reasoning_content": "summary",
        "encrypted_content": "signed-original",
    }
    assert message.content == ""
    assert message.tool_calls[0]["id"] == "c1"
    assert message.usage_metadata["total_tokens"] == 25


async def test_sqlite_checkpoint_and_ovcontext_replay_keep_signed_field(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
        model = ArkChatOpenAI(api_key="offline", http_async_client=http, streaming=True)
        message = await model.ainvoke("q")
    message.id = "assistant-1"
    context = OVContext(session_id="s", messages=[message])
    restored_cache = OVContext.from_state(context.to_state())
    assert restored_cache.messages[0].additional_kwargs == message.additional_kwargs
    builder = StateGraph(MessagesState)
    builder.add_node("hold", lambda state: {})
    builder.add_edge(START, "hold")
    builder.add_edge("hold", END)
    config = {"configurable": {"thread_id": "s"}}
    path = str(tmp_path / "checkpoint.sqlite3")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({"messages": restored_cache.messages}, config)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        rebuilt = builder.compile(checkpointer=saver)
        restored = (await rebuilt.aget_state(config)).values["messages"][0]
    wire = ArkChatOpenAI(api_key="offline", max_completion_tokens=16384).preview_payload(
        [
            HumanMessage(content="q"),
            restored,
            ToolMessage(content="page", tool_call_id="c1"),
        ]
    )
    assert wire["messages"][1]["encrypted_content"] == "signed-original"
    assert wire["messages"][1]["reasoning_content"] == "summary"
    assert wire["max_completion_tokens"] == 16384 and "max_tokens" not in wire
    assert restored.additional_kwargs == message.additional_kwargs
    assert serialized_text_budget([restored]) > serialized_text_budget(
        [restored.model_copy(update={"additional_kwargs": {}})]
    )
    # Encrypted history is eligible for the same provider-confirmed input anchor.
    counter = PayloadUsageCounter()
    counter.observe(wire, 1)
    assert counter.estimate(wire) == 1 + 256


@pytest.mark.parametrize("field", ["reasoning_content", "encrypted_content"])
def test_invalid_opaque_fields_fail_before_request(field):
    from langchain_core.messages import AIMessage

    model = ArkChatOpenAI(api_key="offline")
    with pytest.raises(ValueError, match="field type"):
        model.preview_payload([AIMessage(content="answer", additional_kwargs={field: {"bad": 1}})])
