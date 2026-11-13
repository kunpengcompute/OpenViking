"""Validate only outgoing OV fields, while preserving local native messages."""

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from openviking.integrations import _message_content
from openviking.integrations.langchain.ov_context import reader, writer
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from tests.ov_context_support.message_examples import exchange


def test_writer_codec_is_the_shared_entry_point() -> None:
    assert writer.encode_tool_content is _message_content.encode_tool_content
    assert reader.decode_tool_content is _message_content.decode_tool_content


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), object()])
@pytest.mark.parametrize("field", ["args", "tool_content", "image"])
async def test_invalid_outgoing_json_in_later_batch_prevents_all_writes(value, field) -> None:
    raw: list[BaseMessage] = [HumanMessage(content=str(i)) for i in range(101)]
    if field == "args":
        raw.append(
            AIMessage(content="", tool_calls=[{"id": "c", "name": "lookup", "args": {"x": value}}])
        )
    elif field == "tool_content":
        raw.extend(exchange([[{"type": "text", "text": "result", "x": value}]]))
    else:
        raw.append(
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a", "detail": value},
                    }
                ]
            )
        )
    client = AsyncMock()
    with pytest.raises((ValueError, TypeError)):
        await OVMessageWriter(client).write("s", raw, 0)
    client.batch_add_messages.assert_not_awaited()


async def test_unsent_metadata_does_not_block_writes_or_change_local_objects() -> None:
    raw = exchange(["result"])
    marker = object()
    raw[0].response_metadata = {"nan": float("nan"), "local": marker}
    raw[1].usage_metadata = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    raw[-1].artifact = {"infinity": float("inf"), "local": marker}
    raw[-1].response_metadata = {"local": marker}
    client = AsyncMock()
    client.batch_add_messages.return_value = {"session_id": "s", "added": 3, "message_count": 3}
    result = await OVMessageWriter(client).write("s", raw, 0)
    assert all(
        returned is original for returned, original in zip(result.model_messages, raw, strict=True)
    )
    assert (result.write_index, result.written_count) == (3, 3)
    payload = client.batch_add_messages.await_args.kwargs["messages"]
    assert [set(spec) for spec in payload] == [
        {"role", "parts", "message_kind", "source_message_ids"},
        {"role", "parts"},
        {"role", "parts"},
    ]
    assert payload[0]["message_kind"] == "user_query"
    assert payload[0]["source_message_ids"] == [raw[0].id]
    assert set(payload[-1]["parts"][0]) == {
        "type",
        "tool_id",
        "tool_name",
        "tool_input",
        "tool_output",
        "tool_output_mime_type",
        "tool_status",
    }
    json.dumps({"messages": payload}, allow_nan=False)
    assert raw[0].response_metadata["local"] is marker
    assert raw[-1].artifact["local"] is marker
