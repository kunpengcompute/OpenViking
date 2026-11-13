"""Image tool results remain multimodal through persistence and model views."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from openviking.integrations._message_content import LANGCHAIN_CONTENT_MIME
from openviking.integrations.langchain.ov_context.reader import restore_messages
from openviking.integrations.langchain.ov_context.tool_views import apply_tool_views
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking.message import ToolPart
from openviking.session.tool_views import add_messages_with_tool_views
from tests.ov_context_support.message_examples import exchange
from tests.ov_context_support.writer import SessionBatchClient, make_session


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "base64": "a" * 30000, "mime_type": "image/png"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "b" * 30000}},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "c" * 30000},
        },
    ],
    ids=["langchain-image", "openai-image-url", "anthropic-image"],
)
@pytest.mark.parametrize("mixed", [False, True], ids=["image-only", "text-and-image"])
async def test_images_survive_writer_server_and_reader(tmp_path, block, mixed):
    content = ([{"type": "text", "text": "页内说明 " * 1000}] if mixed else []) + [block]
    raw = exchange([content, "ordinary long text " * 1000], names=["read_file", "lookup"])
    before = deepcopy(raw)
    session = make_session(tmp_path)
    result = await OVMessageWriter(SessionBatchClient(session)).write(session.session_id, raw, 0)

    assert result.model_messages[2] is raw[2]
    assert result.model_messages[2].content == content
    assert result.model_messages[3].content.startswith("[OpenViking tool result externalized]")
    assert raw == before
    image_part = next(
        part
        for message in session.messages
        for part in message.parts
        if isinstance(part, ToolPart)
        and part.tool_id == "call_0"
        and part.tool_status == "completed"
    )
    assert image_part.tool_output_mime_type == LANGCHAIN_CONTENT_MIME
    assert json.loads(image_part.tool_output) == content
    assert image_part.tool_output_ref == "" and not image_part.tool_output_truncated
    restored = restore_messages([message.to_dict() for message in session.messages])
    image = next(
        message for message in restored if getattr(message, "tool_call_id", None) == "call_0"
    )
    assert image.content == content


async def test_writer_ignores_image_short_view_from_an_older_server():
    raw = exchange([[{"type": "image", "base64": "a" * 30000, "mime_type": "image/png"}]])
    client = AsyncMock()
    client.batch_add_messages.return_value = {
        "session_id": "s",
        "added": 3,
        "message_count": 3,
        "tool_views": [
            {
                "message_index": 2,
                "part_index": 0,
                "tool_id": "call_0",
                "tool_output": "image replaced by text",
                "tool_output_ref": "viking://user/alice/sessions/s/tool-results/ref",
            }
        ],
    }
    result = await OVMessageWriter(client).write("s", raw, 0)
    assert result.model_messages[2] is raw[2]


def test_view_application_cannot_replace_images_even_without_a_protection_hint():
    message = exchange(
        [[{"type": "image_url", "image_url": {"url": "https://example.com/p.png"}}]]
    )[2]
    messages = [message]
    apply_tool_views(
        messages,
        {
            "tool_views": [
                {
                    "message_index": 0,
                    "part_index": 0,
                    "tool_id": "call_0",
                    "tool_output": "incorrect textual view",
                    "tool_output_ref": "viking://user/alice/sessions/s/tool-results/ref",
                }
            ]
        },
        {(0, 0): 0},
        session_id="s",
        protected_indices=set(),
    )
    assert messages[0] is message


@pytest.mark.parametrize("mime", [LANGCHAIN_CONTENT_MIME, "image/png"])
def test_image_bytes_do_not_spend_the_text_compression_group_budget(tmp_path, mime):
    session = make_session(
        tmp_path, threshold_chars=100000, assistant_turn_inline_budget_chars=6000
    )
    output = (
        json.dumps([{"type": "image", "base64": "a" * 30000, "mime_type": "image/png"}])
        if mime == LANGCHAIN_CONTENT_MIME
        else "a" * 30000
    )
    image = ToolPart(
        tool_id="image",
        tool_name="read_file",
        tool_status="completed",
        tool_output=output,
        tool_output_mime_type=mime,
    )
    text = ToolPart(
        tool_id="text",
        tool_name="lookup",
        tool_status="completed",
        tool_output="short context",
    )
    receipt = add_messages_with_tool_views(session, [{"role": "user", "parts": [image, text]}])
    assert receipt["tool_views"] == []
    assert image.tool_output == output and not image.tool_output_ref
    assert text.tool_output == "short context" and not text.tool_output_ref
