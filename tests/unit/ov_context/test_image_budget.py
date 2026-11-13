"""Encoded image size is transport cost, not model text-token consumption."""

from copy import deepcopy

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context.assembler import ContextAssembler
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.token_estimation import payload_text_budget


@pytest.mark.parametrize("kind", ["image", "image_url", "input_image"])
def test_image_transport_length_does_not_inflate_token_estimate(kind):
    def payload(size):
        block = (
            {"type": kind, "image_url": {"url": "data:image/png;base64," + "a" * size}}
            if kind == "image_url"
            else {"type": kind, "base64": "a" * size, "mime_type": "image/png"}
        )
        return {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "c",
                    "content": [
                        {"type": "text", "text": "检查当前页面"},
                        block,
                    ],
                }
            ]
        }

    small, large = payload(1000), payload(1000000)
    before = deepcopy(large)
    assert payload_text_budget(small) == payload_text_budget(large)
    assert large == before
    assert payload_text_budget(large, tokens_per_image=2048) - payload_text_budget(large) == 1024


def test_image_estimate_preserves_visible_text_and_tool_fields():
    payload = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": [
                    {"type": "text", "text": "核对金额"},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/page.png"}},
                ],
            }
        ]
    }
    text_only = deepcopy(payload)
    text_only["messages"][0]["content"].pop()
    assert payload_text_budget(payload) == payload_text_budget(text_only) + 1024
    larger = deepcopy(payload)
    larger["messages"][0]["content"][0]["text"] += "新增正文" * 100
    assert payload_text_budget(larger) > payload_text_budget(payload)


def test_six_large_image_results_are_admitted_by_the_real_assembler_counter():
    calls = [{"id": str(i), "name": "read_file", "args": {}} for i in range(6)]
    messages = [
        HumanMessage(content="核验修改后的六页"),
        AIMessage(content="", tool_calls=calls),
        *[
            ToolMessage(
                tool_call_id=str(i),
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "a" * 200000},
                    }
                ],
            )
            for i in range(6)
        ],
    ]
    request = ModelRequest(model=FakeListChatModel(responses=["ok"]), messages=messages, tools=[])
    assembler = ContextAssembler(OVContextConfig(context_window=256000))
    tokens = assembler.check_input_budget(request)
    assert 6 * 1024 <= tokens < 10000
