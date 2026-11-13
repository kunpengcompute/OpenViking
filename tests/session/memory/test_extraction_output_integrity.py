"""Reject truncated memory operations before JSON repair or persistence."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from openviking.models.vlm.backends.litellm_vlm import LiteLLMVLMProvider
from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking.models.vlm.backends.volcengine_vlm import VolcEngineVLM
from openviking.models.vlm.base import VLMResponse
from openviking.session.memory.extract_loop import ExtractLoop


class Operations(BaseModel):
    delete_uris: list[str] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)


def loop_with(response):
    loop = object.__new__(ExtractLoop)
    loop.vlm = SimpleNamespace(get_completion_async=AsyncMock(return_value=response))
    loop._mark_cache_breakpoint = AsyncMock()
    loop._disable_tools_for_iteration = False
    loop._tool_schemas = []
    loop._expected_fields = ["delete_uris", "entities"]
    loop._operations_model = Operations
    return loop


@pytest.mark.asyncio
async def test_truncated_string_cannot_be_repaired_into_memory():
    loop = loop_with(
        '{"delete_uris":[],"entities":[{"content":"统计区间：[2026-10-01T00:00:00+08:00'
    )
    tools, operations = await loop._call_llm([])
    assert tools is None and operations is None


@pytest.mark.asyncio
async def test_length_response_is_rejected_even_if_json_is_complete():
    loop = loop_with(
        VLMResponse(content='{"delete_uris":[],"entities":[]}', finish_reason="length")
    )
    with pytest.raises(RuntimeError, match="(?i)(output|token).*limit|truncat"):
        await loop._call_llm([])


@pytest.mark.asyncio
async def test_fenced_complete_operations_remain_supported():
    _, operations = await loop_with(
        '```json\n{"delete_uris":[],"entities":[{"content":"完整正文"}]}\n```'
    )._call_llm([])
    assert operations.entities[0]["content"] == "完整正文"


@pytest.mark.parametrize("cls", [OpenAIVLM, LiteLLMVLMProvider, VolcEngineVLM])
@pytest.mark.parametrize("has_tools", [False, True])
async def test_backend_does_not_discard_length_finish_reason(cls, has_tools):
    backend = object.__new__(cls)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content='{"entities":[]}', tool_calls=[]),
            )
        ],
        usage=None,
    )
    with pytest.raises(RuntimeError, match="(?i)(output|token).*limit|truncat"):
        backend._build_vlm_response(response, has_tools)


@pytest.mark.asyncio
async def test_real_openai_plaintext_call_checks_limit_after_recording_usage():
    backend = object.__new__(OpenAIVLM)
    backend.thinking = False
    backend.max_retries = 2
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="partial text", tool_calls=[]),
            )
        ],
        usage=None,
    )
    create = AsyncMock(return_value=response)
    backend.get_async_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    backend._build_text_kwargs = lambda *args, **kwargs: {}
    from unittest.mock import Mock

    backend._update_token_usage_from_response = Mock()
    with pytest.raises(RuntimeError, match="output token limit"):
        await backend.get_completion_async(messages=[{"role": "user", "content": "extract"}])
    assert create.await_count == 1
    backend._update_token_usage_from_response.assert_called_once()
