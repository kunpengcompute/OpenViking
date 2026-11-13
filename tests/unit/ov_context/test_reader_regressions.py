"""Regression scenarios from the reader and OVContext review."""

import asyncio
import json
import subprocess
import sys
from copy import deepcopy

import httpx
import pytest
from langchain_core.messages import HumanMessage

from openviking.integrations._message_content import (
    LANGCHAIN_CONTENT_MIME,
    decode_tool_content,
    encode_tool_content,
    normalize_image_block,
)
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import (
    MemoryReaderConfig,
    OVContextReader,
    restore_messages,
)
from openviking.integrations.langchain.ov_context.state import OVContext
from openviking.message import Message
from openviking.message.part import part_from_dict
from openviking_cli._exception_utils import auth_error
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.exceptions import PermissionDeniedError, UnauthenticatedError, UnavailableError
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext


@pytest.mark.parametrize(
    "image",
    [
        {"type": "image", "source_type": "url", "url": "https://example.test/a.png"},
        {"type": "image", "source_type": "base64", "base64": "AA==", "mime_type": "image/png"},
        {"type": "image", "url": "https://example.test/a.png", "detail": "low"},
    ],
)
async def test_image_conversion_allows_refresh_but_changed_image_is_rejected(client, image):
    user = HumanMessage(id="u1", content=[{"type": "text", "text": "分析图片"}, image])
    cache = OVContext("s1", messages=[user], archive_id="archive_1")
    cache.begin_run("run1", user)
    cache.inject_recall("参考内容", "success")
    original = deepcopy(user.model_dump())
    view = cache.messages[0]
    client.get_session_context.return_value.update(
        latest_finished_archive_id="archive_2",
        messages=[
            {
                "id": "u1",
                "role": "user",
                "parts": [
                    {"type": "text", "text": "分析图片"},
                    normalize_image_block(image),
                ],
            }
        ],
    )
    reader = OVContextReader(client)
    snapshot = await reader.exchange_context("s1", "archive_2")
    cache.replace(snapshot.messages, snapshot.archive_id)
    assert cache.messages[-1] is view and cache.archive_id == "archive_2"
    assert user.model_dump() == original
    client.get_session_context.return_value["messages"][0]["parts"][1]["image_url"]["url"] += (
        "changed"
    )
    snapshot = await reader.exchange_context("s1", "archive_2")
    previous, state = cache.messages, deepcopy(cache.to_state())
    with pytest.raises(ContextReadError, match="does not match"):
        cache.replace(snapshot.messages, "archive_3")
    assert cache.messages is previous and cache.to_state() == state
    assert user.model_dump() == original


def tool_record(output, mime=LANGCHAIN_CONTENT_MIME, **metadata):
    return {
        "id": "tool1",
        "role": "user",
        "parts": [
            {
                "type": "tool",
                "tool_id": "call1",
                "tool_name": "browser",
                "tool_status": "completed",
                "tool_output": output,
                "tool_output_mime_type": mime,
                **metadata,
            }
        ],
    }


def test_structured_tool_content_roundtrips_real_ov_message_storage():
    blocks = [
        {"type": "text", "text": "截图结果"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    encoded, mime = encode_tool_content(blocks)
    record = tool_record(encoded, mime)
    message = Message(id="tool1", role="user", parts=[part_from_dict(record["parts"][0])])
    restored = restore_messages([Message.from_dict(json.loads(message.to_jsonl())).to_dict()])[0]
    assert isinstance(restored.content, list) and restored.content == blocks
    restored.content[1]["image_url"]["url"] = "changed"
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,AA=="
    assert record["parts"][0]["tool_output"] == encoded


@pytest.mark.parametrize(
    "metadata",
    [
        {"tool_output_ref": "viking://s/tool-results/r1", "tool_output_truncated": True},
        {
            "tool_output_ref": "",
            "tool_output_truncated": True,
            "tool_output_externalization_error": "store failed",
        },
        {"tool_output_source_ref": "viking://s/tool-results/r1", "tool_output_truncated": False},
        {
            "tool_output_ref": "viking://s/tool-results/r1",
            "tool_output_externalized_reason": "source_read",
        },
    ],
)
def test_json_mime_short_views_stay_text(metadata):
    record = tool_record("文本短视图。原文引用：viking://s/tool-results/r1", **metadata)
    original = deepcopy(record)
    content = restore_messages([record])[0].content
    assert isinstance(content, str) and content == record["parts"][0]["tool_output"]
    assert decode_tool_content(record["parts"][0]) == content
    assert record == original


@pytest.mark.parametrize("output", ["not JSON", '{"wrong": "shape"}', "[1]", ""])
def test_invalid_full_structured_tool_content_is_rejected(output):
    with pytest.raises(ContextReadError, match="Invalid structured tool content"):
        restore_messages([tool_record(output)])


def test_plain_text_tools_and_literal_json_text_are_not_decoded():
    for text in ("普通文本", '[{"type":"text","text":"literal"}]'):
        assert restore_messages([tool_record(text, "text/plain")])[0].content == text


def memory(experience=False, *, category=None):
    return MatchedContext(
        uri="viking://user/memories/experiences/deploy.md"
        if experience
        else "viking://user/memories/preferences/brief.md",
        context_type=ContextType.MEMORY,
        category=category or ("experiences" if experience else "preferences"),
        abstract="部署经验" if experience else "简短回答",
        score=0.9,
        level=2,
    )


@pytest.mark.parametrize(
    "config,query",
    [
        (
            MemoryReaderConfig(search_mode="find", experience_enabled=False),
            "请修复仓库插件并运行测试",
        ),
        (MemoryReaderConfig(search_mode="find", experience_enabled=True), "你好，今天天气怎么样"),
        (
            MemoryReaderConfig(search_mode="find", experience_enabled=True, experience_limit=0),
            "请修复仓库插件并运行测试",
        ),
        (
            MemoryReaderConfig(search_mode="find", experience_enabled=True, experience_max_chars=0),
            "请修复仓库插件并运行测试",
        ),
        (
            MemoryReaderConfig(
                search_mode="find", experience_enabled=True, experience_score_threshold=0.95
            ),
            "请修复仓库插件并运行测试",
        ),
    ],
)
@pytest.mark.parametrize("category", ["experiences", "preferences"])
async def test_ordinary_search_cannot_bypass_experience_controls(client, config, query, category):
    client.find.return_value = FindResult(
        memories=[memory(), memory(True, category=category)], resources=[], skills=[]
    )
    result = await OVContextReader(client, config).recall(query)
    assert result.status == "success" and result.count == 1
    assert "简短回答" in result.content and "部署经验" not in result.content
    assert "[任务经验]" not in result.content


async def test_triggered_experience_is_injected_once_through_its_own_channel(client):
    client.find.return_value = FindResult(
        memories=[memory(), memory(True)], resources=[], skills=[]
    )
    client.read.return_value = "## Approach\n检查部署日志。"
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", experience_enabled=True)
    ).recall("请修复仓库部署脚本并运行测试")
    assert result.count == 2 and "[任务经验]" in result.content
    assert result.content.count(memory(True).uri) == 1
    assert "部署经验" not in result.content.split("[长期记忆]")[1]


def authentication_errors():
    return [
        UnauthenticatedError("test authentication failure"),
        PermissionDeniedError("test permission failure"),
        httpx.HTTPStatusError(
            "test HTTP 401",
            request=httpx.Request("GET", "http://test"),
            response=httpx.Response(401),
        ),
        httpx.HTTPStatusError(
            "test HTTP 403",
            request=httpx.Request("GET", "http://test"),
            response=httpx.Response(403),
        ),
    ]


@pytest.mark.parametrize("error", authentication_errors())
@pytest.mark.parametrize("operation", ["find", "read", "exchange"])
async def test_authentication_errors_propagate_without_summary_fallback(client, error, operation):
    client.find.return_value = FindResult(memories=[memory()], resources=[], skills=[])
    reader = OVContextReader(client, MemoryReaderConfig(search_mode="find", prefer_abstract=False))
    if operation == "exchange":
        client.get_session_context.side_effect = error
        call = reader.exchange_context("s1")
    else:
        getattr(client, operation).side_effect = error
        call = reader.recall("原始问题")
    with pytest.raises(type(error)) as caught:
        await call
    assert caught.value is error and auth_error(caught.value)


async def test_partial_search_auth_failure_is_not_hidden_by_successful_channel(client):
    error = PermissionDeniedError("test permission failure")
    client.find.side_effect = [FindResult(memories=[memory()], resources=[], skills=[]), error]
    reader = OVContextReader(
        client,
        MemoryReaderConfig(search_mode="find", experience_enabled=True),
    )
    with pytest.raises(PermissionDeniedError):
        await reader.recall("帮我修复 service API 报错并完成发布")


async def test_auth_failure_is_not_hidden_by_a_slow_parallel_search(client):
    cancelled = asyncio.Event()

    async def find(query, *, target_uri, **kwargs):
        if isinstance(target_uri, list):
            raise UnauthenticatedError("test authentication failure")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client.find.side_effect = find
    reader = OVContextReader(
        client,
        MemoryReaderConfig(
            search_mode="find",
            target_uris=("viking://user/memories", "viking://agent/memories"),
            experience_enabled=True,
            recall_timeout=0.05,
        ),
    )
    with pytest.raises(UnauthenticatedError):
        await asyncio.wait_for(reader.recall("帮我修复 service API 报错并完成发布"), timeout=0.2)
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


async def test_unavailable_service_still_uses_summary_fallback(client):
    client.find.return_value = FindResult(memories=[memory()], resources=[], skills=[])
    client.read.side_effect = UnavailableError()
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", prefer_abstract=False)
    ).recall("原始问题")
    assert result.status == "success" and "简短回答" in result.content
    client.find.side_effect = UnavailableError()
    assert (
        await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("原始问题")
    ).status == "error"


@pytest.mark.parametrize(
    "status,error_type", [(401, UnauthenticatedError), (403, PermissionDeniedError)]
)
@pytest.mark.parametrize(
    "body",
    [
        b"gateway rejection",
        b'{"detail":"access denied"}',
        b'{"status":"error","error":{"code":"UNKNOWN"}}',
    ],
)
def test_sdk_preserves_http_auth_classification_without_ov_envelope(status, error_type, body):
    # The response handler does not need a configured client or personal CLI config.
    client = object.__new__(AsyncHTTPClient)
    with pytest.raises(error_type) as caught:
        client._handle_response_data(httpx.Response(status, content=body))
    assert auth_error(caught.value)


@pytest.mark.parametrize(
    "status,code,error_type",
    [
        (401, "UNAUTHENTICATED", UnauthenticatedError),
        (403, "PERMISSION_DENIED", PermissionDeniedError),
    ],
)
def test_sdk_preserves_standard_auth_error_message_and_details(status, code, error_type):
    client = object.__new__(AsyncHTTPClient)
    response = httpx.Response(
        status,
        json={
            "status": "error",
            "error": {
                "code": code,
                "message": "Specific access failure",
                "details": {"resource": "memory.md"},
            },
        },
    )
    with pytest.raises(error_type) as caught:
        client._handle_response_data(response)
    assert str(caught.value) == "Specific access failure"
    if status == 403:
        assert caught.value.details["resource"] == "memory.md"


@pytest.mark.parametrize(
    "first,second",
    [
        (
            "openviking.integrations.langchain.ov_context.reader",
            "openviking.integrations.langchain.ov_context",
        ),
        (
            "openviking.integrations.langchain.ov_context",
            "openviking.integrations.langchain.ov_context.reader",
        ),
    ],
)
def test_reader_and_context_import_in_either_order(first, second):
    result = subprocess.run(
        [sys.executable, "-c", f"import {first}; import {second}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
