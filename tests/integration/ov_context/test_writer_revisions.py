"""Writer identity, native tool-block and full request-encoding regressions."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.messages.content import ContentBlock

from openviking.integrations._message_content import MessageContent
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter, WriterConfig
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity
from tests.ov_context_support.writer import SessionBatchClient, make_session


def assistant(kind: str) -> AIMessage:
    call: ToolCall = {"id": "c", "name": "lookup", "args": {"record": 42}}
    if kind == "anthropic":
        return AIMessage(
            id="a",
            content=[
                {"type": "text", "text": "查询中"},
                {"type": "tool_use", "id": "c", "name": "lookup", "input": {"record": 42}},
            ],
            tool_calls=[call],
        )
    blocks: list[ContentBlock] = [
        {"type": "tool_call", "id": "c", "name": "lookup", "args": {"record": 42}}
    ]
    if kind == "mixed":
        blocks.insert(0, {"type": "text", "text": "查询中"})
    return AIMessage(id="a", content_blocks=blocks)


@pytest.mark.parametrize("kind", ["standard", "mixed", "anthropic"])
async def test_native_tool_blocks_write_once_and_keep_original_messages(
    tmp_path: Path, kind: str
) -> None:
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    ai = assistant(kind)
    raw = [
        HumanMessage(id="u", content="查询订单"),
        ai,
        ToolMessage(id="t", tool_call_id="c", content="订单详情。" * 1000, artifact={"record": 42}),
    ]
    before = deepcopy(raw)
    writer = OVMessageWriter(client)
    first = await writer.write(session.session_id, raw[:2], 0, "r")
    second = await writer.write(session.session_id, raw, first.write_index, "r")
    assert (first.write_index, second.write_index) == (2, 3)
    parts = client.requests[0][1]["parts"]
    tools = [part for part in parts if part["type"] == "tool"]
    assert tools == [
        {
            "type": "tool",
            "tool_id": "c",
            "tool_name": "lookup",
            "tool_input": {"record": 42},
            "tool_status": "pending",
        }
    ]
    assert len(parts) == (1 if kind == "standard" else 2)
    assert first.model_messages[1] is ai
    assert second.model_messages[0] is not raw[2]
    assert second.model_messages[0].model_dump(exclude={"content"}) == raw[2].model_dump(
        exclude={"content"}
    )
    assert raw == before
    assert len(session.messages) == 3


@pytest.mark.parametrize("kind", ["standard", "anthropic"])
@pytest.mark.parametrize("field", ["id", "name", "args", "missing"])
async def test_unmatched_native_tool_block_fails_before_writing(kind: str, field: str) -> None:
    ai = assistant(kind)
    if field == "missing":
        ai.tool_calls = []
    elif field == "args":
        ai.tool_calls[0]["args"] = {"record": 99}
    elif field == "id":
        ai.tool_calls[0]["id"] = "different"
    else:
        ai.tool_calls[0]["name"] = "different"
    client = AsyncMock()
    with pytest.raises(ValueError, match="does not match"):
        await OVMessageWriter(client, WriterConfig(batch_size=1)).write(
            "s", [HumanMessage(content="query"), ai], 0
        )
    client.batch_add_messages.assert_not_awaited()


@pytest.mark.parametrize("message_peer", [None, "message-owner"])
async def test_shared_native_sdk_forwards_identity_on_every_batch_without_default_mutation(
    message_peer: str | None,
) -> None:
    identities = [
        OVIdentity("account-a", "user-a", "mail"),
        OVIdentity("account-b", "user-b", "doc"),
    ]
    owners = {f"s-{identity.user_id}": identity for identity in identities}
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)
        payload = json.loads(request.content)
        count = len(payload["messages"])
        sid = request.url.path.split("/")[-3]
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {"session_id": sid, "added": count, "message_count": count},
            },
        )

    sdk = AsyncHTTPClient(
        url="http://ov.test", agent_id="legacy", profile_enabled=False, auth_mode="trusted"
    )
    sdk._http = httpx.AsyncClient(
        base_url="http://ov.test",
        headers={
            "X-API-Key": "test-only",
            "X-OpenViking-Account": "default-account",
            "X-OpenViking-User": "default-user",
            "X-OpenViking-Actor-Peer": "legacy",
        },
        transport=httpx.MockTransport(handle),
    )
    defaults = dict(sdk._http.headers)
    writer = OVMessageWriter(sdk, WriterConfig(batch_size=1))
    try:
        results = await asyncio.gather(
            *(
                writer.write(
                    sid,
                    [
                        HumanMessage(id="u", content="query"),
                        assistant("standard"),
                        ToolMessage(id="t", tool_call_id="c", content="result"),
                    ],
                    0,
                    "run",
                    identity=identity,
                    peer_id=message_peer,
                )
                for sid, identity in owners.items()
            )
        )
        assert [result.write_index for result in results] == [3, 3]
        assert len(requests) == 6
        for request in requests:
            identity = owners[request.url.path.split("/")[-3]]
            for key, value in identity.headers().items():
                assert request.headers[key] == value
            assert request.headers["X-API-Key"] == "test-only"
            payload = json.loads(request.content)
            assert set(payload) == {"messages"}
            spec = payload["messages"][0]
            assert "agent_id" not in spec and "identity" not in spec
            if spec["role"] == "assistant" or spec["parts"][0]["type"] == "tool":
                assert spec["peer_id"] == (message_peer or identity.actor_peer_id)
        assert dict(sdk._http.headers) == defaults
        assert sdk._legacy_agent_id == "legacy"
        await sdk.batch_add_messages("legacy-session", [{"role": "assistant", "content": "legacy"}])
        assert requests[-1].headers["X-OpenViking-User"] == "default-user"
        assert json.loads(requests[-1].content)["messages"][0]["agent_id"] == "legacy"
    finally:
        await sdk.close()


@pytest.mark.parametrize("field", ["text", "args", "tool_text", "tool_blocks", "image"])
async def test_later_batch_invalid_utf8_prevents_first_http_request(field: str) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("No batch may be sent before full UTF-8 validation")

    raw: list[BaseMessage] = [HumanMessage(content=str(index)) for index in range(101)]
    invalid = "\ud800"
    if field == "text":
        raw.append(HumanMessage(content=invalid))
    elif field == "args":
        raw.append(
            AIMessage(
                content="", tool_calls=[{"id": "c", "name": "lookup", "args": {"value": invalid}}]
            )
        )
    elif field in ("tool_text", "tool_blocks"):
        body: MessageContent = (
            invalid if field == "tool_text" else [{"type": "text", "text": invalid}]
        )
        raw.extend([assistant("standard"), ToolMessage(tool_call_id="c", content=body)])
    else:
        raw.append(
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": "https://example.com/" + invalid}}
                ]
            )
        )
    sdk = AsyncHTTPClient(url="http://ov.test", profile_enabled=False)
    sdk._http = httpx.AsyncClient(base_url="http://ov.test", transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(UnicodeEncodeError):
            await OVMessageWriter(sdk).write("s", raw, 0)
        assert requests == []
    finally:
        await sdk.close()
