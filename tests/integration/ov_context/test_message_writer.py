import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)

from openviking.integrations._message_content import LANGCHAIN_CONTENT_MIME
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.errors import IncompleteWriteError
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter, WriterConfig
from tests.ov_context_support.message_examples import exchange
from tests.ov_context_support.writer import SessionBatchClient, make_session


async def test_incremental_write_reuses_messages_and_checkpoint_cursor(tmp_path):
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    raw = exchange(["ok", "done"])
    writer = OVMessageWriter(client)
    first = await writer.write(session.session_id, raw[:2], 0, "run-1")
    assert (first.write_index, first.written_count) == (2, 2)
    assert first.model_messages[0] is raw[0] and first.model_messages[1] is raw[1]
    # A new Writer needs only the persisted cursor, not an internal progress map.
    second = await OVMessageWriter(client).write(session.session_id, raw, first.write_index)
    assert (second.write_index, second.written_count) == (4, 2)
    assert second.model_messages[0] is raw[2] and second.model_messages[1] is raw[3]
    assert len(client.requests[-1]) == 1  # parallel tool results share one spec
    assert len(session.messages) == 4  # native Session splits that spec into two
    assert client.requests[-1][0]["parts"][1]["tool_input"] == {"record_id": 1}
    noop = await writer.write(session.session_id, raw, second.write_index)
    assert noop.model_messages == [] and noop.written_count == 0
    assert len(client.requests) == 2


async def test_large_parallel_result_replaces_only_matching_content(tmp_path):
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    raw = exchange(["small", "关键订单信息。" * 1000])
    raw[3].status = "error"
    before = deepcopy(raw)
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    assert result.model_messages[:3] == raw[:3]
    assert all(result.model_messages[i] is raw[i] for i in range(3))
    short = result.model_messages[3]
    assert short is not raw[3] and len(short.content) < len(raw[3].content)
    assert "viking://" in short.content
    assert short.model_dump(exclude={"content"}) == raw[3].model_dump(exclude={"content"})
    assert raw == before
    result_id = session.messages[-1].parts[0].tool_output_ref.rsplit("/", 1)[-1]
    stored = await session.read_tool_result(result_id, limit=-1)
    assert stored["content"] == raw[3].content
    assert client.requests[0][-1]["parts"][1]["tool_status"] == "error"


async def test_human_and_ai_content_blocks_and_peer_are_preserved(tmp_path):
    session = make_session(tmp_path, enabled=False)
    client = SessionBatchClient(session)
    user = HumanMessage(
        content=[
            {"type": "text", "text": "看图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"},
        ]
    )
    assistant = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "private"},
            {"type": "text", "text": "答复"},
        ],
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    raw = [SystemMessage(content="system"), user, assistant]
    before = deepcopy(raw)
    result = await OVMessageWriter(client, WriterConfig(peer_id="default-peer")).write(
        session.session_id, raw, 0, peer_id="actual-peer"
    )
    assert result.write_index == 3 and result.written_count == 2
    assert result.model_messages[0] is user and result.model_messages[1] is assistant
    assert len(session.messages) == 2
    assert client.requests[0][1]["peer_id"] == "actual-peer"
    assert "peer_id" not in client.requests[0][0]
    assert client.requests[0][1]["parts"] == [{"type": "text", "text": "答复"}]
    assert raw == before


async def test_structured_tool_output_is_written_losslessly(tmp_path):
    session = make_session(tmp_path, enabled=False)
    client = SessionBatchClient(session)
    blocks = [
        {"type": "text", "text": "中文"},
        {"type": "image_url", "image_url": "https://example.com/a.png"},
    ]
    raw = exchange([blocks])
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    part = client.requests[0][-1]["parts"][0]
    assert json.loads(part["tool_output"]) == blocks
    assert part["tool_output_mime_type"] == LANGCHAIN_CONTENT_MIME
    assert result.model_messages[-1] is raw[-1]


@pytest.mark.parametrize(
    "name",
    [
        "openviking_tool_result_read",
        "openviking_tool_result_search_many",
        "openviking_tool_result_list",
    ],
)
async def test_backtracking_model_view_keeps_evidence(tmp_path, name):
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    raw = exchange(["evidence " * 1000], names=[name])
    raw[1].tool_calls[0]["args"] = {
        "ref": session.uri + "/tool-results/existing",
        "offset": 12,
        "limit": 1000,
    }
    # Some ToolMessages omit name; the assistant call is the reliable registry.
    raw[-1].name = None
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    assert result.model_messages[-1] is raw[-1]
    assert result.model_messages[-1].content == raw[-1].content
    part = client.requests[0][-1]["parts"][0]
    assert part["tool_input"] == raw[1].tool_calls[0]["args"]
    assert part["tool_name"] == name


async def test_over_100_specs_and_local_receipt_indices(tmp_path):
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    raw = [HumanMessage(content=str(i)) for i in range(100)] + exchange(["large " * 1000])
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    assert [len(batch) for batch in client.requests] == [100, 3]
    assert result.write_index == 103 and result.written_count == 103
    assert result.model_messages[-1].content != raw[-1].content
    assert result.model_messages[99] is raw[99]


@pytest.mark.parametrize(
    "bad",
    [
        AIMessageChunk(content="partial"),
        SystemMessageChunk(content="partial system"),
        HumanMessage(content=[{"type": "audio", "data": "unsupported"}]),
        HumanMessage(content="recall", additional_kwargs={"ov_context_injected": True}),
        ChatMessage(role="custom", content="unsupported"),
        ToolMessage(content="orphan", tool_call_id="missing"),
    ],
)
async def test_invalid_delta_is_rejected_before_any_batch(bad):
    client = AsyncMock()
    with pytest.raises(ValueError):
        await OVMessageWriter(client, WriterConfig(batch_size=1)).write(
            "s", [HumanMessage(content="valid"), bad], 0
        )
    client.batch_add_messages.assert_not_awaited()


@pytest.mark.parametrize("cursor", [-1, 2, True])
async def test_invalid_cursor(cursor):
    client = AsyncMock()
    with pytest.raises(ValueError):
        await OVMessageWriter(client).write("s", [HumanMessage(content="a")], cursor)
    client.batch_add_messages.assert_not_awaited()


@pytest.mark.parametrize("size", [0, 101, True])
def test_batch_size_validation(size):
    with pytest.raises(ValueError):
        WriterConfig(batch_size=size)


async def test_server_without_tool_views_preserves_raw_tool_view():
    client = AsyncMock()
    client.batch_add_messages.return_value = {
        "session_id": "s",
        "message_count": 3,
        "added": 3,
    }
    raw = exchange(["large " * 1000])
    result = await OVMessageWriter(client).write("s", raw, 0)
    assert result.model_messages[-1] is raw[-1]


async def test_original_count_receipt_returns_local_messages():
    client = AsyncMock()
    client.batch_add_messages.return_value = {"session_id": "s", "message_count": 1, "added": 1}
    original = HumanMessage(id="user", content="a")
    result = await OVMessageWriter(client).write("s", [original], 0)
    assert result.write_index == 1 and result.written_count == 1
    assert result.model_messages[0] is original
    assert client.batch_add_messages.await_count == 1


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"session_id": "wrong", "message_count": 1, "added": 1},
        {"session_id": "s", "message_count": 1, "added": 0},
        {"session_id": "s", "message_count": 0, "added": 1},
    ],
)
async def test_unconfirmed_receipt_never_advances_cursor(receipt):
    client = AsyncMock()
    client.batch_add_messages.return_value = receipt
    with pytest.raises(IncompleteWriteError) as error:
        await OVMessageWriter(client).write("s", [HumanMessage(content="a")], 0)
    assert error.value.confirmed_result.write_index == 0
    assert error.value.confirmed_result.model_messages == []
    assert client.batch_add_messages.await_count == 1


async def test_timeout_is_not_retried():
    client = AsyncMock()
    client.batch_add_messages.side_effect = httpx.ReadTimeout("unknown outcome")
    with pytest.raises(IncompleteWriteError):
        await OVMessageWriter(client).write("s", [HumanMessage(content="a")], 0)
    assert client.batch_add_messages.await_count == 1


async def test_no_delta_does_not_scan_or_copy_history():
    class NoScanHistory(list):
        def __getitem__(self, key):
            raise AssertionError("No-op copied message history")

        def __iter__(self):
            raise AssertionError("No-op scanned message history")

    client = AsyncMock()
    result = await OVMessageWriter(client).write("s", NoScanHistory([HumanMessage(content="a")]), 1)
    assert result.write_index == 1 and result.written_count == 0
    client.batch_add_messages.assert_not_awaited()


async def test_partial_batches_report_confirmed_prefix_without_retry():
    client = AsyncMock()
    client.batch_add_messages.side_effect = [
        {"session_id": "s", "message_count": 1, "added": 1},
        OSError("second batch failed"),
    ]
    raw = [HumanMessage(content="a"), AIMessage(content="b")]
    with pytest.raises(IncompleteWriteError) as error:
        await OVMessageWriter(client, WriterConfig(batch_size=1)).write("s", raw, 0)
    result = error.value.confirmed_result
    assert (result.write_index, result.written_count) == (1, 1)
    assert result.model_messages[0] is raw[0]
    assert client.batch_add_messages.await_count == 2


async def test_existing_session_engine_accepts_real_writer_and_keeps_state_on_failure(tmp_path):
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    engine = SessionEngine(
        OpenVikingConnection(client=client),
        OVContextConfig(),
        None,
        None,
        message_writer=OVMessageWriter(client),
    )
    raw = exchange(["result " * 1000])
    context = OVContext(session.session_id, owner=("a", "u", "test-peer", session.session_id))
    await engine.write(context, raw, identity=context.identity)
    assert context.write_index == 3 and context.messages[-1].content != raw[-1].content
    before = deepcopy(context)
    client.batch_add_messages = AsyncMock(side_effect=OSError("failed"))
    with pytest.raises(OSError):
        await engine.write(context, raw + [AIMessage(content="final")], identity=context.identity)
    assert context == before


async def test_shared_writer_keeps_sessions_independent(tmp_path):
    clients = {sid: SessionBatchClient(make_session(tmp_path / sid)) for sid in ("a", "b")}
    for sid, client in clients.items():
        client.session.session_id = sid

    class RoutingClient:
        async def batch_add_messages(self, session_id, messages):
            return await clients[session_id].batch_add_messages(session_id, messages)

    writer = OVMessageWriter(RoutingClient())
    results = await asyncio.gather(
        *[writer.write(sid, [HumanMessage(content=sid)], 0) for sid in clients]
    )
    assert [r.write_index for r in results] == [1, 1]
    assert [clients[sid].session.messages[0].parts[0].text for sid in clients] == ["a", "b"]
