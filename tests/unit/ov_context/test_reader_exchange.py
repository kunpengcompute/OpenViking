import asyncio
from unittest.mock import call

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader


def record(message_id, role="user", parts=None):
    return {
        "id": message_id,
        "role": role,
        "parts": parts or [{"type": "text", "text": "用户问题"}],
    }


def terminal_reader(client, **kwargs):
    client.get_session_context.return_value = {
        "latest_finished_archive_id": "archive_002",
        "latest_archive_overview": "工作概要",
        "messages": [],
        "stats": {"totalArchives": 2},
    }
    return OVContextReader(client, **kwargs)


async def test_new_session_has_no_placeholder_summary(client):
    snapshot = await OVContextReader(client).exchange_context(session_id="s1")
    assert snapshot.messages == [] and snapshot.archive_id is None
    client.get_session_context.assert_awaited_once_with("s1", token_budget=128_000)
    assert client.mock_calls == [call.get_session_context("s1", token_budget=128_000)]


def test_reader_is_not_bound_to_a_session(client):
    reader = OVContextReader(client)
    assert reader.client is client
    assert set(vars(reader)) == {"client", "config"}
    assert not hasattr(reader, "session_id")
    with pytest.raises(TypeError, match="session_id"):
        OVContextReader(client, session_id="s1")
    with pytest.raises(TypeError, match="session_id"):
        reader.exchange_context()


async def test_one_reader_serves_multiple_explicit_sessions_without_retaining_progress(client):
    async def context(session_id, **kwargs):
        return {
            "latest_finished_archive_id": f"archive:{session_id}",
            "messages": [record(f"message:{session_id}")],
        }

    client.get_session_context.side_effect = context
    reader = OVContextReader(client)
    initial_members = vars(reader).copy()
    snapshots = [await reader.exchange_context(session_id=sid) for sid in ("s1", "s2", "s1")]
    assert [snapshot.archive_id for snapshot in snapshots] == [
        "archive:s1",
        "archive:s2",
        "archive:s1",
    ]
    assert [snapshot.messages[0].id for snapshot in snapshots] == [
        "message:s1",
        "message:s2",
        "message:s1",
    ]
    assert vars(reader) == initial_members
    client.get_session_context.assert_has_awaits(
        [call(sid, token_budget=128_000) for sid in ("s1", "s2", "s1")]
    )


async def test_concurrent_sessions_preserve_ids_and_do_not_serialize_or_mix_snapshots(client):
    session_ids = ("s1", "s2")
    both_reading = asyncio.Event()
    entered = set()

    async def context(session_id, **kwargs):
        entered.add(session_id)
        if len(entered) == 2:
            both_reading.set()
        await both_reading.wait()
        return {
            "latest_finished_archive_id": f"archive:{session_id}",
            "messages": [record("shared_ov_id", parts=[{"type": "text", "text": session_id}])],
        }

    client.get_session_context.side_effect = context
    reader = OVContextReader(client)
    snapshots = await asyncio.wait_for(
        asyncio.gather(*(reader.exchange_context(session_id=sid) for sid in session_ids)), timeout=1
    )
    for sid, snapshot in zip(session_ids, snapshots, strict=True):
        assert snapshot.archive_id == f"archive:{sid}"
        assert snapshot.messages[0].id == "shared_ov_id"
        assert snapshot.messages[0].content == sid
    assert not hasattr(reader, "session_id")


async def test_expected_archive_id_is_verified_from_the_same_context_response(client):
    reader = terminal_reader(client)
    snapshot = await reader.exchange_context(session_id="s2", expected_archive_id="archive_002")
    assert snapshot.archive_id == "archive_002"
    assert client.mock_calls == [call.get_session_context("s2", token_budget=128_000)]


@pytest.mark.parametrize("actual_archive_id", [None, "archive_001", "archive_003"])
async def test_unexpected_archive_boundary_does_not_produce_a_snapshot(client, actual_archive_id):
    client.get_session_context.return_value["latest_finished_archive_id"] = actual_archive_id
    reader = OVContextReader(client)
    with pytest.raises(ContextReadError, match="expected_archive_id"):
        await reader.exchange_context(session_id="s2", expected_archive_id="archive_002")
    client.get_session_context.assert_awaited_once_with("s2", token_budget=128_000)


async def test_done_boundary_and_summary_come_from_one_context_response(client):
    reader = terminal_reader(client)
    snapshot = await reader.exchange_context(session_id="s1")
    assert snapshot.archive_id == "archive_002"
    assert isinstance(snapshot.messages[0], HumanMessage)
    assert snapshot.messages[0].id == "ov_summary:archive_002"
    assert snapshot.messages[0].content == "[Session History Summary]\n工作概要"
    assert client.mock_calls == [call.get_session_context("s1", token_budget=128_000)]
    assert not hasattr(client, "get_session_archive")


async def test_failed_boundary_can_keep_an_older_overview_and_failed_archive_raw_messages(client):
    reader = terminal_reader(client)
    client.get_session_context.return_value.update(
        {
            "latest_archive_overview": "archive_001 生成的概要",
            "messages": [
                record("failed_raw", parts=[{"type": "text", "text": "失败归档保留的原文"}])
            ],
            "stats": {"totalArchives": 2, "failedArchives": 1},
        }
    )
    snapshot = await reader.exchange_context(session_id="s1")
    assert snapshot.archive_id == "archive_002"
    assert snapshot.messages[0].content == "[Session History Summary]\narchive_001 生成的概要"
    assert snapshot.messages[1].content == "失败归档保留的原文"
    assert client.mock_calls == [call.get_session_context("s1", token_budget=128_000)]


async def test_failed_boundary_without_an_overview_still_returns_raw_context(client):
    reader = terminal_reader(client)
    client.get_session_context.return_value.update(
        {
            "latest_archive_overview": "",
            "messages": [record("failed_raw")],
            "stats": {"totalArchives": 1, "failedArchives": 1},
        }
    )
    snapshot = await reader.exchange_context(session_id="s1")
    assert snapshot.archive_id == "archive_002"
    assert len(snapshot.messages) == 1 and snapshot.messages[0].content == "用户问题"


async def test_terminal_version_is_not_inferred_from_commit_count_or_archive_stats(client):
    reader = terminal_reader(client)
    # A newer commit has started, but only archive_002 has reached a terminal state.
    client.get_session_context.return_value.update(
        {
            "commit_count": 9,
            "stats": {"totalArchives": 9},
            "messages": [record("at_boundary_2")],
        }
    )
    snapshot = await reader.exchange_context(session_id="s1")
    assert snapshot.archive_id == "archive_002"
    assert snapshot.messages[1].additional_kwargs["openviking_message_id"] == "at_boundary_2"


@pytest.mark.parametrize("boundary", [0, False, 1.5, "", " \n ", {}, []])
async def test_invalid_terminal_version_is_rejected(client, boundary):
    client.get_session_context.return_value["latest_finished_archive_id"] = boundary
    with pytest.raises(ContextReadError, match="non-empty string or None"):
        await OVContextReader(client).exchange_context(session_id="s1")


async def test_missing_version_is_not_treated_as_a_new_session_or_inferred(client):
    client.get_session_context.return_value = {
        "messages": [],
        "latest_archive_overview": "",
        "commit_count": 3,
        "stats": {"totalArchives": 3},
    }
    with pytest.raises(ContextReadError, match="missing latest_finished_archive_id"):
        await OVContextReader(client).exchange_context(session_id="s1")


@pytest.mark.parametrize("records", [None, "", {}, ["not an OV record"]])
async def test_invalid_messages_do_not_produce_an_empty_snapshot(client, records):
    client.get_session_context.return_value["messages"] = records
    with pytest.raises(ContextReadError, match="list of OV message records"):
        await OVContextReader(client).exchange_context(session_id="s1")


async def test_missing_messages_are_rejected(client):
    client.get_session_context.return_value.pop("messages")
    with pytest.raises(ContextReadError, match="list of OV message records"):
        await OVContextReader(client).exchange_context(session_id="s1")


@pytest.mark.parametrize("summary", [None, 42, ["bad overview"]])
async def test_invalid_overview_is_rejected(client, summary):
    client.get_session_context.return_value["latest_archive_overview"] = summary
    with pytest.raises(ContextReadError, match="overview must be a string"):
        await OVContextReader(client).exchange_context(session_id="s1")


async def test_reader_can_be_called_while_the_caller_holds_its_session_lock(client):
    lock = asyncio.Lock()
    reader = OVContextReader(client)
    async with lock:
        snapshot = await reader.exchange_context(session_id="s1")
        assert lock.locked()
        assert snapshot.messages == [] and snapshot.archive_id is None
    client.get_session_context.assert_awaited_once()


async def test_caller_owned_lock_covers_refresh_and_cache_update_before_waiting_writer(client):
    lock = asyncio.Lock()
    events = []
    writers = []

    async def writer():
        async with lock:
            events.append("write")

    async def context(*args, **kwargs):
        assert lock.locked()
        writers.append(asyncio.create_task(writer()))
        await asyncio.sleep(0)
        return {"latest_finished_archive_id": None, "messages": [], "latest_archive_overview": ""}

    client.get_session_context.side_effect = context
    async with lock:
        await OVContextReader(client).exchange_context(session_id="s1")
        events.append("replace_cache")
    await asyncio.gather(*writers)
    assert events == ["replace_cache", "write"]


async def test_context_deadline_discards_late_results_from_a_cancellation_resistant_client(client):
    released = asyncio.Event()
    finished = asyncio.Event()

    async def context(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await released.wait()
            finished.set()
            return {"latest_finished_archive_id": "too_late", "messages": [record("late_user")]}

    client.get_session_context.side_effect = context
    reader = OVContextReader(client, MemoryReaderConfig(context_timeout=0.01))
    try:
        with pytest.raises(ContextReadError, match="timed out"):
            await asyncio.wait_for(reader.exchange_context(session_id="s1"), timeout=0.5)
    finally:
        released.set()
    await asyncio.wait_for(finished.wait(), timeout=0.5)
    assert set(vars(reader)) == {"client", "config"}
    client.get_session_context.assert_awaited_once_with("s1", token_budget=128_000)


async def test_later_context_response_can_advance_the_terminal_boundary(client):
    client.get_session_context.side_effect = [
        {"latest_finished_archive_id": "archive_001", "messages": [record("v1")]},
        {"latest_finished_archive_id": "archive_002", "messages": [record("v2")]},
    ]
    reader = OVContextReader(client)
    first = await reader.exchange_context(session_id="s1")
    second = await reader.exchange_context(session_id="s1")
    assert first.archive_id == "archive_001" and first.messages[0].id == "v1"
    assert second.archive_id == "archive_002" and second.messages[0].id == "v2"
    assert client.get_session_context.await_count == 2


async def test_restores_user_role_tool_results_and_pending_assistant_calls(client):
    client.get_session_context.return_value["messages"] = [
        record("ov_u"),
        record(
            "ov_a",
            "assistant",
            [
                {
                    "type": "tool",
                    "tool_id": "call_1",
                    "tool_name": "query_order",
                    "tool_input": {"id": 123},
                    "tool_status": "pending",
                    "tool_output": "",
                }
            ],
        ),
        record(
            "ov_t",
            "user",
            [
                {
                    "type": "tool",
                    "tool_id": "call_1",
                    "tool_name": "query_order",
                    "tool_status": "completed",
                    "tool_output": "订单已发货。原文引用：ref1",
                }
            ],
        ),
    ]
    snapshot = await OVContextReader(client).exchange_context(session_id="s1")
    user, assistant, tool = snapshot.messages
    assert isinstance(user, HumanMessage)
    assert isinstance(assistant, AIMessage) and assistant.tool_calls[0]["id"] == "call_1"
    assert assistant.tool_calls[0]["args"] == {"id": 123}
    assert isinstance(tool, ToolMessage) and tool.tool_call_id == "call_1"
    assert tool.content == "订单已发货。原文引用：ref1"
    assert user.id == "ov_u"


async def test_error_and_parallel_tool_parts_keep_their_call_ids(client):
    client.get_session_context.return_value["messages"] = [
        record(
            "ov_group",
            parts=[
                {"type": "tool", "tool_id": "a", "tool_status": "completed", "tool_output": "成功"},
                {"type": "tool", "tool_id": "b", "tool_status": "error", "tool_output": "失败"},
            ],
        )
    ]
    snapshot = await OVContextReader(client).exchange_context(session_id="s1")
    assert [m.tool_call_id for m in snapshot.messages] == ["a", "b"]
    assert [m.status for m in snapshot.messages] == ["success", "error"]
    assert len({m.id for m in snapshot.messages}) == 2


async def test_native_user_id_and_multimodal_content_are_preserved_without_callbacks(client):
    content = [
        {"type": "text", "text": "识别图片"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "low"}},
    ]
    original = HumanMessage(id="native_u", content=content)
    client.get_session_context.return_value["messages"] = [record(original.id, parts=content)]
    reader = OVContextReader(client)
    snapshot = await reader.exchange_context(session_id="s1")
    assert snapshot.messages[0].id == original.id
    assert snapshot.messages[0].content == content
    assert snapshot.messages[0].additional_kwargs["openviking_message_id"] == original.id
    snapshot.messages[0].content[1]["image_url"]["url"] = "changed"
    assert original.content == content
    assert (
        client.get_session_context.return_value["messages"][0]["parts"][1]["image_url"]["url"]
        != "changed"
    )


async def test_context_exchange_has_an_independent_timeout_and_budget(client):
    async def context(*args, **kwargs):
        await asyncio.sleep(0.03)
        return {"latest_finished_archive_id": None, "messages": [], "latest_archive_overview": ""}

    client.get_session_context.side_effect = context
    cfg = MemoryReaderConfig(recall_timeout=0.005, context_timeout=0.2, context_token_budget=512)
    assert (await OVContextReader(client, cfg).exchange_context(session_id="s1")).messages == []
    client.get_session_context.assert_awaited_once_with("s1", token_budget=512)
    cfg = MemoryReaderConfig(context_timeout=0.005)
    with pytest.raises(ContextReadError, match="timed out"):
        await OVContextReader(client, cfg).exchange_context(session_id="s1")


async def test_context_network_failure_is_not_disguised_as_empty_history(client):
    client.get_session_context.side_effect = RuntimeError("connection failed")
    with pytest.raises(ContextReadError, match="could not be verified"):
        await OVContextReader(client).exchange_context(session_id="s1")


@pytest.mark.parametrize(
    "part",
    [
        {"type": "unsupported"},
        {"type": "tool", "tool_status": "completed", "tool_output": "orphan"},
        {"type": "tool", "tool_id": "call", "tool_status": "pending"},
    ],
)
async def test_unsupported_or_incomplete_parts_fail_instead_of_losing_messages(client, part):
    client.get_session_context.return_value["messages"] = [record("ov_u", parts=[part])]
    with pytest.raises(ContextReadError):
        await OVContextReader(client).exchange_context(session_id="s1")
