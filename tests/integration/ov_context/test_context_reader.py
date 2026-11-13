"""Snapshot reader contracts and caller-owned cache publication."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict

from openviking.integrations._message_content import encode_tool_content
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from tests.ov_context_support.mock_context import SnapshotAdapter
from tests.ov_context_support.mock_reader import MockContextReader

from .test_session_hooks import runtime, start

pytestmark = pytest.mark.asyncio


def reader(view):
    client = SimpleNamespace(call=AsyncMock(return_value=view))
    return SnapshotAdapter(client, MemoryReaderConfig()), client


async def test_context_read_normalizes_summary_multimodal_and_paired_tools_without_mutation():
    blocks = [{"type": "text", "text": "mail"}]
    output, mime = encode_tool_content(blocks)
    view = {
        "archive_id": "archive_001",
        "latest_archive_overview": "Opened mailbox",
        "messages": [
            {
                "id": "u",
                "role": "user",
                "parts": [
                    {"type": "text", "text": "Read mail"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                ],
            },
            {
                "id": "a",
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "c",
                        "tool_name": "read",
                        "tool_input": {"path": "mail"},
                    },
                ],
            },
            {
                "id": "t",
                "role": "user",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "c",
                        "tool_name": "read",
                        "tool_status": "completed",
                        "tool_output": output,
                        "tool_output_mime_type": mime,
                    },
                ],
            },
        ],
    }
    context = OVContext("s")
    previous, original_view = deepcopy(context.to_state()), deepcopy(view)
    implementation, client = reader(view)
    snapshot = await implementation.read_context(context, expected_archive_id="archive_001")
    assert context.to_state() == previous and view == original_view
    assert [type(m) for m in snapshot.messages] == [
        HumanMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assert snapshot.messages[0].id == "ov:summary"
    assert snapshot.messages[1].content == view["messages"][0]["parts"]
    assert snapshot.messages[2].tool_calls == [
        {"id": "c", "name": "read", "args": {"path": "mail"}, "type": "tool_call"}
    ]
    assert snapshot.messages[3].tool_call_id == "c"
    assert snapshot.messages[3].content == blocks
    client.call.assert_awaited_once_with(
        "get_session_context",
        identity=None,
        session_id="s",
        token_budget=MemoryReaderConfig().context_token_budget,
    )


async def test_context_read_rejects_unpublished_archive():
    implementation, _ = reader({"messages": [], "archive_id": "archive_001"})
    with pytest.raises(ValueError, match="expected archive"):
        await implementation.read_context(OVContext("s"), "archive_002")


async def test_context_read_restores_full_native_fields_without_local_cache():
    stored = [
        HumanMessage(
            id="u",
            name="user",
            content=[{"type": "text", "text": "Read mail"}],
            additional_kwargs={"trace": "user"},
            response_metadata={"source": "remote"},
        ),
        AIMessage(
            id="a",
            name="agent",
            content="Reading",
            additional_kwargs={"provider": "data"},
            response_metadata={"model": "test-model", "finish_reason": "tool_calls"},
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            tool_calls=[{"id": "c", "name": "read", "args": {"path": "mail"}}],
        ),
        ToolMessage(
            id="t",
            name="read",
            tool_call_id="c",
            content="mail",
            status="success",
            artifact={"file": "mail.txt"},
            additional_kwargs={"trace": "tool"},
            response_metadata={"elapsed_ms": 5},
        ),
    ]
    view = {
        "messages": [
            {"id": "remote-u", "role": "user", "content": "Read mail"},
            {
                "id": "remote-a",
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "c",
                        "tool_name": "read",
                        "tool_input": {"path": "mail"},
                    }
                ],
            },
            {
                "id": "remote-t",
                "role": "user",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "c",
                        "tool_name": "read",
                        "tool_output": "mail",
                        "tool_status": "completed",
                    }
                ],
            },
        ]
    }
    metadata = {
        "_native_payloads": {
            "remote-" + key: [message_to_dict(message)]
            for key, message in zip(("u", "a", "t"), stored, strict=True)
        }
    }
    client = SimpleNamespace(call=AsyncMock(return_value=view))
    implementation = SnapshotAdapter(client, MemoryReaderConfig(), metadata)
    snapshot = await implementation.read_context(OVContext("s"))
    assert [m.model_dump() for m in snapshot.messages] == [m.model_dump() for m in stored]
    assert all(
        result is not original for result, original in zip(snapshot.messages, stored, strict=True)
    )


async def test_fresh_reader_recovers_persisted_metadata_after_writer_run(env):
    user = HumanMessage(
        id="original", name="user", content="Read mail", additional_kwargs={"trace": 1}
    )
    middleware, state = await start(env, [user])
    engine = middleware._ov_session_engine
    # Recreate the reader and context with no previous message cache.
    implementation = SnapshotAdapter(engine.client, MemoryReaderConfig(), env[0].native_host_ids)
    snapshot = await implementation.read_context(OVContext("s"), identity=env[3].sdk_identity)
    assert snapshot.messages[0].model_dump() == user.model_dump()
    assert state["ov_context"]["messages"][0].model_dump() == user.model_dump()


@pytest.mark.parametrize("failure", ["version", "invalid_tools"])
async def test_invalid_context_does_not_replace_cache_or_advance_version(env, failure):
    middleware, state = await start(env)
    context = OVContext.from_state(state["ov_context"])
    previous = deepcopy(context.to_state())
    view = {
        "messages": [],
        "archive_id": -1 if failure == "version" else "archive_001",
        "latest_archive_overview": "Mail opened",
    }
    if failure == "invalid_tools":
        view["messages"] = [
            {
                "id": "orphan",
                "role": "user",
                "parts": [
                    {
                        "type": "tool",
                        "tool_id": "c",
                        "tool_name": "read",
                        "tool_status": "completed",
                        "tool_output": "mail",
                    },
                ],
            }
        ]
    implementation = MockContextReader(
        SimpleNamespace(call=AsyncMock(return_value=view)), MemoryReaderConfig()
    )
    middleware._ov_session_engine.context_reader = implementation
    with pytest.raises(ContextReadError if failure == "version" else ValueError):
        await middleware._ov_session_engine.refresh(context, identity=context.identity)
    assert context.to_state() == previous


async def test_host_reader_handles_both_context_refresh_and_recall(env):
    middleware, state = await start(env)
    engine = middleware._ov_session_engine
    assert isinstance(engine.context_reader, MockContextReader)
    assert callable(engine.context_reader.exchange_context)
    assert callable(engine.context_reader.recall)
    assert env[0].counts["context"] == 1
    state["messages"].append(AIMessage(id="answer", content="done"))
    state.update(await middleware.aafter_model(state, runtime(env)))
    assert [m.id for m in state["ov_context"]["messages"]] == ["m1", "answer"]


def snapshot_engine(meta, view):
    sdk = SimpleNamespace(
        _initialized=True,
        get_session=AsyncMock(return_value={"session_id": "s", **meta}),
        get_session_context=AsyncMock(return_value=view),
        get_task=AsyncMock(return_value={"status": "completed"}),
        list_tasks=AsyncMock(return_value=[]),
    )
    engine = SessionEngine(OpenVikingConnection(client=sdk), OVContextConfig(), None, None)
    engine.context_reader = MockContextReader(engine.client, MemoryReaderConfig())
    return engine, sdk


@pytest.mark.parametrize(
    "cached_id,published_id",
    [(None, "archive_001"), ("archive_001", "archive_002"), ("archive_999", "archive_1000")],
)
async def test_completed_archive_id_switches_cache_and_preserves_current_user(
    cached_id: str | None, published_id: str
) -> None:
    user = HumanMessage(id="u", content="Read mail")
    context = OVContext("s", messages=[user], archive_id=cached_id, owner=("a", "u", "p", "s"))
    context.begin_run("r", user)
    context.inject_recall("Use concise answers", "success")
    context.pending_commit = {"task_id": "t", "archive_id": cached_id}
    view = {
        "archive_id": published_id,
        "latest_archive_overview": "Mailbox opened",
        "messages": [{"id": "u", "role": "user", "content": "Read mail"}],
    }
    engine, _ = snapshot_engine({"latest_finished_archive_id": published_id}, view)
    engine.context_reader.metadata = {"_native_payloads": {"u": [message_to_dict(user)]}}
    result = await engine.check_summary(context, identity=context.identity)
    assert result.archive_id == context.archive_id == published_id
    assert [m.id for m in context.messages] == ["ov:summary", "u"]
    assert "Use concise answers" in context.messages[-1].content
    assert context.pending_commit is None


async def test_same_archive_reuses_cache_without_reading_context() -> None:
    context = OVContext("s", archive_id="archive_002", owner=("a", "u", "p", "s"))
    original = context.messages
    engine, sdk = snapshot_engine({"latest_finished_archive_id": "archive_002"}, {})
    result = await engine.check_summary(context, identity=context.identity)
    assert result.archive_id == "archive_002"
    assert context.messages is original
    sdk.get_session_context.assert_not_awaited()


@pytest.mark.parametrize("published_id", [None, "archive_001"])
async def test_older_archive_is_rejected_before_cache_mutation(published_id: str | None) -> None:
    context = OVContext("s", archive_id="archive_002", owner=("a", "u", "p", "s"))
    previous = deepcopy(context.to_state())
    engine, sdk = snapshot_engine({"latest_finished_archive_id": published_id}, {})
    with pytest.raises(RuntimeError, match="regressed"):
        await engine.check_summary(context, identity=context.identity)
    assert context.to_state() == previous
    sdk.get_session_context.assert_not_awaited()


async def test_force_refresh_same_archive_does_not_complete_pending_commit() -> None:
    context = OVContext(
        "s",
        archive_id="archive_002",
        owner=("a", "u", "p", "s"),
        pending_commit={"task_id": "t", "archive_id": "archive_002"},
        context_status="reload_required",
    )
    view = {"archive_id": "archive_002", "latest_archive_overview": "Mail opened", "messages": []}
    engine, sdk = snapshot_engine({"latest_finished_archive_id": "archive_002"}, view)
    await engine.check_summary(context, identity=context.identity)
    sdk.get_session_context.assert_awaited_once()
    assert context.pending_commit == {"task_id": "t", "archive_id": "archive_002"}
    assert context.context_status == "ready"


async def test_archive_and_context_snapshot_mismatch_keeps_prior_cache() -> None:
    context = OVContext("s", archive_id="archive_001", owner=("a", "u", "p", "s"))
    previous = deepcopy(context.to_state())
    view = {"archive_id": "archive_001", "latest_archive_overview": "Old summary", "messages": []}
    engine, _ = snapshot_engine({"latest_finished_archive_id": "archive_002"}, view)
    with pytest.raises(ContextReadError, match="regressed"):
        await engine.check_summary(context, identity=context.identity)
    assert context.to_state() == previous
