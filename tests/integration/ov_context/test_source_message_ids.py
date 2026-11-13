"""Source identity survives native OV persistence, not only a local cache."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from openviking.client.local import LocalClient
from openviking.client.session import Session as ClientSession
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.assembler import ContextAssembler
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot, restore_messages
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking.message import Message, TextPart
from openviking.session import Session
from openviking_cli.client.sync_http import SyncHTTPClient
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.parametrize("recall", [False, True])
@pytest.mark.parametrize("blocks", [False, True])
async def test_repeated_user_text_recovers_native_ids_after_http_write_and_reload(
    native_api, recall, blocks
):
    session, client = native_api
    content = (
        [
            {"type": "text", "text": "继续"},
            {"type": "image_url", "image_url": {"url": "https://image.test/a.png"}},
        ]
        if blocks
        else "继续"
    )
    raw = [
        HumanMessage(id="first-question", content=deepcopy(content)),
        AIMessage(id="first-answer", content="好的"),
        HumanMessage(
            id="current-question", content=deepcopy(content), additional_kwargs={"trace": "keep"}
        ),
    ]
    original = deepcopy(raw)
    written = await OVMessageWriter(client).write(session.session_id, raw, 0)
    reloaded = Session(session._viking_fs, session_id=session.session_id)
    await reloaded.load()
    context = OVContext(
        session.session_id,
        messages=list(written.model_messages),
        write_index=written.write_index,
        owner=("a", "u", "p", session.session_id),
        context_status="ready",
    )
    context.begin_run("second-run", raw[-1])
    if recall:
        context.inject_recall("Only the current question receives this memory", "success")
    snapshot = ContextSnapshot(
        [
            HumanMessage(id="ov_summary:archive_001", content="Earlier conversation summary"),
            *restore_messages([message.to_dict() for message in reloaded.messages]),
        ],
        "archive_001",
    )
    candidate = ContextAssembler(OVContextConfig(context_window=128000)).refresh(context, snapshot)

    assert reloaded.messages[0].to_dict()["source_message_ids"] == ["first-question"]
    assert reloaded.messages[-1].to_dict()["source_message_ids"] == ["current-question"]
    assert reloaded.messages[-1].to_dict()["message_kind"] == "user_query"
    assert reloaded.messages[-1].id != "current-question"
    assert snapshot.messages[1].id == "first-question"
    assert snapshot.messages[-1].id == "current-question"
    assert candidate.messages[-1].id == "current-question"
    assert candidate.messages[1].content == content
    assert ("[参考记忆]" in str(candidate.messages[-1].content)) is recall
    assert all(a is b for a, b in zip(written.model_messages, raw, strict=True))
    assert raw == original

    candidate.end_run()
    assert candidate.messages[-1] is raw[-1]


def test_user_source_id_does_not_replace_tool_result_record_identity():
    data = record(sources=["native-user"])
    data["parts"].append(
        {
            "type": "tool",
            "tool_id": "call",
            "tool_name": "lookup",
            "tool_status": "completed",
            "tool_output": "result",
        }
    )
    user, tool = restore_messages([data])
    assert user.id == "native-user"
    assert tool.id == "ov-record:call"


def test_checkpoint_id_collision_is_not_a_current_user_match():
    user = HumanMessage(id="current", content="继续")
    context = OVContext("s", messages=[user])
    context.begin_run("r", user)
    data = record("checkpoint", ["archived-record"])
    data["id"] = "current"
    snapshot = ContextSnapshot(restore_messages([data]), "archive_001")
    with pytest.raises(ContextReadError, match="collid"):
        ContextAssembler(OVContextConfig()).refresh(context, snapshot)


@pytest.mark.parametrize("kind,sources", [("user_query", ["one", "two"]), (None, ["current"])])
def test_ov_record_id_collision_cannot_override_non_native_source_metadata(kind, sources):
    user = HumanMessage(id="current", content="继续")
    context = OVContext("s", messages=[user])
    context.begin_run("r", user)
    data = record(kind, sources)
    data["id"] = "current"
    snapshot = ContextSnapshot(restore_messages([data]), "archive_001")
    with pytest.raises(ContextReadError, match="collid"):
        ContextAssembler(OVContextConfig()).refresh(context, snapshot)


def record(kind="user_query", sources=None, *, role="user"):
    return {
        "id": "ov-record",
        "role": role,
        "parts": [{"type": "text", "text": "继续"}],
        "message_kind": kind,
        "source_message_ids": sources,
    }


@pytest.mark.parametrize(
    "kind,sources,expected",
    [
        ("user_query", ["native-user"], "native-user"),
        ("user_query", ["one", "two"], "ov-record"),
        ("user_query", [], "ov-record"),
        ("checkpoint", ["native-user"], "ov-record"),
        ("tool_transport", ["native-user"], "ov-record"),
        (None, ["native-user"], "ov-record"),
    ],
)
def test_only_one_to_one_user_query_recovers_native_id(kind, sources, expected):
    message = restore_messages([record(kind, sources)])[0]
    assert message.id == expected
    assert message.additional_kwargs["openviking_message_id"] == "ov-record"


@pytest.mark.parametrize(
    "kind,sources",
    [
        ("user_query", ["another-question"]),
        ("user_query", ["one", "two"]),
        ("checkpoint", ["current"]),
        ("checkpoint", None),
    ],
)
def test_identified_or_checkpoint_text_is_not_rebound_to_an_unrelated_current_user(kind, sources):
    from openviking.integrations.langchain.ov_context.assembler import is_user_anchor

    user = HumanMessage(id="current", content="继续")
    context = OVContext("s", messages=[user])
    context.begin_run("r", user)
    snapshot = ContextSnapshot(
        [
            HumanMessage(id="ov_summary:archive_001", content="summary"),
            *restore_messages([record(kind, sources)]),
        ],
        "archive_001",
    )
    candidate = ContextAssembler(OVContextConfig()).refresh(context, snapshot)
    assert len(candidate.messages) == 3
    assert candidate.messages[1].id == "current" and is_user_anchor(candidate.messages[1])
    assert candidate.messages[2].id != "current"
    with pytest.raises(ContextReadError):
        context.replace(snapshot.messages, snapshot.archive_id)


def test_legacy_message_round_trip_stays_unchanged():
    legacy = {
        "id": "ov-old",
        "role": "user",
        "parts": [{"type": "text", "text": "旧消息"}],
        "created_at": "2026-10-01T00:00:00Z",
    }
    assert Message.from_dict(legacy).to_dict() == legacy
    assert restore_messages([legacy])[0].id == "ov-old"


@pytest.mark.parametrize("bad", ["native", [1], [""], ["  "]])
def test_malformed_source_ids_in_snapshot_are_rejected(bad):
    with pytest.raises(ContextReadError, match="source_message_ids"):
        restore_messages([record(sources=bad)])


@pytest.mark.parametrize("method", ["single", "batch"])
async def test_local_client_preserves_source_identity(native_api, method):
    session, _ = native_api
    client = LocalClient.__new__(LocalClient)
    client._ctx, client._legacy_agent_id = session.ctx, None
    client._service = SimpleNamespace(
        sessions=SimpleNamespace(open=AsyncMock(return_value=session))
    )
    if method == "single":
        await client.add_message(
            session.session_id,
            "user",
            "继续",
            message_kind="user_query",
            source_message_ids=["local-user"],
        )
    else:
        await client.batch_add_messages(
            session.session_id,
            [
                {
                    "role": "user",
                    "content": "继续",
                    "message_kind": "user_query",
                    "source_message_ids": ["local-user"],
                }
            ],
        )
    stored = session.messages[-1].to_dict()
    assert stored.get("source_message_ids") == ["local-user"]
    assert stored.get("message_kind") == "user_query"


async def test_single_http_sdk_preserves_source_identity(native_api):
    session, client = native_api
    await client.add_message(
        session.session_id,
        "user",
        "继续",
        message_kind="user_query",
        source_message_ids=["http-user"],
    )
    stored = session.messages[-1].to_dict()
    assert stored.get("source_message_ids") == ["http-user"]
    assert stored.get("message_kind") == "user_query"


async def test_sync_http_sdk_forwards_source_identity():
    client = SyncHTTPClient.__new__(SyncHTTPClient)
    client._async_client = AsyncMock()
    client._async_client.add_message.return_value = {"added": 1}
    await asyncio.to_thread(
        client.add_message,
        "s",
        "user",
        "继续",
        message_kind="user_query",
        source_message_ids=["sync-user"],
    )
    assert client._async_client.add_message.await_args.kwargs == {
        "message_kind": "user_query",
        "source_message_ids": ["sync-user"],
    }


@pytest.mark.parametrize("parts", [False, True])
async def test_client_session_wrapper_forwards_source_identity(native_api, parts):
    session, client = native_api
    bound = ClientSession(client, session.session_id, UserIdentifier.the_default_user())
    body = {"parts": [TextPart(text="继续")]} if parts else {"content": "继续"}
    await bound.add_message(
        "user", **body, message_kind="user_query", source_message_ids=["bound-user"]
    )
    assert session.messages[-1].to_dict()["source_message_ids"] == ["bound-user"]


@pytest.mark.parametrize("bad", ["user", [1], [""], [" "]])
async def test_http_rejects_invalid_source_ids_before_writing(native_api, bad):
    session, client = native_api
    response = await client._http.post(
        f"/api/v1/sessions/{session.session_id}/messages/batch",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "继续",
                    "message_kind": "user_query",
                    "source_message_ids": bad,
                }
            ]
        },
    )
    assert response.status_code == 422
    assert not session.messages
