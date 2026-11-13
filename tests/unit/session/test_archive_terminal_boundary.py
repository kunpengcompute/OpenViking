# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Terminal versions and message coverage through production session interfaces."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from openviking.client.local import LocalClient
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.identity import OVIdentity
from openviking.integrations.langchain.ov_context.reader import OVContextReader
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.state import OVContext
from openviking.message import Message, TextPart
from openviking.server.routers import sessions
from openviking.service.session_service import SessionService
from openviking.session.session import Session
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError, UnavailableError


class ArchiveFS:
    """Persisted files only; version and context assembly use production code."""

    def __init__(self):
        self.files = {}
        self.errors = {}
        self.on_read = None

    async def ls(self, uri, ctx=None, node_limit=1000):
        if uri in self.errors:
            raise self.errors[uri]
        names = {
            path[len(uri) + 1 :].split("/")[0] for path in self.files if path.startswith(uri + "/")
        }
        return [{"name": name} for name in sorted(names)[:node_limit]]

    async def read_file(self, uri, ctx=None):
        if self.on_read:
            self.on_read(uri)
        if uri in self.errors:
            raise self.errors[uri]
        if uri not in self.files:
            raise NotFoundError(uri, "file")
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None):
        self.files[uri] = content


@pytest.fixture
def session():
    return Session(viking_fs=ArchiveFS(), session_id="terminal-test")


def add_archive(session, index, *, state="pending", overview=""):
    uri = f"{session.uri}/history/archive_{index:03d}"
    message = Message(id=f"raw-{index}", role="user", parts=[TextPart(f"Archive {index} input")])
    session._viking_fs.files[f"{uri}/messages.jsonl"] = message.to_jsonl() + "\n"
    if state == "done":
        session._viking_fs.files[f"{uri}/.done"] = "{}"
    elif state == "failed":
        session._viking_fs.files[f"{uri}/.failed.json"] = '{"error": "extraction failed"}'
    if overview:
        session._viking_fs.files[f"{uri}/.overview.md"] = overview
        session._viking_fs.files[f"{uri}/.abstract.md"] = overview
    session._meta.commit_count = max(session.meta.commit_count, index)
    return uri


def add_live_user(session):
    message = Message(id="current", role="user", parts=[TextPart("Current question")])
    session._messages.append(message)
    return HumanMessage(id=message.id, content=message.content)


def local_client(session):
    client = object.__new__(LocalClient)
    client._ctx = session.ctx
    session_service = SessionService(viking_fs=session._viking_fs)
    session_service.get = AsyncMock(return_value=session)

    # These interface tests start with an in-memory, preloaded Session. Actual
    # reload/Phase 1 serialization is covered in test_message_commit_lock.
    async def preloaded_context(session_id, ctx, token_budget=128_000):
        return await session.get_session_context(token_budget=token_budget)

    session_service.get_context = preloaded_context
    client._service = SimpleNamespace(sessions=session_service)
    return client


async def test_empty_session_explicitly_reports_no_terminal_archive(session):
    assert await session.get_latest_finished_archive_id() is None
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] is None
    assert context["messages"] == []


@pytest.mark.parametrize("state", ["pending", "done", "failed"])
async def test_terminal_id_uses_markers_not_early_commit_count(session, state):
    add_archive(session, 1, state="done", overview="Old overview")
    add_archive(session, 2, state=state, overview="New overview")
    add_live_user(session)
    assert session.meta.commit_count == 2
    # Discover persisted archives even if an instance's compression index is stale.
    assert session.compression.compression_index == 0
    expected_id = "archive_001" if state == "pending" else "archive_002"
    assert await session.get_latest_finished_archive_id() == expected_id
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == expected_id
    assert context["latest_archive_overview"] == (
        "New overview" if state == "done" else "Old overview"
    )
    assert [m["id"] for m in context["messages"]] == (
        ["current"] if state == "done" else ["raw-2", "current"]
    )
    if state == "failed":
        assert context["stats"]["totalArchives"] == 2
        assert context["stats"]["failedArchives"] == 1


@pytest.mark.parametrize("state", ["pending", "done", "failed"])
async def test_no_usable_overview_preserves_archived_input(session, state):
    add_archive(session, 1, state=state)
    add_live_user(session)
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == (None if state == "pending" else "archive_001")
    assert context["latest_archive_overview"] == ""
    assert [m["id"] for m in context["messages"]] == ["raw-1", "current"]


async def test_newest_summary_less_success_keeps_old_overview_and_raw_messages(session):
    add_archive(session, 1, state="done", overview="Old overview")
    add_archive(session, 2, state="done")
    add_archive(session, 3, state="pending")
    add_live_user(session)
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == "archive_002"
    assert context["latest_archive_overview"] == "Old overview"
    assert [m["id"] for m in context["messages"]] == ["raw-2", "raw-3", "current"]


async def test_archive_ids_are_ordered_numerically(session):
    add_archive(session, 999, state="done", overview="Older overview")
    add_archive(session, 1000, state="failed")
    add_archive(session, 1001, state="pending")
    assert await session.get_latest_finished_archive_id() == "archive_1000"


async def test_long_session_does_not_truncate_archive_discovery(session):
    for index in range(1, 2001):
        session._viking_fs.files[f"{session.uri}/history/archive_{index:03d}/messages.jsonl"] = ""
    session._viking_fs.files[f"{session.uri}/history/archive_2000/.done"] = "{}"
    assert await session.get_latest_finished_archive_id() == "archive_2000"


async def test_metadata_query_stops_after_the_latest_terminal_archive(session):
    add_archive(session, 1, state="done")
    add_archive(session, 2, state="failed")
    add_archive(session, 3)
    reads = []
    session._viking_fs.on_read = reads.append
    assert await session.get_latest_finished_archive_id() == "archive_002"
    assert not any("archive_001" in uri for uri in reads)


async def test_done_takes_precedence_over_leftover_failure_marker(session):
    uri = add_archive(session, 1, state="failed", overview="Valid overview")
    session._viking_fs.files[f"{uri}/.done"] = "{}"
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == "archive_001"
    assert context["latest_archive_overview"] == "Valid overview"
    assert context["messages"] == []
    assert context["stats"]["failedArchives"] == 0


async def test_terminal_version_survives_new_session_instance_without_persisted_meta_field(session):
    add_archive(session, 1, state="done", overview="Old overview")
    uri = add_archive(session, 2)
    await session._write_failed_marker(uri, stage="memory_extraction", error="Failed")
    reloaded = Session(viking_fs=session._viking_fs, session_id=session.session_id)
    assert "latest_finished_archive_id" not in reloaded.meta.to_dict()
    assert await reloaded.get_latest_finished_archive_id() == "archive_002"
    context = await reloaded.get_session_context()
    assert context["latest_finished_archive_id"] == "archive_002"
    assert context["latest_archive_overview"] == "Old overview"
    assert [m["id"] for m in context["messages"]] == ["raw-2"]


async def test_completion_during_assembly_does_not_mix_versions(session):
    first_uri = add_archive(session, 1, state="done", overview="Old overview")
    next_uri = add_archive(session, 2, overview="New overview")
    add_live_user(session)

    def finish_during_overview_read(uri):
        if uri == f"{first_uri}/.overview.md":
            session._viking_fs.files[f"{next_uri}/.done"] = "{}"

    session._viking_fs.on_read = finish_during_overview_read
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == "archive_001"
    assert context["latest_archive_overview"] == "Old overview"
    assert [m["id"] for m in context["messages"]] == ["raw-2", "current"]
    context = await session.get_session_context()
    assert context["latest_finished_archive_id"] == "archive_002"
    assert context["latest_archive_overview"] == "New overview"
    assert [m["id"] for m in context["messages"]] == ["current"]


@pytest.mark.parametrize("operation", ["history", ".done", ".failed.json"])
@pytest.mark.parametrize("error", [PermissionDeniedError("Denied"), UnavailableError("storage")])
async def test_terminal_discovery_propagates_storage_errors(session, operation, error):
    uri = add_archive(session, 1)
    path = f"{session.uri}/history" if operation == "history" else f"{uri}/{operation}"
    session._viking_fs.errors[path] = error
    with pytest.raises(type(error)):
        await session.get_latest_finished_archive_id()
    with pytest.raises(type(error)):
        await session.get_session_context()


async def test_missing_history_is_not_an_error(session):
    session._viking_fs.errors[f"{session.uri}/history"] = NotFoundError("history")
    assert await session.get_latest_finished_archive_id() is None


async def test_uncovered_message_read_error_is_not_silently_dropped(session):
    uri = add_archive(session, 1, state="failed")
    session._viking_fs.errors[f"{uri}/messages.jsonl"] = PermissionDeniedError("Denied")
    with pytest.raises(PermissionDeniedError):
        await session.get_session_context()


async def test_invalid_archived_json_is_not_silently_dropped(session):
    uri = add_archive(session, 1, state="failed")
    session._viking_fs.files[f"{uri}/messages.jsonl"] = "invalid JSON\n"
    with pytest.raises(json.JSONDecodeError):
        await session.get_session_context()


@pytest.mark.parametrize("state", [None, "pending", "done", "failed"])
async def test_http_and_local_metadata_expose_same_terminal_version(session, monkeypatch, state):
    if state:
        add_archive(session, 1, state=state, overview="Overview")
    expected = "archive_001" if state in ("done", "failed") else None
    client = local_client(session)
    monkeypatch.setattr(sessions, "get_service", lambda: client._service)
    app = FastAPI()
    app.include_router(sessions.router)
    from unittest.mock import AsyncMock

    from openviking.server.routers import tasks

    app.include_router(tasks.router)
    monkeypatch.setattr(
        tasks, "get_task_tracker", lambda: SimpleNamespace(list_tasks=AsyncMock(return_value=[]))
    )

    async def request_context():
        return session.ctx

    app.dependency_overrides[sessions.get_request_context] = request_context
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        response = await http.get(f"/api/v1/sessions/{session.session_id}")
        assert response.status_code == 200
        assert response.json()["result"]["latest_finished_archive_id"] == expected
        context_response = await http.get(f"/api/v1/sessions/{session.session_id}/context")
        assert context_response.status_code == 200
        assert context_response.json()["result"]["latest_finished_archive_id"] == expected
    assert (await client.get_session(session.session_id))["latest_finished_archive_id"] == expected
    assert (await client.get_session_context(session.session_id))[
        "latest_finished_archive_id"
    ] == expected


@pytest.mark.parametrize("state", ["done", "failed"])
async def test_production_reader_and_engine_switch_using_real_session_responses(
    session, state, monkeypatch
):
    add_archive(session, 1, state="done", overview="Old overview")
    uri = add_archive(session, 2, overview="New overview")
    user = add_live_user(session)
    local = local_client(session)
    monkeypatch.setattr(sessions, "get_service", lambda: local._service)
    app = FastAPI()
    app.include_router(sessions.router)
    from unittest.mock import AsyncMock

    from openviking.server.routers import tasks

    app.include_router(tasks.router)
    monkeypatch.setattr(
        tasks, "get_task_tracker", lambda: SimpleNamespace(list_tasks=AsyncMock(return_value=[]))
    )

    async def request_context():
        return session.ctx

    app.dependency_overrides[sessions.get_request_context] = request_context
    client = AsyncHTTPClient(url="http://test", profile_enabled=False)
    client._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    try:
        reader = OVContextReader(client)
        engine = SessionEngine(
            OpenVikingConnection(client=client),
            OVContextConfig(),
            None,
            None,
            context_reader=reader,
        )
        # This client is already backed by the in-memory service; no initialization.
        engine.client.ready = True
        context = OVContext(
            session_id=session.session_id,
            owner=(session.ctx.account_id, session.ctx.user.user_id, "reader", session.session_id),
        )
        identity = OVIdentity(session.ctx.account_id, session.ctx.user.user_id, "reader")
        context = await engine.refresh(context, identity=identity, initial=True)
        context.begin_run("run-1", user)
        context.inject_recall("Remember this", "success")
        injected_user = context.messages[-1]
        assert context.archive_id == "archive_001"
        assert [m.id for m in context.messages] == ["ov_summary:archive_001", "raw-2", "current"]
        await engine.check_summary(context, identity=identity)
        assert context.archive_id == "archive_001"

        if state == "done":
            await session._write_done_file(uri, "raw-2", "raw-2")
        else:
            await session._write_failed_marker(uri, stage="memory_extraction", error="Failed")
        await engine.check_summary(context, identity=identity)
        assert context.archive_id == "archive_002"
        assert context.messages[-1] is injected_user
        assert context.messages[0].content.endswith(
            "New overview" if state == "done" else "Old overview"
        )
        assert [m.id for m in context.messages] == (
            ["ov_summary:archive_002", "current"]
            if state == "done"
            else ["ov_summary:archive_002", "raw-2", "current"]
        )
    finally:
        await client.close()
