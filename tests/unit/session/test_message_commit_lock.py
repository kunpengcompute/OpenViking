"""Exercise production message entrypoints against a paused archive writer."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.client.local import LocalClient
from openviking.message import Message, TextPart
from openviking.server.routers import sessions as routes
from openviking.service.session_service import SessionService
from openviking.session.auto_commit_policy import AutoCommitPolicy
from openviking.session.session import Session
from openviking.storage.transaction import LockManager
from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig
from tests.transaction.conftest import MemoryAgfs


class MemoryFS:
    def __init__(self):
        self.files = {}
        self.archive_started = asyncio.Event()
        self.finish_archive = asyncio.Event()

    async def exists(self, uri, *, ctx):
        return uri in self.files or any(path.startswith(uri + "/") for path in self.files)

    async def stat(self, uri, *, ctx, skip_count=False, include_lock_status=True):
        if not await self.exists(uri, ctx=ctx):
            raise FileNotFoundError(uri)
        return {"isDir": True}

    def _uri_to_path(self, uri, *, ctx):
        suffix = uri.split("/history/", 1)
        return "/sessions/test" + ("/history/" + suffix[1] if len(suffix) == 2 else "")

    async def ls(self, uri, *, ctx, **kwargs):
        prefix = uri + "/"
        names = {
            path[len(prefix) :].split("/")[0] for path in self.files if path.startswith(prefix)
        }
        return [{"name": name} for name in sorted(names)]

    async def read_file(self, uri, *, ctx):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, *, ctx):
        if "/history/" in uri and uri.endswith("/messages.jsonl"):
            self.archive_started.set()
            await self.finish_archive.wait()
        self.files[uri] = content

    async def append_file(self, uri, content, *, ctx):
        self.files[uri] = self.files.get(uri, "") + content


def install_session_locks(monkeypatch, *, timeout=3.0):
    import openviking.storage.transaction as transaction

    agfs = MemoryAgfs()
    agfs.mkdir("/sessions")
    agfs.mkdir("/sessions/test")
    agfs.mkdir("/sessions/test/history")
    manager = LockManager(agfs, lock_timeout=timeout, redo_recovery_enabled=False)
    monkeypatch.setattr(transaction, "get_lock_manager", lambda: manager)
    return manager


@pytest.mark.parametrize("entry", ["http", "local"])
async def test_context_read_waits_for_archive_and_never_duplicates_moved_messages(
    monkeypatch, entry
):
    import openviking.service.task_tracker as tracking

    fs = MemoryFS()
    fs.finish_archive.set()
    session = Session(viking_fs=fs, session_id="test")
    message = Message(id="old", role="user", parts=[TextPart(text="old")])
    session._messages = [message]
    fs.files[f"{session.uri}/messages.jsonl"] = message.to_jsonl() + "\n"
    session.meta.message_count = 1
    await session._save_meta()
    install_session_locks(monkeypatch)
    monkeypatch.setattr(session, "_run_memory_extraction", AsyncMock())
    monkeypatch.setattr(
        tracking,
        "get_task_tracker",
        lambda: SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(task_id="task-1"))),
    )
    service = SimpleNamespace(sessions=SessionService(viking_fs=fs))
    monkeypatch.setattr(routes, "get_service", lambda: service)
    clearing, release = asyncio.Event(), asyncio.Event()
    original = session._write_to_agfs_async

    async def pause(*, messages):
        clearing.set()
        await release.wait()
        await original(messages=messages)

    monkeypatch.setattr(session, "_write_to_agfs_async", pause)
    committing = asyncio.create_task(session.commit_async())
    await asyncio.wait_for(clearing.wait(), 3)
    if entry == "http":
        reading = asyncio.create_task(routes.get_session_context("test", 128000, session.ctx))
    else:
        client = object.__new__(LocalClient)
        client._service, client._ctx = service, session.ctx
        reading = asyncio.create_task(client.get_session_context("test"))
    try:
        await asyncio.sleep(0.05)
        assert not reading.done(), (
            "Context observed an archive prefix before the live tail was replaced"
        )
        release.set()
        await committing
        response = await reading
        result = response.result if entry == "http" else response
        assert [m["id"] for m in result["messages"]] == ["old"]
    finally:
        release.set()
        await asyncio.gather(committing, reading, return_exceptions=True)
        if session._background_tasks:
            await asyncio.gather(*list(session._background_tasks))


@pytest.mark.parametrize("entrypoint", ["local_single", "local_batch", "http_single", "http_batch"])
async def test_message_waits_for_archive_and_preserves_fresh_metadata(monkeypatch, entrypoint):
    import openviking.service.task_tracker as tracking

    fs = MemoryFS()
    fs.finish_archive.set()
    commit_session = Session(viking_fs=fs, session_id="test")
    writer_session = Session(viking_fs=fs, session_id="test", ctx=commit_session.ctx)
    old = Message(id="old", role="user", parts=[TextPart(text="old")])
    commit_session._messages = [old]
    commit_session.meta.message_count = 1
    commit_session.meta.last_message_at = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).isoformat()
    commit_session._rebuild_pending_tokens()
    uri = commit_session.uri
    fs.files[f"{uri}/messages.jsonl"] = old.to_jsonl() + "\n"
    await commit_session._save_meta()
    await writer_session.load()

    install_session_locks(monkeypatch)
    monkeypatch.setattr(commit_session, "_run_memory_extraction", AsyncMock())
    monkeypatch.setattr(
        tracking,
        "get_task_tracker",
        lambda: SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(task_id="task-1")),
        ),
    )
    service = SimpleNamespace(
        sessions=SimpleNamespace(
            get=AsyncMock(return_value=writer_session),
            open=AsyncMock(return_value=writer_session),
            maybe_schedule_auto_commit=AsyncMock(return_value=False),
        )
    )
    local = object.__new__(LocalClient)
    local._service = service
    local._ctx = writer_session.ctx
    local._legacy_agent_id = None
    monkeypatch.setattr(routes, "get_service", lambda: service)

    clear_started = asyncio.Event()
    finish_clear = asyncio.Event()
    original_write = commit_session._write_to_agfs_async

    async def pause_clear(*, messages):
        clear_started.set()
        await finish_clear.wait()
        await original_write(messages)

    monkeypatch.setattr(commit_session, "_write_to_agfs_async", pause_clear)
    committing = asyncio.create_task(commit_session.commit_async(pre_commit_check=lambda _: True))
    await asyncio.wait_for(clear_started.wait(), timeout=3)
    if entrypoint == "local_single":
        operation = local._add_message_impl("test", "user", "new", None, None, None)
    elif entrypoint == "local_batch":
        operation = local._batch_add_messages_impl("test", [{"role": "user", "content": "new"}])
    elif entrypoint == "http_single":
        operation = routes.add_message(
            routes.AddMessageRequest(role="user", content="new"),
            "test",
            writer_session.ctx,
        )
    else:
        operation = routes.batch_add_messages(
            routes.BatchAddMessageRequest(
                messages=[routes.AddMessageRequest(role="user", content="new")]
            ),
            "test",
            writer_session.ctx,
        )
    writing = asyncio.create_task(operation)
    try:
        await asyncio.sleep(0.05)
        assert not writing.done(), "Message write bypassed the commit's session lock"
        finish_clear.set()
        await asyncio.wait_for(writing, timeout=3)
        fresh_timestamp = json.loads(fs.files[f"{uri}/.meta.json"])["last_message_at"]
        fs.finish_archive.set()
        result = await asyncio.wait_for(committing, timeout=3)
        if commit_session._background_tasks:
            await asyncio.gather(*list(commit_session._background_tasks))
        live = [
            Message.from_dict(json.loads(line))
            for line in fs.files[f"{uri}/messages.jsonl"].splitlines()
        ]
        meta = json.loads(fs.files[f"{uri}/.meta.json"])
        assert result["archived"] is True
        assert [message.content for message in live] == ["new"]
        assert meta["message_count"] == 1
        assert meta["pending_tokens"] == sum(message.estimated_tokens for message in live)
        assert meta["last_message_at"] == fresh_timestamp
        assert meta["commit_count"] == 1
        archived = fs.files[f"{result['archive_uri']}/messages.jsonl"]
        assert [json.loads(line)["id"] for line in archived.splitlines()] == ["old"]
    finally:
        finish_clear.set()
        fs.finish_archive.set()
        await asyncio.wait_for(
            asyncio.gather(committing, writing, return_exceptions=True), timeout=5
        )


@pytest.mark.parametrize("empty_snapshot", [False, True])
async def test_real_message_write_cancels_stale_idle_commit(monkeypatch, empty_snapshot):
    fs = MemoryFS()
    committing = Session(viking_fs=fs, session_id="test")
    writer = Session(viking_fs=fs, session_id="test", ctx=committing.ctx)
    old = Message(id="old", role="user", parts=[TextPart(text="old")])
    committing._messages = [old]
    committing.meta.message_count = 1
    committing.meta.last_message_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    committing.meta.auto_commit_policy_configured = True
    committing.meta.auto_commit_policy = AutoCommitPolicy(idle_timeout_seconds=3600).to_dict()
    committing._rebuild_pending_tokens()
    uri = committing.uri
    fs.files[f"{uri}/messages.jsonl"] = old.to_jsonl() + "\n"
    await committing._save_meta()
    await writer.load()
    service = SessionService(viking_fs=fs)
    service.set_session_auto_commit_config(SessionAutoCommitConfig(enabled=True))
    assert service._should_run_auto_commit(committing, "idle_timeout")
    install_session_locks(monkeypatch)
    await asyncio.to_thread(writer.add_message, "user", [TextPart(text="new")])
    meta_before = fs.files[f"{uri}/.meta.json"]
    if empty_snapshot:
        committing._messages = []
    result = await committing.commit_async(
        pre_commit_check=lambda session: service._should_run_auto_commit(session, "idle_timeout"),
    )
    assert result["status"] == "skipped"
    assert result["archived"] is False
    assert result["reason"] == "commit_condition_changed"
    assert fs.files[f"{uri}/.meta.json"] == meta_before
    live = [
        Message.from_dict(json.loads(line))
        for line in fs.files[f"{uri}/messages.jsonl"].splitlines()
    ]
    assert [message.content for message in live] == ["old", "new"]


async def test_background_meta_merge_serializes_with_message_and_policy_writes(monkeypatch):
    fs = MemoryFS()
    background = Session(viking_fs=fs, session_id="test")
    writer = Session(viking_fs=fs, session_id="test", ctx=background.ctx)
    policy_writer = Session(viking_fs=fs, session_id="test", ctx=background.ctx)
    fs.files[f"{background.uri}/messages.jsonl"] = ""
    await background._save_meta()
    install_session_locks(monkeypatch)
    read_started = asyncio.Event()
    finish_read = asyncio.Event()
    original_read = fs.read_file
    paused = False

    async def pause_meta_read(uri, *, ctx):
        nonlocal paused
        snapshot = await original_read(uri, ctx=ctx)
        if uri.endswith("/.meta.json") and not paused:
            paused = True
            read_started.set()
            await finish_read.wait()
        return snapshot

    monkeypatch.setattr(fs, "read_file", pause_meta_read)
    merging = asyncio.create_task(background._merge_and_save_commit_meta(1, {"profile": 1}, None))
    await asyncio.wait_for(read_started.wait(), timeout=3)
    writing = asyncio.create_task(
        asyncio.to_thread(writer.add_message, "user", [TextPart(text="new")])
    )
    updating = asyncio.create_task(
        policy_writer.update_auto_commit_policy({"idle_timeout_seconds": 600})
    )
    try:
        await asyncio.sleep(0.05)
        assert not writing.done()
        assert not updating.done()
    finally:
        finish_read.set()
        await asyncio.wait_for(asyncio.gather(merging, writing, updating), timeout=5)
    meta = json.loads(fs.files[f"{background.uri}/.meta.json"])
    assert meta["message_count"] == 1
    assert meta["pending_tokens"] > 0
    assert meta["last_message_at"]
    assert meta["commit_count"] == 1
    assert meta["memories_extracted"]["profile"] == 1
    assert meta["auto_commit_policy"]["idle_timeout_seconds"] == 600


async def test_session_write_explicitly_waits_despite_global_fail_fast_default(monkeypatch):
    from openviking.storage.transaction import LockContext

    fs = MemoryFS()
    session = Session(viking_fs=fs, session_id="test")
    fs.files[f"{session.uri}/messages.jsonl"] = ""
    await session._save_meta()
    manager = install_session_locks(monkeypatch, timeout=0.0)
    async with LockContext(manager, ["/sessions/test"], lock_mode="exact"):
        writing = asyncio.create_task(
            session.add_messages_async([{"role": "user", "parts": [TextPart(text="new")]}])
        )
        await asyncio.sleep(0.05)
        assert not writing.done()
    await asyncio.wait_for(writing, timeout=3)
    assert len(session.messages) == 1
