"""Failure/restart tests through production Session, task store and path locks."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart, ToolPart
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskStatus, TaskTracker
from openviking.session import commit_recovery
from openviking.session.auto_commit_policy import AutoCommitPolicy
from openviking.session.session import Session
from tests.test_task_tracker import _FakeAgfs
from tests.unit.session.test_message_commit_lock import MemoryFS, install_session_locks


class FaultFS(MemoryFS):
    def __init__(self):
        super().__init__()
        self.finish_archive.set()
        self.fail_write = None
        self.read_failures = 0
        self.write_log = []

    async def read_file(self, uri, *, ctx):
        if uri.endswith("/.meta.json") and "/history/" not in uri and self.read_failures:
            self.read_failures -= 1
            raise TimeoutError("temporary metadata outage")
        return await super().read_file(uri, ctx=ctx)

    async def write_file(self, uri, content, *, ctx):
        if self.fail_write and self.fail_write(uri):
            self.fail_write = None
            raise OSError("injected write failure")
        self.write_log.append(uri)
        await super().write_file(uri=uri, content=content, ctx=ctx)


@pytest.fixture
def runtime(monkeypatch):
    fs = FaultFS()
    install_session_locks(monkeypatch, timeout=0)
    store = PersistentTaskStore(_FakeAgfs())
    tracker = TaskTracker(store=store)
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr("openviking.service.session_service.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                extraction_enabled=False, session_skill_extraction_enabled=False
            ),
        ),
    )
    summary = AsyncMock(return_value="# Summary\n\nA simulated summary.")
    monkeypatch.setattr(Session, "_generate_archive_summary_async", summary)
    return SimpleNamespace(fs=fs, tracker=tracker, store=store, summary=summary)


async def seeded(runtime):
    session = Session(viking_fs=runtime.fs, session_id="test")
    messages = [
        Message(id=f"m{i}", role="user", parts=[TextPart(text=f"message {i}")]) for i in range(3)
    ]
    session._messages = messages
    session.meta.message_count = 3
    session.meta.last_message_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    session.meta.auto_commit_policy_configured = True
    session.meta.auto_commit_policy = AutoCommitPolicy().to_dict()
    session._rebuild_pending_tokens()
    runtime.fs.files[f"{session.uri}/messages.jsonl"] = "".join(
        m.to_jsonl() + "\n" for m in messages
    )
    await session._save_meta()
    return session


async def test_existing_archive_reserves_its_index_like_upstream(runtime):
    session = await seeded(runtime)
    existing = f"{session.uri}/history/archive_001"
    original = (
        Message(id="previous", role="user", parts=[TextPart(text="previous archive")]).to_jsonl()
        + "\n"
    )
    runtime.fs.files[f"{existing}/messages.jsonl"] = original
    runtime.fs.files[f"{existing}/.done"] = "{}"
    assert session.meta.commit_count == 0
    result = await session.commit_async()
    await drain(session)
    assert result["archive_uri"] == f"{session.uri}/history/archive_002"
    assert runtime.fs.files[f"{existing}/messages.jsonl"] == original
    assert f"{result['archive_uri']}/.done" in runtime.fs.files


async def drain(session):
    if session._background_tasks:
        await asyncio.wait_for(asyncio.gather(*list(session._background_tasks)), timeout=5)


def live_ids(fs, session):
    return [
        json.loads(line)["id"] for line in fs.files[f"{session.uri}/messages.jsonl"].splitlines()
    ]


@pytest.mark.parametrize("fault", ["archive_raw", "root_live", "root_meta", "ready_marker"])
async def test_phase1_failure_is_durable_and_recovered_after_new_session(
    runtime, monkeypatch, fault
):
    session = await seeded(runtime)
    archive = f"{session.uri}/history/archive_001"
    predicates = {
        "archive_raw": lambda uri: uri == f"{archive}/messages.jsonl",
        "root_live": lambda uri: uri == f"{session.uri}/messages.jsonl",
        "root_meta": lambda uri: uri == f"{session.uri}/.meta.json",
        "ready_marker": lambda uri: (
            uri == f"{archive}/{commit_recovery.INTENT_FILE}"
            and f"{archive}/messages.jsonl" in runtime.fs.files
        ),
    }
    runtime.fs.fail_write = predicates[fault]
    with pytest.raises(OSError, match="injected"):
        await session.commit_async(keep_recent_count=0, persist_keep_recent_count=False)
    original_ids = set(live_ids(runtime.fs, session))
    record = json.loads(runtime.fs.files[f"{archive}/{commit_recovery.INTENT_FILE}"])
    original_ids.update(item["id"] for item in record["archived_messages"])
    assert original_ids == {"m0", "m1", "m2"}
    # Simulate a new process's Session and task-tracker cache over the same store.
    tracker = TaskTracker(store=runtime.store)
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr("openviking.service.session_service.get_task_tracker", lambda: tracker)
    fresh = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    await commit_recovery.recover_pending(fresh)
    assert live_ids(runtime.fs, session) == []
    assert f"{archive}/.done" in runtime.fs.files
    task = await tracker.get(
        record["task_id"], account_id=session.ctx.account_id, user_id=session.ctx.user.user_id
    )
    assert task.status == TaskStatus.COMPLETED
    assert runtime.summary.await_count == 1
    if fault == "root_meta":
        assert runtime.fs.write_log.index(f"{archive}/messages.jsonl") < runtime.fs.write_log.index(
            f"{session.uri}/messages.jsonl"
        )


async def test_task_registration_failure_is_republished_with_same_task_id(runtime, monkeypatch):
    session = await seeded(runtime)
    create = runtime.tracker.create
    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("task store unavailable")
        return await create(*args, **kwargs)

    monkeypatch.setattr(runtime.tracker, "create", fail_once)
    with pytest.raises(OSError, match="task store"):
        await session.commit_async()
    archive = f"{session.uri}/history/archive_001"
    intent = json.loads(runtime.fs.files[f"{archive}/{commit_recovery.INTENT_FILE}"])
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    task = await runtime.tracker.get(
        intent["task_id"], account_id=session.ctx.account_id, user_id=session.ctx.user.user_id
    )
    assert task.status == TaskStatus.COMPLETED
    assert runtime.summary.await_count == 1
    assert len(runtime.tracker._tasks) == 1


async def test_statistics_retry_preserves_new_policy_and_does_not_repeat_model_or_counts(
    runtime, monkeypatch
):
    session = await seeded(runtime)

    async def summary(*args, **kwargs):
        runtime.fs.read_failures = 3
        return "# Summary\n\nA simulated summary."

    runtime.summary.side_effect = summary
    result = await session.commit_async()
    await drain(session)
    archive = result["archive_uri"]
    assert f"{archive}/{commit_recovery.STATS_FILE}" in runtime.fs.files
    assert f"{archive}/.failed.json" not in runtime.fs.files
    assert f"{archive}/.done" not in runtime.fs.files
    writer = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    await writer.add_messages_async([{"role": "user", "parts": [TextPart(text="new")]}])
    await writer.update_auto_commit_policy(None)
    before = json.loads(runtime.fs.files[f"{session.uri}/.meta.json"])
    stats = json.loads(runtime.fs.files[f"{archive}/{commit_recovery.STATS_FILE}"])
    stats["memories_extracted"] = {"profile": 1}
    runtime.fs.files[f"{archive}/{commit_recovery.STATS_FILE}"] = json.dumps(stats)
    complete = runtime.tracker.complete
    attempts = 0

    async def fail_completion_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("completion store unavailable")
        return await complete(*args, **kwargs)

    monkeypatch.setattr(runtime.tracker, "complete", fail_completion_once)
    fresh = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    await commit_recovery.recover_pending(fresh)
    await commit_recovery.recover_pending(fresh)
    after = json.loads(runtime.fs.files[f"{session.uri}/.meta.json"])
    assert after["auto_commit_policy"] is None
    assert after["pending_tokens"] == before["pending_tokens"]
    assert after["last_message_at"] == before["last_message_at"]
    assert after["message_count"] == 1
    assert after["memories_extracted"]["profile"] == 1
    assert runtime.summary.await_count == 1
    assert f"{archive}/.done" in runtime.fs.files


async def test_short_message_lock_conflict_completes_without_failed_archive(runtime, monkeypatch):
    session = await seeded(runtime)
    writer = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    append = runtime.fs.append_file
    write = runtime.fs.write_file
    locked = asyncio.Event()
    writing = None

    async def delayed_append(*args, **kwargs):
        locked.set()
        await asyncio.sleep(0.2)
        await append(*args, **kwargs)

    monkeypatch.setattr(runtime.fs, "append_file", delayed_append)

    async def delayed_live_write(uri, content, *, ctx):
        if uri == f"{session.uri}/messages.jsonl" and "new" in content:
            locked.set()
            await asyncio.sleep(0.2)
        await write(uri, content, ctx=ctx)

    monkeypatch.setattr(runtime.fs, "write_file", delayed_live_write)

    async def summary(*args, **kwargs):
        nonlocal writing
        writing = asyncio.create_task(
            writer.add_messages_async([{"role": "user", "parts": [TextPart(text="new")]}])
        )
        await locked.wait()
        return "# Summary\n\nA simulated summary."

    runtime.summary.side_effect = summary
    result = await session.commit_async()
    await drain(session)
    await writing
    assert f"{result['archive_uri']}/.done" in runtime.fs.files
    assert f"{result['archive_uri']}/.failed.json" not in runtime.fs.files
    assert len(live_ids(runtime.fs, session)) == 1


async def test_stale_tool_update_keeps_new_messages(runtime):
    session = await seeded(runtime)
    tool = Message(
        id="tool",
        role="assistant",
        parts=[ToolPart(tool_id="call", tool_name="lookup", tool_status="running")],
    )
    runtime.fs.files[f"{session.uri}/messages.jsonl"] = tool.to_jsonl() + "\n"
    stale = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    await stale.load()
    await session.add_messages_async([{"role": "user", "parts": [TextPart(text="new")]}])
    await stale.update_tool_part_async("tool", "call", "done")
    assert len(live_ids(runtime.fs, session)) == 2


async def test_metadata_outage_does_not_restore_disabled_policy(runtime):
    stale = await seeded(runtime)
    updater = Session(viking_fs=runtime.fs, session_id="test", ctx=stale.ctx)
    await updater.update_auto_commit_policy(None)
    before = runtime.fs.files[f"{stale.uri}/.meta.json"]
    runtime.fs.read_failures = 3
    assert await stale._merge_and_save_commit_meta(1, {}, None) is False
    assert runtime.fs.files[f"{stale.uri}/.meta.json"] == before


async def test_cancelled_worker_can_be_resumed_after_restart(runtime):
    session = await seeded(runtime)
    started = asyncio.Event()

    async def blocked_summary(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    runtime.summary.side_effect = blocked_summary
    result = await session.commit_async()
    await asyncio.wait_for(started.wait(), timeout=3)
    workers = list(session._background_tasks)
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    archive = result["archive_uri"]
    assert f"{archive}/.failed.json" not in runtime.fs.files
    runtime.summary.side_effect = None
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert f"{archive}/.done" in runtime.fs.files


async def test_completion_lock_timeout_retries_without_repeating_summary(runtime, monkeypatch):
    from openviking.storage.transaction import get_lock_manager

    session = await seeded(runtime)
    manager = get_lock_manager()
    holder = manager.create_handle()
    monkeypatch.setattr(commit_recovery, "SESSION_LOCK_TIMEOUT", 0.01)

    async def summary(*args, **kwargs):
        assert await manager.acquire_exact_path(holder, "/sessions/test", timeout=0)
        return "# Summary"

    runtime.summary.side_effect = summary
    try:
        result = await session.commit_async()
        await drain(session)
        archive = result["archive_uri"]
        assert f"{archive}/.failed.json" not in runtime.fs.files
        assert f"{archive}/.done" not in runtime.fs.files
        assert f"{archive}/{commit_recovery.STATS_FILE}" in runtime.fs.files
    finally:
        await manager.release(holder)
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert f"{archive}/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1


async def test_disabled_auto_commit_skips_discovery_and_submission(runtime, monkeypatch):
    from openviking.service.session_auto_commit import SessionAutoCommitScheduler
    from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

    session = await seeded(runtime)
    service = SimpleNamespace(
        maybe_schedule_commit_recovery=AsyncMock(),
        maybe_schedule_auto_commit=AsyncMock(),
    )
    scheduler = SessionAutoCommitScheduler(
        service, runtime.fs, SessionAutoCommitConfig(enabled=False)
    )
    path = "/local/default/user/default/sessions/test/.meta.json"

    async def paths(_):
        yield path

    monkeypatch.setattr(scheduler, "_iter_session_meta_paths", paths)
    monkeypatch.setattr(
        scheduler,
        "_get_agfs_client",
        lambda: SimpleNamespace(read=AsyncMock(return_value=json.dumps(session.meta.to_dict()))),
    )
    await scheduler._scan_once()
    service.maybe_schedule_commit_recovery.assert_not_awaited()
    service.maybe_schedule_auto_commit.assert_not_awaited()


async def test_disabled_idle_scan_never_opens_storage(monkeypatch):
    from openviking.service.session_auto_commit import SessionAutoCommitScheduler
    from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

    scheduler = SessionAutoCommitScheduler(
        SimpleNamespace(), SimpleNamespace(), SessionAutoCommitConfig(enabled=False)
    )

    def unexpected_storage_access():
        pytest.fail("Disabled idle scanning must not open the storage client")

    monkeypatch.setattr(scheduler, "_get_agfs_client", unexpected_storage_access)
    await scheduler._scan_once()


@pytest.mark.parametrize("category", ["completed", "recent", "policy_disabled"])
async def test_idle_scan_filters_metadata_before_constructing_session(
    runtime, monkeypatch, category
):
    from openviking.service.session_auto_commit import SessionAutoCommitScheduler
    from openviking.service.session_service import SessionService
    from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

    session = await seeded(runtime)
    meta = session.meta.to_dict()
    if category == "completed":
        meta.update(message_count=0, pending_tokens=0, commit_count=5)
    elif category == "recent":
        meta["last_message_at"] = datetime.now(timezone.utc).isoformat()
    else:
        meta["auto_commit_policy"] = None
    service = SessionService(viking_fs=runtime.fs)
    config = SessionAutoCommitConfig(enabled=True, scan_rate_limit_files_per_second=1_000_000)
    service.set_session_auto_commit_config(config)
    scheduler = SessionAutoCommitScheduler(service, runtime.fs, config)

    async def paths(_):
        for index in range(100):
            yield f"/local/default/user/default/sessions/test-{index}/.meta.json"

    read = AsyncMock(return_value=json.dumps(meta))
    monkeypatch.setattr(scheduler, "_iter_session_meta_paths", paths)
    monkeypatch.setattr(scheduler, "_get_agfs_client", lambda: SimpleNamespace(read=read))
    construct = AsyncMock(side_effect=AssertionError("Ineligible session was loaded"))
    monkeypatch.setattr(service, "get", construct)
    recovery = AsyncMock()
    monkeypatch.setattr(commit_recovery, "recover_pending", recovery)
    await scheduler._scan_once()
    assert read.await_count == 100
    construct.assert_not_awaited()
    recovery.assert_not_awaited()
    assert not service._auto_commit_tasks


async def test_failed_archive_is_not_replayed_by_repeated_idle_scans(runtime, monkeypatch):
    from openviking.service.session_auto_commit import SessionAutoCommitScheduler
    from openviking.service.session_service import SessionService
    from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

    session = await seeded(runtime)
    runtime.summary.side_effect = ValueError("summary failed")
    result = await session.commit_async(keep_recent_count=0)
    await drain(session)
    archive = result["archive_uri"]
    failed = runtime.fs.files[f"{archive}/.failed.json"]
    task = await runtime.tracker.get(result["task_id"])
    assert task.status == TaskStatus.FAILED
    recovery = AsyncMock()
    monkeypatch.setattr(commit_recovery, "recover_pending", recovery)
    service = SessionService(viking_fs=runtime.fs)
    config = SessionAutoCommitConfig(enabled=True, scan_rate_limit_files_per_second=1_000_000)
    service.set_session_auto_commit_config(config)
    scheduler = SessionAutoCommitScheduler(service, runtime.fs, config)

    async def paths(_):
        yield "/local/default/user/default/sessions/test/.meta.json"

    async def read(_):
        return runtime.fs.files[f"{session.uri}/.meta.json"]

    monkeypatch.setattr(scheduler, "_iter_session_meta_paths", paths)
    monkeypatch.setattr(scheduler, "_get_agfs_client", lambda: SimpleNamespace(read=read))
    for _ in range(3):
        await scheduler._scan_once()
        await service.close_background_tasks()
    assert runtime.summary.await_count == 1
    recovery.assert_not_awaited()
    assert runtime.fs.files[f"{archive}/.failed.json"] == failed
    assert f"{archive}/.done" not in runtime.fs.files
    assert (await runtime.tracker.get(result["task_id"])).status == TaskStatus.FAILED


async def seeded_idle(runtime):
    session = await seeded(runtime)
    session.meta.auto_commit_policy["idle_timeout_seconds"] = 600
    await session._save_meta()
    return session


def idle_service(runtime, monkeypatch, session, *, enabled=True):
    from openviking.service.session_service import SessionService
    from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

    service = SessionService(viking_fs=runtime.fs)
    service.set_session_auto_commit_config(SessionAutoCommitConfig(enabled=enabled))
    monkeypatch.setattr(service, "get", AsyncMock(return_value=session))
    monkeypatch.setattr(service, "open", AsyncMock(return_value=session), raising=False)
    return service


@pytest.mark.parametrize("keep_recent_count", [0, 2, 8])
async def test_idle_commit_archives_all_without_changing_saved_keep(
    runtime, monkeypatch, keep_recent_count
):
    session = await seeded_idle(runtime)
    session.meta.keep_recent_count = keep_recent_count
    session.meta.auto_commit_policy["keep_recent_count"] = keep_recent_count
    await session._save_meta()
    service = idle_service(runtime, monkeypatch, session)
    assert await service.maybe_schedule_idle_auto_commit("test", session.ctx)
    await service.close_background_tasks()
    await drain(session)
    meta = json.loads(runtime.fs.files[f"{session.uri}/.meta.json"])
    assert live_ids(runtime.fs, session) == []
    assert meta["keep_recent_count"] == keep_recent_count
    assert meta["auto_commit_policy"]["keep_recent_count"] == keep_recent_count
    archive = f"{session.uri}/history/archive_001"
    assert [
        json.loads(line)["id"]
        for line in runtime.fs.files[f"{archive}/messages.jsonl"].splitlines()
    ] == ["m0", "m1", "m2"]
    assert f"{archive}/.done" in runtime.fs.files
    assert meta["last_auto_commit_at"]
    assert not await service.maybe_schedule_idle_auto_commit("test", session.ctx)
    assert meta["commit_count"] == 1
    runtime.summary.assert_awaited_once()


@pytest.mark.parametrize("skip_reason", ["disabled", "recent-activity", "session-disabled"])
async def test_idle_scheduler_skips_ineligible_sessions(runtime, monkeypatch, skip_reason):
    session = await seeded_idle(runtime)
    if skip_reason == "recent-activity":
        session.meta.last_message_at = datetime.now(timezone.utc).isoformat()
    elif skip_reason == "session-disabled":
        session.meta.auto_commit_policy = None
    await session._save_meta()
    service = idle_service(runtime, monkeypatch, session, enabled=skip_reason != "disabled")
    before = dict(runtime.fs.files)
    assert not await service.maybe_schedule_idle_auto_commit("test", session.ctx)
    assert not service._auto_commit_tasks and not service._auto_commit_inflight
    assert runtime.fs.files == before
    runtime.summary.assert_not_awaited()


@pytest.mark.parametrize("same_owner", [True, False], ids=["own-task", "other-user-task"])
async def test_idle_scheduler_checks_running_tasks_with_owner(runtime, monkeypatch, same_owner):
    session = await seeded_idle(runtime)
    await runtime.tracker.create(
        "session_commit",
        "test",
        account_id=session.ctx.account_id,
        user_id=session.ctx.user.user_id if same_owner else "other-user",
    )
    service = idle_service(runtime, monkeypatch, session)
    assert await service.maybe_schedule_idle_auto_commit("test", session.ctx) is (not same_owner)
    await service.close_background_tasks()
    await drain(session)
    assert session.meta.commit_count == (0 if same_owner else 1)
    assert runtime.summary.await_count == (0 if same_owner else 1)
    assert not service._auto_commit_inflight


async def test_concurrent_idle_scans_archive_session_once(runtime, monkeypatch):
    session = await seeded_idle(runtime)
    service = idle_service(runtime, monkeypatch, session)
    results = await asyncio.gather(
        *(service.maybe_schedule_idle_auto_commit("test", session.ctx) for _ in range(5))
    )
    await service.close_background_tasks()
    await drain(session)
    assert results.count(True) == 1
    assert session.meta.commit_count == 1
    runtime.summary.assert_awaited_once()
    assert not service._auto_commit_inflight


async def test_idle_submission_failure_releases_claim_for_retry(runtime, monkeypatch, caplog):
    from openviking.service.session_service import logger

    caplog.set_level("WARNING", logger=logger.name)
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    session = await seeded_idle(runtime)
    service = idle_service(runtime, monkeypatch, session)
    commit = service.commit_async
    monkeypatch.setattr(service, "commit_async", AsyncMock(side_effect=OSError("unavailable")))
    before = dict(runtime.fs.files)
    assert await service.maybe_schedule_idle_auto_commit("test", session.ctx)
    await service.close_background_tasks()
    assert runtime.fs.files == before
    assert not service._auto_commit_inflight
    assert "Automatic session commit failed" in caplog.text
    runtime.summary.assert_not_awaited()

    monkeypatch.setattr(service, "commit_async", commit)
    assert await service.maybe_schedule_idle_auto_commit("test", session.ctx)
    await service.close_background_tasks()
    await drain(session)
    assert session.meta.commit_count == 1
    assert f"{session.uri}/history/archive_001/.done" in runtime.fs.files
    runtime.summary.assert_awaited_once()


async def test_writer_repairs_incomplete_phase1_before_appending(runtime):
    session = await seeded(runtime)
    runtime.fs.fail_write = lambda uri: uri == f"{session.uri}/.meta.json"
    with pytest.raises(OSError, match="injected"):
        await session.commit_async()
    writer = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    added = await writer.add_messages_async([{"role": "user", "parts": [TextPart(text="fresh")]}])
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert live_ids(runtime.fs, session) == [added[0].id]
    archive = f"{session.uri}/history/archive_001"
    assert [
        json.loads(line)["id"]
        for line in runtime.fs.files[f"{archive}/messages.jsonl"].splitlines()
    ] == ["m0", "m1", "m2"]
    assert f"{archive}/.done" in runtime.fs.files


async def pending_statistics(runtime):
    session = await seeded(runtime)

    async def summary(*args, **kwargs):
        runtime.fs.read_failures = 3
        return "# Summary"

    runtime.summary.side_effect = summary
    result = await session.commit_async()
    await drain(session)
    runtime.summary.side_effect = None
    return session, result["archive_uri"]


async def test_task_start_store_outage_remains_recoverable(runtime, monkeypatch):
    session = await seeded(runtime)
    update = runtime.store.update
    attempts = 0

    async def fail_once(task):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary task state write outage")
        await update(task)

    monkeypatch.setattr(runtime.store, "update", fail_once)
    result = await session.commit_async()
    await drain(session)
    archive = result["archive_uri"]
    assert f"{archive}/.failed.json" not in runtime.fs.files
    assert runtime.summary.await_count == 0
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert f"{archive}/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1


async def test_real_extraction_error_remains_terminal(runtime):
    session = await seeded(runtime)
    runtime.summary.side_effect = ValueError("summary failed")
    result = await session.commit_async()
    await drain(session)
    archive = result["archive_uri"]
    assert f"{archive}/.failed.json" in runtime.fs.files
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert runtime.summary.await_count == 1
    assert f"{archive}/.done" not in runtime.fs.files


async def test_failed_archive_does_not_block_next_commit(runtime):
    first = await seeded(runtime)
    runtime.summary.side_effect = ValueError("first extraction failed")
    failed = await first.commit_async()
    await drain(first)
    old_marker = runtime.fs.files[f"{failed['archive_uri']}/.failed.json"]
    old_raw = runtime.fs.files[f"{failed['archive_uri']}/messages.jsonl"]
    runtime.summary.side_effect = None
    following = Session(viking_fs=runtime.fs, session_id="test", ctx=first.ctx)
    await following.add_messages_async(
        [{"role": "user", "parts": [TextPart(text="next extraction")]}]
    )
    result = await following.commit_async()
    await drain(following)
    assert result["archive_uri"].endswith("archive_002")
    assert f"{result['archive_uri']}/.done" in runtime.fs.files
    assert f"{result['archive_uri']}/.failed.json" not in runtime.fs.files
    assert runtime.fs.files[f"{failed['archive_uri']}/.failed.json"] == old_marker
    assert runtime.fs.files[f"{failed['archive_uri']}/messages.jsonl"] == old_raw
    assert f"{failed['archive_uri']}/.done" not in runtime.fs.files
    assert runtime.summary.await_count == 2
    task = await runtime.tracker.get(
        failed["task_id"], account_id=first.ctx.account_id, user_id=first.ctx.user.user_id
    )
    assert task.status == TaskStatus.FAILED


@pytest.mark.parametrize("same_instance", [False, True])
@pytest.mark.parametrize("predecessor_fails", [False, True])
async def test_overlapping_commit_waits_without_recovery_scan(
    runtime, monkeypatch, same_instance, predecessor_fails
):
    first = await seeded(runtime)
    started, finish_first, following_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def summary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await finish_first.wait()
            if predecessor_fails:
                raise ValueError("predecessor failed")
        return "# Extraction succeeded"

    wait_for_previous = Session._wait_for_previous_archive_terminal

    async def observe_wait(session, archive_index):
        if archive_index > 1:
            following_waiting.set()
        await wait_for_previous(session, archive_index)

    monkeypatch.setattr(Session, "_wait_for_previous_archive_terminal", observe_wait)
    recovery = AsyncMock(side_effect=AssertionError("Normal commits must not need recovery"))
    monkeypatch.setattr(commit_recovery, "recover_pending", recovery)
    runtime.summary.side_effect = summary
    preceding = await first.commit_async(keep_recent_count=0)
    following = (
        first if same_instance else Session(viking_fs=runtime.fs, session_id="test", ctx=first.ctx)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        await following.add_messages_async(
            [{"role": "user", "parts": [TextPart(text="following input")]}]
        )
        result = await following.commit_async(keep_recent_count=0)
        await asyncio.wait_for(following_waiting.wait(), timeout=1)
        assert following._background_tasks
        assert runtime.summary.await_count == 1
        assert (await runtime.tracker.get(result["task_id"])).status == TaskStatus.PENDING
        assert f"{result['archive_uri']}/.done" not in runtime.fs.files
        assert f"{result['archive_uri']}/.failed.json" not in runtime.fs.files
        # Waiting holds only its archive lease; a new message can take the Session mutex.
        writer = Session(viking_fs=runtime.fs, session_id="test", ctx=first.ctx)
        added = await asyncio.wait_for(
            writer.add_messages_async(
                [{"role": "user", "parts": [TextPart(text="message during extraction")]}]
            ),
            timeout=1,
        )
        assert not finish_first.is_set()
    finally:
        finish_first.set()
        await drain(first)
        await drain(following)
    assert runtime.summary.await_count == 2
    assert f"{result['archive_uri']}/.done" in runtime.fs.files
    expected = TaskStatus.FAILED if predecessor_fails else TaskStatus.COMPLETED
    assert (await runtime.tracker.get(preceding["task_id"])).status == expected
    assert (await runtime.tracker.get(result["task_id"])).status == TaskStatus.COMPLETED
    assert not await runtime.tracker.has_running(
        "session_commit", "test", account_id=first.ctx.account_id, user_id=first.ctx.user.user_id
    )
    assert live_ids(runtime.fs, first) == [added[0].id]
    recovery.assert_not_awaited()
    if predecessor_fails:
        assert f"{preceding['archive_uri']}/.failed.json" in runtime.fs.files
        assert f"{preceding['archive_uri']}/.done" not in runtime.fs.files


async def test_three_overlapping_commits_finish_in_order_without_scanning(runtime, monkeypatch):
    first = await seeded(runtime)
    started, release = asyncio.Event(), asyncio.Event()
    waiting = {2: asyncio.Event(), 3: asyncio.Event()}
    calls = 0
    waiting_method = Session._wait_for_previous_archive_terminal

    async def summary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
        return "# Ordered summary"

    async def wait(session, index):
        if index in waiting:
            waiting[index].set()
        await waiting_method(session, index)

    monkeypatch.setattr(Session, "_wait_for_previous_archive_terminal", wait)
    recovery = AsyncMock(side_effect=AssertionError("Unexpected recovery scan"))
    monkeypatch.setattr(commit_recovery, "recover_pending", recovery)
    runtime.summary.side_effect = summary
    sessions = [first]
    results = [await first.commit_async(keep_recent_count=0)]
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        for index in (2, 3):
            session = Session(viking_fs=runtime.fs, session_id="test", ctx=first.ctx)
            sessions.append(session)
            await session.add_messages_async(
                [{"role": "user", "parts": [TextPart(text=f"turn {index}")]}]
            )
            results.append(await session.commit_async(keep_recent_count=0))
            await asyncio.wait_for(waiting[index].wait(), timeout=1)
        assert runtime.summary.await_count == 1
    finally:
        release.set()
        await asyncio.gather(*(drain(session) for session in sessions))
    assert runtime.summary.await_count == 3
    assert [uri for uri in runtime.fs.write_log if uri.endswith("/.done")] == [
        f"{result['archive_uri']}/.done" for result in results
    ]
    for result in results:
        assert (await runtime.tracker.get(result["task_id"])).status == TaskStatus.COMPLETED
    assert not await runtime.tracker.has_running("session_commit", "test")
    recovery.assert_not_awaited()


async def test_failed_terminal_is_not_reopened_by_saved_statistics(runtime):
    session, archive = await pending_statistics(runtime)
    await session._write_failed_marker(archive, stage="memory_extraction", error="terminal failure")
    marker = runtime.fs.files[f"{archive}/.failed.json"]
    metadata = runtime.fs.files[f"{session.uri}/.meta.json"]
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert runtime.fs.files[f"{archive}/.failed.json"] == marker
    assert runtime.fs.files[f"{session.uri}/.meta.json"] == metadata
    assert f"{archive}/.done" not in runtime.fs.files
    assert runtime.summary.await_count == 1


@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("partial_write", [False, True])
async def test_completion_record_write_outage_does_not_fail_or_repeat_extraction(
    runtime, monkeypatch, failures, partial_write
):
    session = await seeded(runtime)
    write = runtime.fs.write_file
    attempts = 0

    async def fail_statistics(uri, content, *, ctx):
        nonlocal attempts
        if uri.endswith("/" + commit_recovery.STATS_FILE):
            attempts += 1
            if attempts <= failures:
                if partial_write:
                    runtime.fs.files[uri] = '{"task_id":'
                raise OSError("temporary result write outage")
        await write(uri, content, ctx=ctx)

    monkeypatch.setattr(runtime.fs, "write_file", fail_statistics)
    result = await session.commit_async()
    await drain(session)
    archive = result["archive_uri"]
    assert f"{archive}/.failed.json" not in runtime.fs.files
    assert runtime.summary.await_count == 1
    if failures == 1:
        assert f"{archive}/.done" in runtime.fs.files
        assert attempts == 2
    else:
        assert attempts == 3
        assert f"{archive}/.done" not in runtime.fs.files
        # New Session, same process: reuse the completed result rather than the model.
        await commit_recovery.recover_pending(
            Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
        )
        assert attempts == 4
        assert f"{archive}/.done" in runtime.fs.files
    task = await runtime.tracker.get(
        result["task_id"], account_id=session.ctx.account_id, user_id=session.ctx.user.user_id
    )
    assert task.status == TaskStatus.COMPLETED
    assert runtime.summary.await_count == 1


async def test_cancelled_completion_write_reuses_result_in_same_process(runtime, monkeypatch):
    session = await seeded(runtime)
    write = runtime.fs.write_file
    started = asyncio.Event()

    async def blocked_statistics(uri, content, *, ctx):
        if uri.endswith("/" + commit_recovery.STATS_FILE):
            started.set()
            await asyncio.Event().wait()
        await write(uri, content, ctx=ctx)

    monkeypatch.setattr(runtime.fs, "write_file", blocked_statistics)
    result = await session.commit_async()
    await asyncio.wait_for(started.wait(), timeout=3)
    workers = list(session._background_tasks)
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    archive = result["archive_uri"]
    assert f"{archive}/.failed.json" not in runtime.fs.files
    monkeypatch.setattr(runtime.fs, "write_file", write)
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    assert f"{archive}/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1


async def test_production_fs_timeout_does_not_repeat_completed_extraction(runtime, monkeypatch):
    from openviking.storage.viking_fs import VikingFS

    session, archive = await pending_statistics(runtime)
    stats_uri = f"{archive}/{commit_recovery.STATS_FILE}"
    read = runtime.fs.read_file
    adapter = SimpleNamespace(
        _ensure_access=lambda *args: None,
        _ctx_or_default=lambda ctx: ctx,
        _uri_to_path=lambda uri, ctx: uri,
        _read_paths=lambda uri, ctx: [uri],
        _read_path_visible=AsyncMock(return_value=True),
        _async_agfs=SimpleNamespace(
            stat=AsyncMock(return_value={"isDir": False}),
            read=AsyncMock(side_effect=TimeoutError("backend timeout")),
        ),
    )
    injected = False

    async def read_with_outage(uri, *, ctx):
        nonlocal injected
        if uri == stats_uri and not injected:
            injected = True
            return await VikingFS.read_file(adapter, uri, ctx=ctx)
        return await read(uri, ctx=ctx)

    monkeypatch.setattr(runtime.fs, "read_file", read_with_outage)
    fresh = Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    await commit_recovery.recover_pending(fresh)
    assert runtime.summary.await_count == 1
    assert f"{archive}/.done" not in runtime.fs.files
    await commit_recovery.recover_pending(fresh)
    assert runtime.summary.await_count == 1
    assert f"{archive}/.done" in runtime.fs.files


async def test_concurrent_completion_is_not_reset_by_recovery_lookup_outage(runtime, monkeypatch):
    session, archive = await pending_statistics(runtime)
    record = json.loads(runtime.fs.files[f"{archive}/{commit_recovery.STATS_FILE}"])
    old_tracker = runtime.tracker
    recovery_tracker = TaskTracker(store=runtime.store)
    read_file = runtime.fs.read_file
    task_read = runtime.store._agfs.read
    completion_injected = False
    lookup_failures = 0

    async def read_done_after_other_worker_finishes(uri, *, ctx):
        nonlocal completion_injected
        if uri == f"{archive}/.done" and not completion_injected:
            completion_injected = True
            monkeypatch.setattr(
                "openviking.service.task_tracker.get_task_tracker", lambda: old_tracker
            )
            await commit_recovery.finish_statistics(session, archive, record)
            monkeypatch.setattr(
                "openviking.service.task_tracker.get_task_tracker", lambda: recovery_tracker
            )
            # This read began before the other worker published its marker.
            raise FileNotFoundError(uri)
        return await read_file(uri, ctx=ctx)

    async def fail_task_lookup_once(path, *args, **kwargs):
        nonlocal lookup_failures
        if not lookup_failures and record["task_id"] in path:
            lookup_failures += 1
            raise TimeoutError("task lookup outage")
        return await task_read(path, *args, **kwargs)

    monkeypatch.setattr(runtime.fs, "read_file", read_done_after_other_worker_finishes)
    monkeypatch.setattr(runtime.store._agfs, "read", fail_task_lookup_once)
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker", lambda: recovery_tracker
    )
    await commit_recovery.recover_pending(
        Session(viking_fs=runtime.fs, session_id="test", ctx=session.ctx)
    )
    monkeypatch.setattr(runtime.store._agfs, "read", task_read)
    task = await TaskTracker(store=runtime.store).get(
        record["task_id"], account_id=session.ctx.account_id, user_id=session.ctx.user.user_id
    )
    assert task.status == TaskStatus.COMPLETED
    assert (
        lookup_failures == 0
    )  # The marker is rechecked under the archive lease before task writes.
