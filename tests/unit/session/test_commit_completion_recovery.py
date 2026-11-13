# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Commit completion retries persisted counts and resumes through service access."""

import asyncio
import json

import pytest

from openviking.service.session_service import SessionService
from openviking.service.task_tracker import TaskStatus
from openviking.session import commit_recovery
from tests.unit.session.test_commit_recovery import drain, seeded


async def fail_live_counts(runtime, monkeypatch, count):
    session = await seeded(runtime)
    original_read = runtime.fs.read_file
    remaining = 0

    async def read_file(uri, *, ctx):
        nonlocal remaining
        if uri == f"{session.uri}/messages.jsonl" and remaining:
            remaining -= 1
            raise TimeoutError("temporary live count outage")
        return await original_read(uri, ctx=ctx)

    async def summary(*args, **kwargs):
        nonlocal remaining
        remaining = count
        return "# Summary\n\nA complete summary."

    monkeypatch.setattr(runtime.fs, "read_file", read_file)
    runtime.summary.side_effect = summary
    receipt = await session.commit_async()
    await drain(session)
    return session, receipt


@pytest.mark.parametrize("failures", [1, 2])
async def test_transient_live_count_retry_finishes_without_reextracting(
    runtime, monkeypatch, failures
):
    session, receipt = await fail_live_counts(runtime, monkeypatch, failures)
    assert f"{receipt['archive_uri']}/.done" in runtime.fs.files
    task = await runtime.tracker.get(
        receipt["task_id"], account_id=session.ctx.account_id, user_id=session.ctx.user.user_id
    )
    assert task.status == TaskStatus.COMPLETED
    assert runtime.summary.await_count == 1
    await session._wait_for_previous_archive_terminal(2)


async def test_session_service_resumes_exhausted_completion_and_caches_terminal_scan(
    runtime, monkeypatch
):
    session, receipt = await fail_live_counts(runtime, monkeypatch, 3)
    archive = receipt["archive_uri"]
    assert f"{archive}/{commit_recovery.STATS_FILE}" in runtime.fs.files
    assert f"{archive}/.done" not in runtime.fs.files
    service = SessionService(viking_fs=runtime.fs)
    await service.get(session.session_id, session.ctx)
    await service.close_background_tasks()
    assert f"{archive}/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1
    meta = json.loads(runtime.fs.files[f"{session.uri}/.meta.json"])
    assert meta["last_statistics_archive_index"] == 1
    original = commit_recovery.recover_pending
    scans = 0

    async def observe(candidate):
        nonlocal scans
        scans += 1
        return await original(candidate)

    monkeypatch.setattr(commit_recovery, "recover_pending", observe)
    for _ in range(3):
        await service.get(session.session_id, session.ctx)
        await service.close_background_tasks()
    assert scans == 0
    assert runtime.summary.await_count == 1


@pytest.mark.parametrize("failure", ["live", "meta"])
async def test_service_access_recovers_intent_before_commit_count_is_saved(runtime, failure):
    session = await seeded(runtime)
    filename = "messages.jsonl" if failure == "live" else ".meta.json"
    runtime.fs.fail_write = lambda uri: uri == f"{session.uri}/{filename}"
    with pytest.raises(OSError, match="injected"):
        await session.commit_async()
    persisted_meta = json.loads(runtime.fs.files[f"{session.uri}/.meta.json"])
    assert persisted_meta["commit_count"] == 0
    service = SessionService(viking_fs=runtime.fs)
    await service.get(session.session_id, session.ctx)
    await service.close_background_tasks()
    assert f"{session.uri}/history/archive_001/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1


async def test_context_access_schedules_recovery_without_blocking_or_duplicate_workers(
    runtime, monkeypatch
):
    session, receipt = await fail_live_counts(runtime, monkeypatch, 3)
    original = commit_recovery.recover_pending
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocked(candidate):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return await original(candidate)

    monkeypatch.setattr(commit_recovery, "recover_pending", blocked)
    service = SessionService(viking_fs=runtime.fs)
    try:
        await asyncio.wait_for(service.get_context(session.session_id, session.ctx), 1)
        await asyncio.wait_for(started.wait(), 1)
        for _ in range(3):
            await asyncio.wait_for(service.get(session.session_id, session.ctx), 1)
        assert calls == 1
        assert f"{receipt['archive_uri']}/.done" not in runtime.fs.files
    finally:
        release.set()
        await service.close_background_tasks()
    assert f"{receipt['archive_uri']}/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1


async def test_failed_commit_invalidates_an_inflight_empty_scan(runtime, monkeypatch):
    session = await seeded(runtime)
    original = commit_recovery.recover_pending
    started = asyncio.Event()
    release = asyncio.Event()
    first = True

    async def paused(candidate):
        nonlocal first
        if first:
            first = False
            complete = await original(candidate)
            started.set()
            await release.wait()
            return complete
        return await original(candidate)

    monkeypatch.setattr(commit_recovery, "recover_pending", paused)
    service = SessionService(viking_fs=runtime.fs)
    await service.get(session.session_id, session.ctx)
    await asyncio.wait_for(started.wait(), 1)
    try:
        runtime.fs.fail_write = lambda uri: uri == f"{session.uri}/.meta.json"
        with pytest.raises(OSError, match="injected"):
            await service.commit_async(session.session_id, session.ctx)
    finally:
        release.set()
        await service.close_background_tasks()
    await service.get(session.session_id, session.ctx)
    await service.close_background_tasks()
    assert f"{session.uri}/history/archive_001/.done" in runtime.fs.files
    assert runtime.summary.await_count == 1
