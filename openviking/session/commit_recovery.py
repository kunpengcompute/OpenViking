# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Durable Phase 1 intents and resumable commit completion for the legacy runtime."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

from openviking.message import Message
from openviking.server.error_mapping import is_not_found_error
from openviking_cli.utils import get_logger

logger = get_logger(__name__)
SESSION_LOCK_TIMEOUT = 30.0
INTENT_FILE = ".commit.json"
STATS_FILE = ".commit_stats.json"
# Covers exhausted writes/cancellation while this process remains alive. This
# is not a durable checkpoint: a process loss before a successful write may
# require extraction again, just as a crash during the model operation does.
_pending_completions = {}


class CompletionPendingError(RuntimeError):
    """Extraction succeeded, but its result still needs to be persisted."""


def _completion_key(session, archive_uri):
    return (session.ctx.account_id, session.ctx.user.user_id, archive_uri)


async def save_completion(session, archive_uri, record):
    """Persist an immutable result with three bounded, idempotent attempts."""
    content = json.dumps(record, ensure_ascii=False)
    key = _completion_key(session, archive_uri)
    _pending_completions[key] = record
    for attempt in range(3):
        try:
            await session._viking_fs.write_file(
                uri=f"{archive_uri}/{STATS_FILE}", content=content, ctx=session.ctx
            )
        except Exception as exc:
            if attempt == 2:
                raise CompletionPendingError(
                    f"Commit result awaits persistence: {archive_uri}"
                ) from exc
            await asyncio.sleep(0.1 * (2**attempt))
        else:
            _pending_completions.pop(key, None)
            return


async def read_record(session, archive_uri, filename):
    try:
        raw = await session._viking_fs.read_file(f"{archive_uri}/{filename}", ctx=session.ctx)
    except Exception as exc:
        if is_not_found_error(exc):
            return None
        raise
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid commit record: {archive_uri}/{filename}")
    return value


async def write_record(session, archive_uri, filename, value):
    await session._viking_fs.write_file(
        uri=f"{archive_uri}/{filename}",
        content=json.dumps(value, ensure_ascii=False),
        ctx=session.ctx,
    )


async def marker_exists(session, archive_uri, filename):
    try:
        await session._viking_fs.read_file(f"{archive_uri}/{filename}", ctx=session.ctx)
    except Exception as exc:
        if is_not_found_error(exc):
            return False
        raise
    return True


async def repair_phase1_locked(session):
    """Finish the latest sealed intent before another writer changes root state.

    Every root writer uses the Session mutex. Later archives cannot be allocated
    until this step succeeds, so only the newest archive can have unfinished
    Phase 1. Fresh messages are kept by ID rather than replaced by an old tail.
    Return whether the live file was rewritten so appenders discard old raw data.
    """
    refs = await session._list_archive_refs()
    if not refs:
        return False
    archive = max(refs, key=lambda item: item["index"])
    intent = await read_record(session, archive["archive_uri"], INTENT_FILE)
    if intent is None or intent["status"] != "preparing":
        return False
    await apply_phase1_locked(session, archive["archive_uri"], intent)
    return True


async def apply_phase1_locked(session, archive_uri, intent):
    """Archive raw is durable before any destructive live write, including replay."""
    archived = [Message.from_dict(item) for item in intent["archived_messages"]]
    await session._viking_fs.write_file(
        uri=f"{archive_uri}/messages.jsonl",
        content="".join(message.to_jsonl() + "\n" for message in archived),
        ctx=session.ctx,
    )
    archived_ids = {message.id for message in archived}
    retained = [message for message in session._messages if message.id not in archived_ids]
    session._messages = retained
    await session._write_to_agfs_async(messages=retained)
    session._meta.message_count = len(retained)
    if intent["persist_keep_recent_count"]:
        session._meta.keep_recent_count = intent["keep_recent_count"]
    # Recalculate rather than erase counters for new messages appended after a
    # previous failed root write. The retained window itself is never pending.
    session._rebuild_pending_tokens()
    session._meta.commit_count = max(session._meta.commit_count, intent["archive_index"])
    session._meta.last_commit_at = intent["committed_at"]
    if intent["record_auto_commit_success"]:
        session._meta.last_auto_commit_at = intent["committed_at"]
    await session._save_meta()
    intent["status"] = "ready"
    await write_record(session, archive_uri, INTENT_FILE, intent)


async def publish(session, archive_uri, intent):
    """Register a stable task ID, then dispatch. The intent survives either failure."""
    from openviking.storage.errors import LockAcquisitionError
    from openviking.storage.transaction import LockContext, get_lock_manager

    archive_path = session._viking_fs._uri_to_path(archive_uri, ctx=session.ctx)
    try:
        async with LockContext(get_lock_manager(), [archive_path], lock_mode="exact", timeout=0):
            if await session._get_archive_state(archive_uri) != "pending":
                _pending_completions.pop(_completion_key(session, archive_uri), None)
                return
            await register_task_locked(session, intent)
    except LockAcquisitionError:
        # A recovery worker already owns this durable job. Do not register or
        # overwrite its task outside the lease, nor wait through model work.
        return
    task = asyncio.create_task(run_worker(session, archive_uri))
    session._background_tasks.add(task)
    task.add_done_callback(session._background_tasks.discard)


async def register_task_locked(session, intent):
    """Register only while holding the archive lease and after checking its terminal state."""
    from openviking.service.task_tracker import get_task_tracker

    await get_task_tracker().create(
        "session_commit",
        resource_id=session.session_id,
        account_id=session.ctx.account_id,
        user_id=session.ctx.user.user_id,
        task_id=intent["task_id"],
    )


async def finish_statistics(session, archive_uri, record):
    """Retry completion only; never rerun extraction because of a metadata conflict."""
    from openviking.service.task_tracker import get_task_tracker

    saved = await session._merge_and_save_commit_meta(
        archive_index=record["archive_index"],
        memories_extracted=record["memories_extracted"],
        telemetry_snapshot=SimpleNamespace(summary=record["telemetry_summary"]),
    )
    if not saved:
        return False
    result = dict(record["result"])
    result["token_usage"] = {
        "llm": dict(session._meta.llm_token_usage),
        "embedding": dict(session._meta.embedding_token_usage),
        "total": {
            "total_tokens": session._meta.llm_token_usage["total_tokens"]
            + session._meta.embedding_token_usage["total_tokens"],
            "cached_tokens": session._meta.llm_token_usage["cached_tokens"],
            "reasoning_tokens": session._meta.llm_token_usage["reasoning_tokens"],
        },
    }
    await get_task_tracker().complete(
        record["task_id"],
        result,
        account_id=session.ctx.account_id,
        user_id=session.ctx.user.user_id,
    )
    # Write the terminal marker last. Replay after a task-store error is safe:
    # the root metadata contains an applied-statistics watermark.
    await session._write_done_file(
        archive_uri, record["first_message_id"], record["last_message_id"]
    )
    return True


async def run_worker(session, archive_uri):
    from openviking.storage.errors import LockAcquisitionError
    from openviking.storage.transaction import LockContext, get_lock_manager

    archive_path = session._viking_fs._uri_to_path(archive_uri, ctx=session.ctx)
    try:
        # A separate lease serializes extraction/recovery across workers without
        # holding the root Session mutex throughout any model calls.
        async with LockContext(get_lock_manager(), [archive_path], lock_mode="exact", timeout=0):
            if await session._get_archive_state(archive_uri) != "pending":
                _pending_completions.pop(_completion_key(session, archive_uri), None)
                return
            intent = await read_record(session, archive_uri, INTENT_FILE)
            if intent is None:
                return
            session_path = session._viking_fs._uri_to_path(session._session_uri, ctx=session.ctx)
            async with LockContext(
                get_lock_manager(), [session_path], lock_mode="exact", timeout=SESSION_LOCK_TIMEOUT
            ):
                await session._reload_commit_snapshot_locked()
                await repair_phase1_locked(session)
            intent = await read_record(session, archive_uri, INTENT_FILE)
            if intent["status"] != "ready":
                return
            # Extraction waits for its predecessor without holding the Session
            # mutex. Keep this worker alive rather than relying on a recovery scan.
            statistics = _pending_completions.get(_completion_key(session, archive_uri))
            if statistics is not None:
                # Also repairs a partially written statistics file using the
                # complete in-process result, before trying to parse that file.
                await save_completion(session, archive_uri, statistics)
            else:
                statistics = await read_record(session, archive_uri, STATS_FILE)
            await register_task_locked(session, intent)
            if statistics is not None:
                await finish_statistics(session, archive_uri, statistics)
                return
            await session._run_memory_extraction(
                task_id=intent["task_id"],
                archive_uri=archive_uri,
                messages=[Message.from_dict(item) for item in intent["archived_messages"]],
                usage_records=[_restore_usage(item) for item in intent["usage_records"]],
                first_message_id=intent["archived_messages"][0]["id"],
                last_message_id=intent["archived_messages"][-1]["id"],
                memory_policy=intent["memory_policy"],
            )
    except asyncio.CancelledError:
        raise
    except LockAcquisitionError:
        logger.debug("Session commit already claimed or awaiting its mutex: %s", archive_uri)
    except Exception:
        # Lock conflicts, unreadable metadata and task-store failures leave the
        # durable intent/statistics available to the next recovery scan.
        logger.warning("Session commit awaits recovery: %s", archive_uri, exc_info=True)


def _restore_usage(value):
    from openviking.session.session import Usage

    return Usage(**value)


def snapshot_usage(records):
    return [asdict(record) for record in records]


async def recover_pending(session) -> bool:
    """Restart interrupted or unpublished work, with a non-blocking per-archive claim."""

    complete = True
    for archive in sorted(await session._list_archive_refs(), key=lambda item: item["index"]):
        uri = archive["archive_uri"]
        if await session._get_archive_state(uri) != "pending":
            _pending_completions.pop(_completion_key(session, uri), None)
            continue
        intent = await read_record(session, uri, INTENT_FILE)
        if intent is None:
            complete = False
            continue  # Legacy archives have no restart-safe snapshot.
        await run_worker(session, uri)
        if await session._get_archive_state(uri) == "pending":
            complete = False
    return complete
