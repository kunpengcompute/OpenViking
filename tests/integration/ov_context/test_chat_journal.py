"""Chat observations recover from the append-only log before API reads."""

import httpx

from examples.deepagents_ov_context.chat.history import RunStatistics
from examples.deepagents_ov_context.chat.server import create_app
from examples.deepagents_ov_context.chat.stream import ActiveRun

from .test_chat_api import new_session


async def test_new_app_replays_logged_events_missing_from_sqlite(chat):
    client, rt = chat
    sid = await new_session(client, user_id="alice")
    session = rt.store.get(sid)
    session.last_run = RunStatistics(run_id="interrupted-run")
    rt.store.history.begin(sid, "interrupted-run", "question")
    rt.store.save(session)
    for ordinal in (1, 2):
        rt.event_log.append(
            {
                "timestamp": "2026-11-06T00:00:00+00:00",
                "event": "memory.recall",
                "event_kind": "stage",
                "account_id": session.account_id,
                "user_id": session.user_id,
                "actor_peer_id": session.agent_id,
                "session_id": sid,
                "thread_id": session.thread_id,
                "run_id": "interrupted-run",
                "sequence": ordinal,
                "elapsed_ms": ordinal * 10,
                "data": {"name": "memory.recall", "count": ordinal},
            }
        )
    rt.event_log.flush()

    fresh = create_app(rt.settings)
    async with fresh.router.lifespan_context(fresh):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fresh), base_url="http://localhost"
        ) as reader:
            response = await reader.get(f"/api/sessions/{sid}/runs/interrupted-run")
            assert response.status_code == 200
            saved = response.json()
            assert [e["ordinal"] for e in saved["events"]] == [1, 2]
            assert [e["data"]["count"] for e in saved["events"]] == [1, 2]
            assert saved["statistics"]["status"] == "interrupted"


async def test_sqlite_projection_failure_recovers_from_log_without_losing_delivery(chat):
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history)
    run.bind_log(session, rt.event_log)
    rt.store.db.execute("PRAGMA query_only = ON")
    try:
        for index in range(12):
            await run.emit("stage", name="observed", index=index)
    finally:
        rt.store.db.execute("PRAGMA query_only = OFF")
    assert run.queue.qsize() == 12
    saved = rt.store.history.get(sid, "r")
    assert [e.ordinal for e in saved.events] == list(range(1, 13))
    assert [e.data["index"] for e in saved.events] == list(range(12))
    assert run.observations_complete is False


async def test_real_process_kill_recovers_unprojected_events_once(chat):
    import asyncio
    import os
    import subprocess
    import sys

    client, rt = chat
    sid = await new_session(client, user_id="crash-user")
    session = rt.store.get(sid)
    session.last_run = RunStatistics(run_id="killed-run")
    rt.store.history.begin(sid, "killed-run", "question")
    rt.store.save(session)
    script = r"""
import asyncio,sys
from pathlib import Path
from examples.deepagents_ov_context.chat.event_log import RunEventLog
from examples.deepagents_ov_context.chat.session import SessionStore
from examples.deepagents_ov_context.chat.stream import ActiveRun
async def main():
    path=Path(sys.argv[1]);sid=sys.argv[2]
    store=SessionStore(path/'sessions.sqlite3')
    log=RunEventLog(path/'logs/runs.jsonl')
    store.history.bind_log(log,store.observation_identity)
    run=ActiveRun(run_id='killed-run',session_id=sid,history=store.history)
    run.bind_log(store.get(sid),log)
    for index in range(3):
        await run.emit('stage',name='before-kill',index=index)
    print('READY',flush=True)
    await asyncio.Event().wait()
asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(rt.settings.data_dir), sid],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    try:

        async def ready():
            while True:
                line = await asyncio.to_thread(process.stdout.readline)
                assert line, "child exited before writing observations"
                if line.strip() == "READY":
                    return

        await asyncio.wait_for(ready(), timeout=30)
        process.kill()
        await asyncio.to_thread(process.wait, timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait, timeout=10)
        process.stdout.close()
    assert process.returncode == -9
    for _ in range(2):
        fresh = create_app(rt.settings)
        async with fresh.router.lifespan_context(fresh):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=fresh), base_url="http://localhost"
            ) as reader:
                saved = (await reader.get(f"/api/sessions/{sid}/runs/killed-run")).json()
                assert [e["ordinal"] for e in saved["events"]] == [1, 2, 3]
                assert [e["data"]["index"] for e in saved["events"]] == [0, 1, 2]
                assert saved["statistics"]["status"] == "interrupted"


async def test_read_flush_and_terminal_are_visible_to_another_store(chat):
    from examples.deepagents_ov_context.chat.session import SessionStore

    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history)
    run.bind_log(session, rt.event_log)
    await run.emit("stage", name="first")
    saved = (await client.get(f"/api/sessions/{sid}/runs/r")).json()
    assert [e["data"]["name"] for e in saved["events"]] == ["first"]
    await run.emit("done", message="complete")
    other = SessionStore(rt.settings.data_dir / "sessions.sqlite3")
    try:
        assert [e.event for e in other.history.get(sid, "r").events] == ["stage", "done"]
    finally:
        other.close()


async def test_torn_log_tail_is_repaired_before_later_observations(chat):
    import json

    from examples.deepagents_ov_context.chat.event_log import RunEventLog

    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history)
    run.bind_log(session, rt.event_log)
    await run.emit("stage", name="完整记录")
    path = rt.settings.data_dir / "logs/runs.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"data":"incomplete\xe4')
    repaired = RunEventLog(path)
    try:
        run.bind_log(session, repaired)
        await run.emit("done", message="下一条")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [e["sequence"] for e in records] == [1, 2]
        assert [e.data for e in rt.store.history.get(sid, "r").events] == [
            {"name": "完整记录"},
            {"message": "下一条"},
        ]
    finally:
        repaired.close()


async def test_log_failure_falls_back_to_sqlite_and_keeps_sequence(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history)
    run.bind_log(session, rt.event_log)
    original = rt.event_log.append

    def fail(record):
        raise OSError("log unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(rt.event_log, "append", fail)
        await run.emit("stage", name="sql-fallback")
    assert rt.event_log.append == original
    await run.emit("done", message="finished")
    saved = rt.store.history.get(sid, "r")
    assert [e.ordinal for e in saved.events] == [1, 2]
    assert saved.events[0].data == {"name": "sql-fallback"}
    assert run.observations_complete is False


async def test_terminal_fsync_failure_keeps_written_event_replayable(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    statistics = RunStatistics(run_id="r", status="done")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history, statistics=statistics)
    run.bind_log(session, rt.event_log)

    def fail():
        raise OSError("fsync unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(rt.event_log, "flush", fail)
        await run.emit("done", statistics=statistics.model_dump())
    delivered = run.queue.get_nowait()
    assert delivered.data["statistics"]["observations_complete"] is False
    assert run.observations_complete is False
    assert statistics.observations_complete is False
    saved = rt.store.history.get(sid, "r")
    assert [(e.ordinal, e.event) for e in saved.events] == [(1, "done")]
    assert len(rt.store.history.get(sid, "r").events) == 1


async def test_replay_preserves_user_isolation(chat):
    client, rt = chat
    sessions = [await new_session(client, user_id=user) for user in ("alice", "bob")]
    for sid in sessions:
        session = rt.store.get(sid)
        rt.store.history.begin(sid, "same-run", sid)
        run = ActiveRun(run_id="same-run", session_id=sid, history=rt.store.history)
        run.bind_log(session, rt.event_log)
        await run.emit("stage", name="owner", user=session.user_id)
    for sid, user in zip(sessions, ("alice", "bob"), strict=True):
        saved = (await client.get(f"/api/sessions/{sid}/runs/same-run")).json()
        assert [e["data"]["user"] for e in saved["events"]] == [user]


async def test_legacy_log_without_kind_or_elapsed_is_replayed_idempotently(chat):
    from datetime import datetime, timezone

    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "legacy", "question")
    rt.event_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "run.start",
            "account_id": session.account_id,
            "user_id": session.user_id,
            "actor_peer_id": session.agent_id,
            "session_id": sid,
            "run_id": "legacy",
            "sequence": 1,
            "data": {"run_id": "legacy"},
        }
    )
    for _ in range(2):
        saved = (await client.get(f"/api/sessions/{sid}/runs/legacy")).json()
        assert [e["event"] for e in saved["events"]] == ["start"]
        assert saved["events"][0]["elapsed_ms"] >= 0


async def test_partial_log_write_rolls_back_and_sqlite_fallback_remains_readable(chat, monkeypatch):
    import os

    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    run = ActiveRun(run_id="r", session_id=sid, history=rt.store.history)
    run.bind_log(session, rt.event_log)
    original = os.write
    writes = 0

    def short_write(fd, payload):
        nonlocal writes
        if fd != rt.event_log.fd:
            return original(fd, payload)
        writes += 1
        if writes == 1:
            return original(fd, payload[:8])
        raise OSError("interrupted append")

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", short_write)
        await run.emit("stage", name="fallback")
    await run.emit("done", message="next valid record")
    saved = rt.store.history.get(sid, "r")
    assert [e.ordinal for e in saved.events] == [1, 2]
    assert saved.events[1].data == {"message": "next valid record"}


async def test_replay_rejects_cross_user_log_record(chat):
    import pytest

    client, rt = chat
    sid = await new_session(client, user_id="alice")
    session = rt.store.get(sid)
    rt.store.history.begin(sid, "r", "question")
    path = rt.event_log.path
    original = path.read_bytes()
    rt.event_log.append(
        {
            "event": "observed",
            "event_kind": "stage",
            "session_id": sid,
            "run_id": "r",
            "account_id": session.account_id,
            "user_id": "bob",
            "actor_peer_id": session.agent_id,
            "sequence": 1,
            "elapsed_ms": 0,
            "data": {"name": "observed"},
        }
    )
    try:
        with pytest.raises(ValueError, match="identity"):
            rt.store.history.get(sid, "r")
    finally:
        # This test's invalid record must not affect the fixture's shutdown.
        path.write_bytes(original)
    assert rt.store.history.get(sid, "r").events == []
