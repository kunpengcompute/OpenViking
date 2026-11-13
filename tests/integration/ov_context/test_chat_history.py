"""Run observations survive refresh/restart and remain scoped to their Session."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from examples.deepagents_ov_context.chat.history import RunEvent, RunStatistics
from examples.deepagents_ov_context.chat.server import create_app
from examples.deepagents_ov_context.chat.session import (
    ContextView,
    SessionInfo,
    SessionStore,
)
from examples.deepagents_ov_context.chat.stream import ActiveRun

from .test_chat_api import new_session, parse_events


def session(sid: str) -> SessionInfo:
    return SessionInfo(
        session_id=sid, account_id="account", user_id=sid, agent_id="mail", created_at=0
    )


def test_old_session_database_migrates_without_inventing_traces(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    original = session("old")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, data TEXT)")
        db.execute("INSERT INTO sessions VALUES (?, ?)", ("old", original.model_dump_json()))
    store = SessionStore(path)
    try:
        assert store.get("old") == original
        assert store.history.list("old") == []
    finally:
        store.close()


async def test_persist_before_delivery_even_when_queue_full_and_cancelled(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    store = SessionStore(path)
    store.save(session("s"))
    store.history.begin("s", "r", "question")
    run = ActiveRun(history=store.history, session_id="s", run_id="r", queue=asyncio.Queue(1))
    await run.emit("start", run_id="r")
    run.emit_metric("memory.recall", {"status": "success", "count": 2})
    run.cancel_requested = True
    await run.emit("cancelled", message="stopped")
    await run.emit("token", text="transport only")
    store.close()
    reopened = SessionStore(path)
    try:
        events = reopened.history.get("s", "r").events
        assert [e.event for e in events] == ["start", "stage", "cancelled"]
        assert [e.ordinal for e in events] == [1, 2, 3]
        assert events[1].data["count"] == 2
        assert events[0].elapsed_ms <= events[-1].elapsed_ms
    finally:
        reopened.close()


def test_identical_run_ids_are_scoped_and_unknown_run_is_404(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        for sid in ("alice", "bob"):
            store.save(session(sid))
            store.history.begin(sid, "same-run", sid)
            store.history.append(
                sid,
                "same-run",
                RunEvent(ordinal=1, event="stage", data={"owner": sid}, elapsed_ms=1),
            )
        assert store.history.get("alice", "same-run").events[0].data == {"owner": "alice"}
        assert store.history.get("bob", "same-run").events[0].data == {"owner": "bob"}
        with pytest.raises(HTTPException) as exc:
            store.history.get("alice", "missing")
        assert exc.value.status_code == 404
    finally:
        store.close()


async def test_api_all_model_calls_and_events_survive_new_app(chat, monkeypatch) -> None:
    client, rt = chat
    sid = await new_session(client, user_id="alice")
    other = await new_session(client, user_id="bob")
    model_inputs: list[list[dict[str, Any]]] = []

    class Graph:
        def __init__(self, run: ActiveRun) -> None:
            self.run = run

        async def astream(self, inputs, config, **kwargs):
            trace = config["callbacks"][0]
            self.run.emit_metric("memory.recall", {"status": "success", "count": 1})
            user = HumanMessage(
                id="u",
                content="[参考记忆]\n周报偏好\n\n[用户原文]\nhi",
                additional_kwargs={"ov_context_injected": True},
            )
            ai = AIMessage(
                id="a",
                content="",
                tool_calls=[{"id": "c", "name": "debug_echo", "args": {"text": "hi"}}],
            )
            result = ToolMessage(
                id="t", content="hi", tool_call_id="c", name="debug_echo", status="success"
            )
            messages = [SystemMessage(content="system rules"), user]
            for output in (
                ai,
                AIMessage(
                    id="final",
                    content="done",
                    usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                ),
            ):
                call_id = uuid.uuid4()
                model_inputs.append([message.model_dump(mode="json") for message in messages])
                # Provider/config information is deliberately not copied into the trace.
                await trace.on_chat_model_start(
                    {"api_key": "test-private-value"}, [messages], run_id=call_id
                )
                await trace.on_llm_end(
                    LLMResult(generations=[[ChatGeneration(message=output)]]), run_id=call_id
                )
                yield "updates", {"model": {"messages": [output]}}
                if output is ai:
                    yield "updates", {"tools": {"messages": [result]}}
                    messages = [*messages, ai, result]

    async def context(_, *, graph=None):
        return ContextView(source="checkpoint", messages=[], raw_message_count=0, write_index=0)

    monkeypatch.setattr(rt, "graph", lambda sdk, run, **kwargs: (Graph(run), None))
    monkeypatch.setattr(rt, "context", context)
    run_ids = []
    for prompt in ("first", "second"):
        response = await client.post("/api/chat", json={"session_id": sid, "message": prompt})
        streamed = parse_events(response.text)
        assert streamed[-1][0] == "done"
        rid = streamed[0][1]["run_id"]
        run_ids.append(rid)
        saved = (await client.get(f"/api/sessions/{sid}/runs/{rid}")).json()
        assert saved["statistics"]["status"] == "done"
        assert saved["statistics"]["model_calls"] == 2
        assert saved["statistics"]["total_tokens"] == 12
        assert [(e["event"], e["data"]) for e in saved["events"]] == streamed
        assert [
            e["data"]["messages"] for e in saved["events"] if e["event"] == "model_input"
        ] == model_inputs[-2:]
        assert "test-private-value" not in json.dumps(saved)
        assert (await client.get(f"/api/sessions/{other}/runs/{rid}")).status_code == 404
    assert [r["run_id"] for r in (await client.get(f"/api/sessions/{sid}/runs")).json()] == run_ids[
        ::-1
    ]
    assert (await client.get(f"/api/sessions/{other}/runs")).json() == []
    assert (await client.get("/api/sessions/missing/runs")).status_code == 404
    app = create_app(rt.settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as fresh:
            restored = (await fresh.get(f"/api/sessions/{sid}/runs/{run_ids[0]}")).json()
            assert restored["prompt"] == "first"
            assert restored["statistics"]["status"] == "done"
            assert [
                e["data"]["messages"] for e in restored["events"] if e["event"] == "model_input"
            ] == model_inputs[:2]


@pytest.mark.parametrize("fail_one", [False, True])
async def test_concurrent_chat_runs_keep_sdk_observations_and_usage_scoped(
    chat, monkeypatch, fail_one: bool
) -> None:
    client, rt = chat
    sessions = [
        await new_session(client, user_id=user, execution={"mode": "baseline"})
        for user in ("alice", "bob")
    ]
    hooks = rt.ov_http.event_hooks
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        event_hooks=hooks,
    )
    entered = 0
    ready = asyncio.Event()

    class Graph:
        def __init__(self, run: ActiveRun) -> None:
            self.run = run

        async def astream(self, inputs, config, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=5)
            await rt.ov_http.get(f"http://ov.invalid/observed/{self.run.run_id}")
            trace = config["callbacks"][0]
            call_id = uuid.uuid4()
            await trace.on_chat_model_start({}, [inputs["messages"]], run_id=call_id)
            await asyncio.sleep(0)
            if fail_one and inputs["messages"][0].content == "alice":
                raise RuntimeError("model failed")
            output = AIMessage(
                content="done",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            )
            await trace.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=output)]]), run_id=call_id
            )
            yield "updates", {"model": {"messages": [output]}}

    async def context(_, *, graph=None):
        return ContextView(source="checkpoint", messages=[])

    monkeypatch.setattr(rt, "graph", lambda sdk, run, **kwargs: (Graph(run), None))
    monkeypatch.setattr(rt, "context", context)
    responses = await asyncio.gather(
        *(
            client.post("/api/chat", json={"session_id": sid, "message": user})
            for sid, user in zip(sessions, ("alice", "bob"), strict=True)
        )
    )
    # A request outside either producer must not append to the last finished Run.
    await rt.ov_http.get("http://ov.invalid/outside")
    for index, (sid, response) in enumerate(zip(sessions, responses, strict=True)):
        assert response.status_code == 200
        events = parse_events(response.text)
        rid = events[0][1]["run_id"]
        saved = (await client.get(f"/api/sessions/{sid}/runs/{rid}")).json()
        observed = [
            event["data"]["path"]
            for event in saved["events"]
            if event["event"] == "stage" and event["data"].get("name") == "ov_request"
        ]
        assert observed == [f"/observed/{rid}"]
        inputs = [event for event in saved["events"] if event["event"] == "model_input"]
        assert inputs[0]["data"]["messages"][0]["content"] == ("alice", "bob")[index]
        failed = fail_one and index == 0
        assert events[-1][0] == ("error" if failed else "done")
        assert saved["statistics"]["model_calls"] == 1
        assert saved["statistics"]["total_tokens"] == (None if failed else 12)


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_or_cancelled_run_keeps_inputs_before_failure(
    chat, monkeypatch, failure
) -> None:
    client, rt = chat
    sid = await new_session(client)

    class Graph:
        async def astream(self, inputs, config, **kwargs):
            await config["callbacks"][0].on_chat_model_start(
                {}, [[HumanMessage(content="question")]], run_id=uuid.uuid4()
            )
            raise failure("private provider error")
            yield  # pragma: no cover

    monkeypatch.setattr(rt, "graph", lambda *args, **kwargs: (Graph(), None))
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    events = parse_events(response.text)
    rid = events[0][1]["run_id"]
    saved = (await client.get(f"/api/sessions/{sid}/runs/{rid}")).json()
    terminal = "cancelled" if failure is asyncio.CancelledError else "error"
    assert saved["statistics"]["status"] == terminal
    assert saved["events"][-1]["event"] == terminal
    assert any(e["event"] == "model_input" for e in saved["events"])
    assert "private provider error" not in json.dumps(saved)


async def test_restart_marks_incomplete_run_and_keeps_its_events(chat) -> None:
    client, rt = chat
    sid = await new_session(client)
    run = ActiveRun(history=rt.store.history, session_id=sid)
    rt.store.history.begin(sid, run.run_id, "unfinished")
    original = rt.store.get(sid)
    original.last_run = RunStatistics(run_id=run.run_id)
    rt.store.save(original)
    await run.emit("model_input", messages=[HumanMessage(content="unfinished").model_dump()])
    app = create_app(rt.settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as fresh:
            saved = (await fresh.get(f"/api/sessions/{sid}/runs/{run.run_id}")).json()
            assert saved["statistics"]["status"] == "interrupted"
            assert saved["statistics"]["seconds"] is None
            assert len(saved["events"]) == 1
            assert not app.state.runtime.store.get(sid).needs_restore


def test_unchanged_session_save_preserves_run_without_rewriting_sqlite(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        value = session("unchanged")
        value.last_run = RunStatistics(run_id="r")
        store.history.begin(value.session_id, "r", "question")
        store.save(value)
        changes = store.db.total_changes
        store.save(store.get(value.session_id))
        assert store.db.total_changes == changes
        assert store.history.get(value.session_id, "r").statistics == value.last_run
        value.last_run.seconds = 1.25
        value.last_run.status = "done"
        store.save(value)
        assert store.get(value.session_id).last_run == value.last_run
        assert store.history.get(value.session_id, "r").statistics == value.last_run
    finally:
        store.close()


def test_unchanged_session_save_repairs_stale_run_statistics(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        value = session("repair")
        value.last_run = RunStatistics(run_id="r", status="done", seconds=2)
        # The session may be saved before the corresponding history is imported.
        store.save(value)
        store.history.begin(value.session_id, "r", "question")
        changes = store.db.total_changes
        store.save(value)
        assert store.db.total_changes == changes + 1
        assert store.history.get(value.session_id, "r").statistics == value.last_run
    finally:
        store.close()


def test_save_checks_authoritative_identity_after_another_connection_changes_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.sqlite3"
    store = SessionStore(path)
    other = SessionStore(path)
    try:
        value = session("owner")
        store.save(value)
        replacement = value.model_copy(update={"user_id": "different-owner"})
        # Simulate an externally restored session directory, not a normal save.
        with other.db:
            other.db.execute(
                "UPDATE sessions SET data = ? WHERE id = ?",
                (replacement.model_dump_json(), value.session_id),
            )
        with pytest.raises(HTTPException) as exc:
            store.save(value)
        assert exc.value.status_code == 403
        assert store.get(value.session_id).user_id == "different-owner"
    finally:
        other.close()
        store.close()
