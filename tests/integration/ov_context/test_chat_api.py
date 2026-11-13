"""Chat HTTP/SSE contracts. Model/service E2E is exercised by the live demo."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage

from examples.deepagents_ov_context.chat.server import (
    ActiveRun,
    ChatRuntime,
    ChatSettings,
    NewSession,
    SessionInfo,
    SessionStore,
    StreamEvent,
    create_app,
)
from openviking.integrations.langchain.ov_context import OVContext
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import OVContextReader
from tests.ov_context_support.commit_settings import EXAMPLE_COMMIT_CONFIG


@pytest_asyncio.fixture
async def chat(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "api_key": "test-private-value",
                "api_base": "https://model.invalid/v1",
                "chat_model": "test-chat",
                "embedding_model": "test-embedding",
                "embedding_dimension": 1024,
            }
        )
    )
    app = create_app(ChatSettings(config, tmp_path, commit_config=EXAMPLE_COMMIT_CONFIG))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            yield client, app.state.runtime


def parse_events(text):
    result = []
    for frame in text.split("\n\n"):
        rows = frame.splitlines()
        event = next((row[7:] for row in rows if row.startswith("event: ")), None)
        data = next((row[6:] for row in rows if row.startswith("data: ")), None)
        if event and data:
            result.append((event, json.loads(data)))
    return result


async def new_session(client, **body):
    result = await client.post("/api/sessions", json=body)
    assert result.status_code == 201
    return result.json()["session_id"]


async def test_homepage_serves_chat_and_session_controls(chat):
    client, _ = chat
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>OVContext · Chat Lab</title>" in response.text
    assert 'id="create-form"' in response.text


@pytest.mark.parametrize("value", ["", "../user", "u\\a", "user\nadmin"])
def test_identity_rejects_header_and_path_values(value):
    with pytest.raises(ValueError):
        NewSession(user_id=value)


def test_thread_is_scoped_by_entire_owner():
    session = SessionInfo(
        session_id="same-id", account_id="a", user_id="u", agent_id="p", created_at=0
    )
    variants = [
        session,
        session.model_copy(update={"account_id": "b"}),
        session.model_copy(update={"user_id": "v"}),
        session.model_copy(update={"agent_id": "q"}),
        session.model_copy(update={"session_id": "other-id"}),
    ]
    assert len({item.thread_id for item in variants}) == 5


def test_session_records_survive_new_runtime_without_secret(tmp_path):
    path = tmp_path / "sessions.db"
    session = SessionInfo(session_id="s", account_id="a", user_id="u", agent_id="p", created_at=0)
    store = SessionStore(path)
    store.save(session)
    store.close()
    store = SessionStore(path)
    try:
        assert store.get("s") == session
        assert store.list() == [session]
    finally:
        store.close()


def test_sse_chinese_and_newline_frame():
    text = "第一行\n第二行🙂"
    frame = StreamEvent("token", {"text": text}).encode(2)
    assert frame.count("\ndata: ") == 1
    assert parse_events(frame) == [("token", {"text": text})]
    assert "id: 2\n" in frame


async def test_create_and_list_and_validation(chat):
    client, _ = chat
    sid = await new_session(client, user_id="alice", agent_id="mail")
    sessions = (await client.get("/api/sessions")).json()
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["user_id"] == "alice"
    assert "test-private-value" not in json.dumps(sessions)
    for message in ("", "   ", "x" * 32_001):
        response = await client.post("/api/chat", json={"session_id": sid, "message": message})
        assert response.status_code == 422
    response = await client.post("/api/chat", json={"session_id": "missing", "message": "hi"})
    assert response.status_code == 404


async def test_browser_cross_origin_rejected(chat):
    client, _ = chat
    response = await client.post(
        "/api/sessions", json={}, headers={"Origin": "https://foreign.example"}
    )
    assert response.status_code == 403
    response = await client.post("/api/sessions", json={}, headers={"Origin": "http://localhost"})
    assert response.status_code == 201
    response = await client.get("/api/sessions", headers={"Host": "foreign.example"})
    assert response.status_code == 400


async def test_new_session_has_empty_remote_history(chat):
    client, rt = chat
    sid = await new_session(client)
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                404, json={"status": "error", "error": {"code": "NOT_FOUND", "message": "missing"}}
            )
        )
    )
    response = await client.get(f"/api/sessions/{sid}")
    assert response.status_code == 200
    assert response.json()["context"]["source"] == "empty"
    response = await client.post(f"/api/sessions/{sid}/restore")
    assert response.status_code == 200


async def test_checkpoint_exposes_native_cache_not_raw_state(chat):
    _, rt = chat
    session = SessionInfo(
        session_id="s", account_id=rt.settings.account_id, user_id="u", agent_id="p", created_at=0
    )
    session.config_snapshot = rt.snapshot(session)
    rt.store.save(session)
    raw_user = HumanMessage(id="u1", content="original")
    cached_ai = AIMessage(
        id="a1",
        content="OV-only",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    sdk = rt.sdk()
    graph, middleware = rt.graph(sdk, ActiveRun())
    context = OVContext(
        session_id="s",
        messages=[cached_ai],
        owner=session.identity("r").session_key,
        write_index=1,
    )
    await graph.aupdate_state(
        session.graph_config,
        {"messages": [raw_user], "ov_context": context.to_state()},
        as_node="OpenVikingContextMiddleware.after_agent",
    )
    await middleware.aclose()
    await sdk.close()
    view = await rt.context(session)
    assert view.source == "checkpoint"
    assert view.raw_message_count == 1
    assert view.messages[0]["type"] == "ai"
    assert view.messages[0]["id"] == "a1"
    assert view.messages[0]["usage_metadata"]["output_tokens"] == 2
    assert view.message_reprs[0].startswith("AIMessage(")
    assert "usage_metadata=" in view.message_reprs[0]
    assert "id='a1'" in view.message_reprs[0]
    assert "original" not in view.model_dump_json()


async def test_sse_done_and_no_duplicate_token_events(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)

    async def produce(self, session, prompt, run):
        await run.emit("start", session_id=session.session_id)
        await run.emit("token", text="你")
        await run.emit("token", text="好")
        await run.emit("message", message={"type": "ai", "content": "你好"})
        await run.emit("done", context={"cached_message_count": 2})
        await run.queue.put(None)

    monkeypatch.setattr(ChatRuntime, "produce", produce)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    assert [kind for kind, _ in events] == ["start", "token", "token", "message", "done"]
    assert "".join(data["text"] for kind, data in events if kind == "token") == "你好"
    assert not rt.active


async def test_same_session_rejected_other_session_runs(chat, monkeypatch):
    client, rt = chat
    one, two = await new_session(client), await new_session(client)
    rt.active[one] = ActiveRun()

    async def produce(self, session, prompt, run):
        await run.emit("done", session_id=session.session_id)
        await run.queue.put(None)

    monkeypatch.setattr(ChatRuntime, "produce", produce)
    assert (
        await client.post("/api/chat", json={"session_id": one, "message": "hi"})
    ).status_code == 409
    assert (await client.post(f"/api/sessions/{one}/restore")).status_code == 409
    assert (
        await client.post("/api/chat", json={"session_id": two, "message": "hi"})
    ).status_code == 200
    rt.active.clear()


async def test_restore_reservation_blocks_new_run(chat):
    client, rt = chat
    sid = await new_session(client)
    rt.restoring.add(sid)
    assert (
        await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    ).status_code == 409
    rt.restoring.clear()


async def test_failed_run_requires_explicit_restore(chat):
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    session.needs_restore = True
    rt.store.save(session)
    assert (
        await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    ).status_code == 409


async def test_failed_restore_retains_checkpoint_and_unlocks(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    sdk = rt.sdk()
    graph, middleware = rt.graph(sdk, ActiveRun())
    context = OVContext(
        session_id=sid,
        messages=[HumanMessage(id="old", content="keep me")],
        owner=session.identity("r").session_key,
    )
    await graph.aupdate_state(
        session.graph_config,
        {"ov_context": context.to_state()},
        as_node="OpenVikingContextMiddleware.after_agent",
    )
    await middleware.aclose()
    await sdk.close()
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"status": "ok", "result": {"session_id": sid}}
            )
        )
    )

    async def unavailable(self, *args, **kwargs):
        raise ContextReadError("private detail must not be exposed")

    monkeypatch.setattr(OVContextReader, "exchange_context", unavailable)
    response = await client.post(f"/api/sessions/{sid}/restore")
    assert response.status_code == 503
    assert "private detail" not in response.text
    assert sid not in rt.restoring
    view = await rt.context(session)
    assert view.source == "checkpoint" and view.messages[0]["content"] == "keep me"


async def test_model_error_is_terminal_and_redacted(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)

    def broken_graph(self, sdk, run, *, browser=None, skill_run=None, session=None):
        raise RuntimeError("test-private-value")

    monkeypatch.setattr(ChatRuntime, "graph", broken_graph)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert parse_events(response.text)[-1][0] == "error"
    assert parse_events(response.text)[-1][1]["error_type"] == "RuntimeError"
    assert "test-private-value" not in response.text
    assert rt.store.get(sid).needs_restore
    assert not rt.active


async def test_output_limit_is_error_instead_of_successful_done(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)

    class LimitedGraph:
        async def astream(self, *args, **kwargs):
            yield (
                "updates",
                {
                    "model": {
                        "messages": [
                            AIMessage(
                                content="partial", response_metadata={"finish_reason": "length"}
                            )
                        ]
                    }
                },
            )

    monkeypatch.setattr(ChatRuntime, "graph", lambda *args, **kwargs: (LimitedGraph(), None))
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error_type"] == "ModelOutputLimitError"
    assert all(kind != "done" for kind, _ in events)
    assert rt.store.get(sid).needs_restore and not rt.active


async def test_cancel_is_idempotent_and_releases_session(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client)
    ready = asyncio.Event()

    async def produce(self, session, prompt, run):
        try:
            ready.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            session.needs_restore = True
            self.store.save(session)
            await run.emit("cancelled", message="stopped")
        finally:
            await run.queue.put(None)

    monkeypatch.setattr(ChatRuntime, "produce", produce)
    request = asyncio.create_task(
        client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    )
    await asyncio.wait_for(ready.wait(), 2)
    assert (await client.post(f"/api/sessions/{sid}/cancel")).json() == {"cancelled": True}
    response = await asyncio.wait_for(request, 2)
    assert parse_events(response.text)[-1][0] == "cancelled"
    assert not rt.active
    assert (await client.post(f"/api/sessions/{sid}/cancel")).json() == {"cancelled": False}


async def test_artifact_routes_retain_session_ownership(chat):
    from examples.deepagents_ov_context.chat.artifacts import FileArtifact

    client, rt = chat
    first = await new_session(client, user_id="alice")
    second = await new_session(client, user_id="bob")
    folder = rt.settings.data_dir / "artifacts" / first / "r-test"
    folder.mkdir(parents=True)
    (folder / "intro.pptx").write_bytes(b"PPTX test file")
    artifact = FileArtifact(
        filename="intro.pptx", run_id="r-test", url="ignored", slides=10, bytes=14
    )
    (folder / "intro.artifact.json").write_text(artifact.model_dump_json())
    items = (await client.get(f"/api/sessions/{first}/artifacts")).json()
    assert len(items) == 1
    download = await client.get(items[0]["url"])
    assert download.status_code == 200 and download.content == b"PPTX test file"
    assert (await client.get(f"/api/sessions/{second}/artifacts")).json() == []
    assert (
        await client.get(f"/api/sessions/{second}/artifacts/r-test/intro.pptx")
    ).status_code == 404
    assert (await client.get("/api/sessions/missing/artifacts")).status_code == 404


async def test_run_statistics_survive_reload_and_reset_each_run(chat, monkeypatch):
    import uuid

    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration, LLMResult

    from examples.deepagents_ov_context.chat.server import ContextView

    client, rt = chat
    sid = await new_session(client)

    class Graph:
        async def astream(self, inputs, config, **kwargs):
            trace = config["callbacks"][0]
            call_id = uuid.uuid4()
            await trace.on_chat_model_start(
                {}, [[HumanMessage(content="question")]], run_id=call_id
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="", additional_kwargs={"reasoning_content": "hidden"}),
                    {"langgraph_node": "model"},
                ),
            )
            yield "messages", (AIMessageChunk(content="你"), {"langgraph_node": "model"})
            yield "messages", (AIMessageChunk(content="好"), {"langgraph_node": "model"})
            message = AIMessage(
                content="你好",
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            )
            result = LLMResult(generations=[[ChatGeneration(message=message)]])
            await trace.on_llm_end(result, run_id=call_id)
            await trace.on_llm_end(result, run_id=call_id)
            yield "updates", {"model": {"messages": [message]}}

    async def context(session, *, graph=None):
        return ContextView(
            source="checkpoint",
            messages=[],
            raw_message_count=2,
            cached_message_count=2,
            write_index=2,
        )

    monkeypatch.setattr(rt, "graph", lambda *args, **kwargs: (Graph(), None))
    monkeypatch.setattr(rt, "context", context)
    first_run = None
    for _ in range(2):
        response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
        events = parse_events(response.text)
        assert events[-1][0] == "done"
        statistics = events[-1][1]["statistics"]
        assert statistics["status"] == "done"
        assert 0 <= statistics["first_token_seconds"] <= statistics["seconds"]
        assert statistics["input_tokens"] == 10 and statistics["output_tokens"] == 4
        assert statistics["total_tokens"] == 14
        assert statistics["usage_calls"] == statistics["model_calls"] == 1
        if first_run:
            assert statistics["run_id"] != first_run
        first_run = statistics["run_id"]
        tokens = [data for kind, data in events if kind == "token"]
        assert tokens[0]["first_token_seconds"] == tokens[1]["first_token_seconds"]
        assert (await client.get("/api/sessions/" + sid)).json()["session"][
            "last_run"
        ] == statistics
        reopened = SessionStore(rt.settings.data_dir / "sessions.sqlite3")
        try:
            assert reopened.get(sid).last_run.model_dump() == statistics
        finally:
            reopened.close()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_or_cancelled_stream_retains_observed_timings(chat, monkeypatch, failure):
    from langchain_core.messages import AIMessageChunk

    client, rt = chat
    sid = await new_session(client)

    class Graph:
        async def astream(self, *args, **kwargs):
            yield "messages", (AIMessageChunk(content="partial"), {"langgraph_node": "model"})
            raise failure("private error")

    monkeypatch.setattr(rt, "graph", lambda *args, **kwargs: (Graph(), None))
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    events = parse_events(response.text)
    expected = "cancelled" if failure is asyncio.CancelledError else "error"
    assert events[-1][0] == expected
    statistics = events[-1][1]["statistics"]
    assert statistics["status"] == expected
    assert 0 <= statistics["first_token_seconds"] <= statistics["seconds"]
    assert statistics["total_tokens"] is None  # Missing usage is never zero.
    assert rt.store.get(sid).last_run.model_dump() == statistics
    assert "private error" not in response.text


async def test_restart_preserves_statistics_and_marks_unfinished_run(chat):
    from examples.deepagents_ov_context.chat.server import RunStatistics

    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    session.last_run = RunStatistics(
        run_id="stopped",
        first_token_seconds=1.2,
        input_tokens=12,
        output_tokens=8,
        total_tokens=20,
        model_calls=2,
        usage_calls=1,
    )
    rt.store.save(session)
    restarted = ChatRuntime(rt.settings, rt.model_config, checkpointer=rt.saver)
    try:
        restored = restarted.store.get(sid)
        assert restored.last_run.status == "interrupted" and not restored.needs_restore
        assert restored.last_run.first_token_seconds == 1.2
        assert restored.last_run.seconds is None
        assert restored.last_run.total_tokens == 20
    finally:
        await restarted.close()


async def test_ov_history_and_model_input_display_native_message_repr(chat, monkeypatch):
    import uuid
    from copy import deepcopy
    from unittest.mock import AsyncMock

    from langchain_core.messages import ToolMessage

    from examples.deepagents_ov_context.chat.server import InputTrace, RunStatistics
    from openviking.integrations.langchain.ov_context.reader import ContextSnapshot

    client, rt = chat
    sid = await new_session(client)
    messages = [
        HumanMessage(id="u", content="你好 <script>plain text</script>"),
        AIMessage(
            id="a", content="", tool_calls=[{"name": "echo", "id": "c", "args": {"text": "你好"}}]
        ),
        ToolMessage(
            id="t", content="你好", tool_call_id="c", status="error", artifact={"local": "value"}
        ),
    ]
    before = deepcopy(messages)
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": {"session_id": sid}})
        )
    )
    monkeypatch.setattr(
        OVContextReader, "exchange_context", AsyncMock(return_value=ContextSnapshot(messages, None))
    )
    response = await client.get(f"/api/sessions/{sid}")
    assert response.status_code == 200
    view = response.json()["context"]
    assert view["source"] == "ov"
    displays = view["message_reprs"]
    assert len(displays) == len(messages)
    display = "\n".join(displays)
    assert displays[0].startswith("HumanMessage(")
    assert displays[1].startswith("AIMessage(")
    assert displays[2].startswith("ToolMessage(")
    assert "AIMessage(" in display and "ToolMessage(" in display
    assert "tool_call_id='c'" in display and "tool_calls=" in display
    assert "status='error'" in display and "artifact={'local': 'value'}" in display
    assert "你好 <script>plain text</script>" in display
    assert [item["type"] for item in view["messages"]] == ["human", "ai", "tool"]
    session = rt.store.get(sid)
    session.last_run = RunStatistics(run_id="r")
    run = ActiveRun()
    trace = InputTrace(run, session, rt.store)
    await trace.on_chat_model_start({}, [messages], run_id=uuid.uuid4())
    assert (await run.queue.get()).event == "statistics"
    start_event = await run.queue.get()
    assert start_event.data["name"] == "model.start"
    event = await run.queue.get()
    assert event.event == "model_input"
    assert event.data["message_reprs"] == displays
    assert event.data["messages"] == view["messages"]
    assert messages == before


@pytest.mark.parametrize("field", ["account_id", "user_id", "agent_id"])
def test_session_directory_rejects_changed_owner_after_restart(tmp_path: Path, field: str) -> None:
    path = tmp_path / "sessions.db"
    original = SessionInfo(
        session_id="same-id", account_id="a", user_id="alice", agent_id="mail", created_at=0
    )
    store = SessionStore(path)
    store.save(original)
    store.close()
    store = SessionStore(path)
    try:
        changed = original.model_copy(update={field: "other"})
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as rejected:
            store.save(changed)
        assert rejected.value.status_code == 403
        with pytest.raises(HTTPException) as rejected:
            store.require_identity(changed.identity("next-run"))
        assert rejected.value.status_code == 403
        assert store.require_identity(original.identity("next-run")) == original
        original.title = "same-owner-update"
        store.save(original)
        assert store.get(original.session_id).title == "same-owner-update"
    finally:
        store.close()


def test_concurrent_application_session_creation_preserves_one_owner(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from fastapi import HTTPException

    path = tmp_path / "sessions.db"
    SessionStore(path).close()
    barrier = Barrier(2)

    def create(peer: str) -> tuple[str, int]:
        store = SessionStore(path)
        try:
            barrier.wait(timeout=5)
            session = SessionInfo(
                session_id="same-id", account_id="a", user_id="alice", agent_id=peer, created_at=0
            )
            try:
                store.save(session)
                return peer, 201
            except HTTPException as exc:
                return peer, exc.status_code
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, ["mail", "doc"]))
    assert sorted(status for _, status in results) == [201, 403]
    store = SessionStore(path)
    try:
        assert store.get("same-id").agent_id == next(p for p, s in results if s == 201)
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["context", "memory_status", "extract_memory", "produce"])
async def test_application_rejects_wrong_peer_before_network_or_graph(
    chat: tuple[httpx.AsyncClient, ChatRuntime], monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from fastapi import HTTPException

    client, rt = chat
    sid = await new_session(client, user_id="alice", agent_id="mail")
    changed = rt.store.get(sid).model_copy(update={"agent_id": "doc"})

    def unexpected_sdk() -> None:
        pytest.fail("SDK must not be created for a different session owner")

    monkeypatch.setattr(rt, "sdk", unexpected_sdk)
    with pytest.raises(HTTPException) as rejected:
        if operation == "produce":
            await rt.produce(changed, "hi", ActiveRun())
        else:
            await getattr(rt, operation)(changed)
    assert rejected.value.status_code == 403
