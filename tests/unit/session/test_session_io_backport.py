"""Public Session entrypoints retain correctness while avoiding redundant reads."""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from openviking.client.local import LocalClient
from openviking.message import Message, TextPart
from openviking.server.auth import get_request_context
from openviking.server.routers import sessions as routes
from openviking.service.session_service import SessionService
from openviking.session.session import Session
from tests.unit.session.test_message_commit_lock import MemoryFS, install_session_locks


class CountingFS(MemoryFS):
    def __init__(self):
        super().__init__()
        self.finish_archive.set()
        self.reads = Counter()
        self.appends = 0
        self.fail_write = None

    async def stat(self, uri, *, ctx, skip_count=False, include_lock_status=True):
        if not await self.exists(uri, ctx=ctx):
            raise FileNotFoundError(uri)
        return {"isDir": True}

    async def read_file(self, uri, *, ctx):
        self.reads[uri] += 1
        return await super().read_file(uri, ctx=ctx)

    async def write_file(self, uri, content, *, ctx):
        if uri == self.fail_write:
            self.fail_write = None
            raise OSError("injected write failure")
        return await super().write_file(uri, content, ctx=ctx)

    async def append_file(self, uri, content, *, ctx):
        self.appends += 1
        if uri == self.fail_write:
            self.fail_write = None
            raise OSError("injected write failure")
        return await super().append_file(uri, content, ctx=ctx)

    async def rm(self, uri, *, recursive=False, ctx):
        assert recursive
        self.files = {
            path: value
            for path, value in self.files.items()
            if path != uri and not path.startswith(uri + "/")
        }


@pytest.fixture
async def io_case(monkeypatch):
    fs = CountingFS()
    session = Session(viking_fs=fs, session_id="test")
    old = Message(id="old", role="user", parts=[TextPart(text="abcd")])
    raw = old.to_dict()
    raw["foreign_field"] = {"preserve": "original formatting"}
    original = json.dumps(raw, ensure_ascii=False, separators=(", ", " : ")) + "\n"
    fs.files[f"{session.uri}/messages.jsonl"] = original
    session._messages = [old]
    session.meta.message_count = 1
    session.meta.total_message_count = 1
    session._rebuild_pending_tokens()
    await session._save_meta()
    install_session_locks(monkeypatch)
    sessions = SessionService(viking_fs=fs)
    sessions.maybe_schedule_auto_commit = AsyncMock(return_value=False)
    service = SimpleNamespace(sessions=sessions)
    monkeypatch.setattr(routes, "get_service", lambda: service)
    local = object.__new__(LocalClient)
    local._service = service
    local._ctx = session.ctx
    local._legacy_agent_id = None
    fs.reads.clear()
    return SimpleNamespace(
        fs=fs, session=session, sessions=sessions, local=local, original=original
    )


@pytest.mark.parametrize("entry", ["http_single", "http_batch", "local_single", "local_batch"])
async def test_message_entrypoint_reads_live_file_once_and_returns_tokens(io_case, entry):
    case = io_case
    if entry == "http_single":
        result = await routes.add_message(
            routes.AddMessageRequest(role="assistant", content="efgh"), "test", case.session.ctx
        )
        receipt = result.result
    elif entry == "http_batch":
        result = await routes.batch_add_messages(
            routes.BatchAddMessageRequest(
                messages=[routes.AddMessageRequest(role="assistant", content="efgh")]
            ),
            "test",
            case.session.ctx,
        )
        receipt = result.result
    elif entry == "local_single":
        receipt = await case.local._add_message_impl("test", "assistant", "efgh", None, None, None)
    else:
        receipt = await case.local._batch_add_messages_impl(
            "test", [{"role": "assistant", "content": "efgh"}]
        )
    live_uri = f"{case.session.uri}/messages.jsonl"
    assert case.fs.reads[live_uri] == 1
    assert case.fs.appends == 0  # append_file would reread the same stored bytes.
    assert receipt["pending_tokens"] == 2
    assert receipt["message_count"] == 2
    assert case.fs.files[live_uri].startswith(case.original)
    assert [
        json.loads(line)["parts"][0]["text"] for line in case.fs.files[live_uri].split("\n") if line
    ] == ["abcd", "efgh"]


@pytest.mark.parametrize(
    "method,path",
    [
        ("list_tool_results", "/tool-results"),
        ("read_tool_result", "/tool-results/r1"),
        ("search_tool_result", "/tool-results/r1/search?q=word"),
        ("get_session_archive", "/archives/archive_001"),
    ],
)
async def test_reference_reads_do_not_preload_live_messages(io_case, monkeypatch, method, path):
    # The referenced-file operation is independent of live Session state.
    monkeypatch.setattr(Session, method, AsyncMock(return_value={"source": "reference"}))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_request_context] = lambda: io_case.session.ctx
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://offline-test"
    ) as client:
        response = await client.get("/api/v1/sessions/test" + path)
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"source": "reference"}
    assert io_case.fs.reads[f"{io_case.session.uri}/messages.jsonl"] == 0
    assert io_case.fs.reads[f"{io_case.session.uri}/.meta.json"] == 0


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
async def test_unicode_jsonl_survives_public_append(io_case, separator):
    case = io_case
    message = Message(id="unicode", role="user", parts=[TextPart(text="a" + separator + "b")])
    original = message.to_jsonl() + "\n"
    uri = f"{case.session.uri}/messages.jsonl"
    case.fs.files[uri] = original
    await case.session.add_messages_async([{"role": "assistant", "parts": [TextPart(text="next")]}])
    assert case.fs.files[uri].startswith(original)
    values = [json.loads(line) for line in case.fs.files[uri].split("\n") if line]
    assert [value["parts"][0]["text"] for value in values] == ["a" + separator + "b", "next"]


@pytest.mark.parametrize("target", ["messages.jsonl", ".meta.json"])
async def test_failed_append_reload_observes_durable_messages_and_policy(io_case, target):
    case = io_case
    await case.session.load()
    meta_uri = f"{case.session.uri}/.meta.json"
    latest = json.loads(case.fs.files[meta_uri])
    latest["memory_policy"] = {"self": {"enabled": False}, "peer": {"enabled": False}}
    case.fs.files[meta_uri] = json.dumps(latest)
    case.fs.fail_write = f"{case.session.uri}/{target}"
    with pytest.raises(OSError, match="injected"):
        await case.session.add_messages_async(
            [{"role": "assistant", "parts": [TextPart(text="efgh")]}]
        )
    await case.session.load()
    assert [message.content for message in case.session.messages] == (
        ["abcd"] if target == "messages.jsonl" else ["abcd", "efgh"]
    )
    assert case.session.meta.memory_policy == latest["memory_policy"]
    # Checkpoint the uncertain outcome without replaying the original message.
    await case.session.add_messages_async([])
    stored = json.loads(case.fs.files[meta_uri])
    assert stored["message_count"] == len(case.session.messages)
    assert stored["total_message_count"] == len(case.session.messages)
    assert stored["memory_policy"] == latest["memory_policy"]


async def test_append_after_phase1_repair_does_not_restore_archived_messages(io_case):
    case = io_case
    session = case.session
    messages = [Message(id=f"m{i}", role="user", parts=[TextPart(text=str(i))]) for i in range(3)]
    case.fs.files[f"{session.uri}/messages.jsonl"] = "".join(m.to_jsonl() + "\n" for m in messages)
    session._messages = messages
    session.meta.message_count = 3
    session.meta.total_message_count = 3
    await session._save_meta()
    case.fs.fail_write = f"{session.uri}/messages.jsonl"
    with pytest.raises(OSError, match="injected"):
        await session.commit_async(keep_recent_count=1)
    writer = Session(viking_fs=case.fs, session_id="test", ctx=session.ctx)
    await writer.add_messages_async([{"role": "user", "parts": [TextPart(text="after repair")]}])
    live = [
        json.loads(line)
        for line in case.fs.files[f"{session.uri}/messages.jsonl"].split("\n")
        if line
    ]
    assert [row["parts"][0]["text"] for row in live] == ["2", "after repair"]
    archive = case.fs.files[f"{session.uri}/history/archive_001/messages.jsonl"]
    assert [json.loads(line)["id"] for line in archive.split("\n") if line] == ["m0", "m1"]


async def test_load_without_storage_keeps_in_memory_messages():
    session = Session(viking_fs=None, session_id="memory-only")
    await session.add_messages_async([{"role": "user", "parts": [TextPart(text="abcd")]}])
    await session.load()
    assert [message.content for message in session.messages] == ["abcd"]
    assert session.meta.pending_tokens == 1


async def test_delete_and_exists_do_not_require_valid_message_contents(io_case):
    case = io_case
    case.fs.files[f"{case.session.uri}/messages.jsonl"] = "broken JSONL"
    assert await case.local.session_exists("test") is True
    await case.sessions.delete("test", case.session.ctx)
    assert not case.fs.files


async def _seed_four_messages(case):
    messages = [
        Message(
            id=f"m{i}",
            role="user" if i % 2 == 0 else "assistant",
            parts=[TextPart(text="x" * 400)],
        )
        for i in range(4)
    ]
    case.fs.files[f"{case.session.uri}/messages.jsonl"] = "".join(
        message.to_jsonl() + "\n" for message in messages
    )
    case.session._messages = messages
    case.session.meta.message_count = 4
    case.session.meta.total_message_count = 4
    case.session._rebuild_pending_tokens()
    await case.session._save_meta()


async def test_reader_snapshot_crosses_commit_without_double_counting(io_case, monkeypatch):
    from openviking.session import commit_recovery

    case = io_case
    await _seed_four_messages(case)
    monkeypatch.setattr(commit_recovery, "publish", AsyncMock())
    writer = Session(viking_fs=case.fs, session_id="test", ctx=case.session.ctx)
    original_read = case.fs.read_file
    crossed = False

    async def read_across_commit(uri, *, ctx):
        nonlocal crossed
        captured = await original_read(uri, ctx=ctx)
        if not crossed and uri == f"{case.session.uri}/messages.jsonl":
            crossed = True
            await writer.commit_async(keep_recent_count=1)
        return captured

    monkeypatch.setattr(case.fs, "read_file", read_across_commit)
    reader = await case.sessions.get("test", case.session.ctx)
    assert reader.meta.total_message_count == 4
    assert json.loads(case.fs.files[f"{case.session.uri}/.meta.json"])["total_message_count"] == 4


@pytest.mark.parametrize("auto_commit_first", [False, True])
async def test_commit_rechecks_receipt_before_archiving(io_case, monkeypatch, auto_commit_first):
    from langchain_core.messages import AIMessage, HumanMessage

    from openviking.integrations.langchain.client import OpenVikingConnection
    from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy
    from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
    from openviking.integrations.langchain.ov_context.commit import CommitConfig
    from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
    from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
    from openviking.message.part import part_from_dict
    from openviking.service import session_service
    from openviking.session import commit_recovery
    from openviking.session.tool_views import add_messages_with_tool_views_async

    case = io_case
    await _seed_four_messages(case)
    case.session.meta.auto_commit_policy_configured = True
    case.session.meta.auto_commit_policy = {
        "pending_token_threshold": 0,
        "message_count_threshold": 3,
        "idle_timeout_seconds": 0,
        "keep_recent_count": 2,
        "min_commit_interval_seconds": 0,
    }
    await case.session._save_meta()
    service = SessionService(viking_fs=case.fs)
    monkeypatch.setattr(commit_recovery, "publish", AsyncMock())
    monkeypatch.setattr(
        session_service,
        "get_task_tracker",
        lambda: SimpleNamespace(has_running=AsyncMock(return_value=False)),
    )

    class SDK:
        _initialized = True
        commit_calls = 0

        async def get_session(self, session_id, *, identity=None, auto_create=False):
            session = await service.get(session_id, case.session.ctx, auto_create=auto_create)
            return session.meta.to_dict()

        async def batch_add_messages(self, session_id, messages, *, identity=None):
            session = await service.open(session_id, case.session.ctx)
            specs = [{**m, "parts": [part_from_dict(p) for p in m["parts"]]} for m in messages]
            return await add_messages_with_tool_views_async(session, specs)

        async def commit_session(self, session_id, *, identity=None, keep_recent_count=0):
            self.commit_calls += 1
            return await service.commit_async(session_id, case.session.ctx, keep_recent_count)

    sdk = SDK()
    policy = OpenVikingCommitPolicy(CommitConfig(1000, 0.5, 100, 0), sdk)
    engine = SessionEngine(
        OpenVikingConnection(client=sdk),
        OVContextConfig(context_window=100000),
        None,
        None,
        message_writer=OVMessageWriter(sdk),
        session_commit_policy=policy,
    )
    raw = [
        (HumanMessage if i % 2 == 0 else AIMessage)(id=f"h{i}", content="x" * 400) for i in range(6)
    ]
    context = OVContext(
        session_id="test",
        owner=("a", "u", "p", "test"),
        messages=raw[:4],
        write_index=4,
        run_id="r",
        commit_count=0,
        context_status="ready",
    )
    engine.active[context.session_key] = "r"
    await engine.write(context, raw, identity=context.identity)
    if auto_commit_first:
        assert await service.maybe_schedule_auto_commit(
            "test", case.session.ctx, reason_hint="message_write"
        )
        await service.close_background_tasks()
    context.end_run()
    context.completed_runs = 1
    submitted = await engine.commit(
        context,
        pending_tokens=await engine.pending_tokens(context, identity=context.identity),
        identity=context.identity,
    )
    latest = await service.get("test", case.session.ctx)
    assert submitted is (not auto_commit_first)
    assert sdk.commit_calls == (0 if auto_commit_first else 1)
    assert latest.meta.commit_count == 1
    assert len(latest.messages) == (2 if auto_commit_first else 0)
