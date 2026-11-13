"""Hook contract, recovery and isolation checks with offline external I/O."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.middleware import (
    OpenVikingCommitPolicy,
)
from openviking.integrations.langchain.ov_context import (
    ContextBudgetExceeded,
    LatencySamples,
    Metrics,
    OVContext,
    OVContextConfig,
    RecallResult,
    WriteResult,
)
from openviking.integrations.langchain.ov_context.assembler import INJECTED
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot, MemoryReaderConfig
from openviking_cli.exceptions import AlreadyExistsError, ConflictError, NotFoundError
from tests.ov_context_support.commit_settings import EXAMPLE_COMMIT_CONFIG
from tests.ov_context_support.fixtures import counter
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter
from tests.ov_context_support.transport import OVHTTPError

from .test_io_contracts import SuppliedWriter
from .test_session_hooks import call, runtime, start

pytestmark = pytest.mark.asyncio


class Reader:
    """Publish prevalidated snapshots independently for each Session."""

    def __init__(self) -> None:
        self.snapshots: dict[str, tuple[str | None, list[BaseMessage]]] = {}
        self.exchanges: list[tuple[str, str | None]] = []
        self.queries: list[tuple[str, str]] = []
        self.tokens: int | None = None
        self.error: Exception | None = None

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity=None,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        self.exchanges.append((session_id, expected_archive_id))
        if self.error:
            raise self.error
        archive, messages = self.snapshots.get(session_id, (None, []))
        return ContextSnapshot(list(messages), archive)

    async def recall(self, query: str, *, identity=None, session_id: str) -> RecallResult:
        self.queries.append((session_id, query))
        await asyncio.sleep(0)
        return RecallResult(f"Reference for {session_id}", "success", 1)


def make(
    env: tuple[Any, ...],
    reader: Reader,
    writer: SuppliedWriter | None = None,
    *,
    retry_attempts: int = 0,
    **kwargs: Any,
) -> OpenVikingContextMiddleware:
    kwargs.setdefault("commit_config", None)  # New-policy tests supply a policy/config explicitly.
    env[0].session_meta_hook = lambda sid: {
        "latest_finished_archive_id": reader.snapshots.get(sid, (None, []))[0],
        "pending_tokens": reader.tokens,
    }
    return OpenVikingContextMiddleware(
        client=TransportSDKAdapter(env[1]),
        account="a",
        user="u",
        actor_peer_id="p",
        peer_id="p",
        context_reader=reader,
        message_writer=writer or SuppliedWriter(),
        context_config=OVContextConfig(context_window=100000, retry_attempts=retry_attempts),
        token_counter=counter,
        **kwargs,
    )


async def test_configured_policy_commits_after_five_completed_runs(env: tuple[Any, ...]) -> None:
    reader, client = Reader(), CommitClient()
    reader.tokens = 1  # Trigger by completed Runs, not the capacity threshold.
    middleware = make(env, reader, commit_config=EXAMPLE_COMMIT_CONFIG)
    middleware._connection.client.commit_session = client.commit_session
    state: dict[str, Any] = {"messages": []}
    for index in range(5):
        rt = runtime(env, f"r{index}")
        state["messages"].append(HumanMessage(id=f"u{index}", content="Question"))
        state.update(await middleware.abefore_agent(state, rt))
        state["messages"].append(AIMessage(id=f"a{index}", content="Answer"))
        state.update(await middleware.aafter_model(state, rt))
        assert client.calls == []
        state.update(await middleware.aafter_agent(state, rt))
        assert len(client.calls) == int(index == 4)
    assert client.calls == [("s", 0)]
    assert state["ov_context"]["completed_runs"] == 0
    assert state["ov_context"]["run_id"] is None


@pytest.mark.parametrize("run_id", [None, "", " ", "r/1", "r\n1", 12])
async def test_invalid_run_id_fails_before_io(env: tuple[Any, ...], run_id: Any) -> None:
    reader, writer = Reader(), SuppliedWriter()
    mw = make(env, reader, writer)
    mw.identity_resolver = lambda state, runtime: replace(env[3], run_id=run_id)
    state = {"messages": env[4].messages}
    with pytest.raises(ValueError):
        await mw.abefore_agent(state, runtime(env))
    assert not env[0].requests and not writer.calls and not reader.exchanges
    assert "ov_context" not in state and not mw._ov_session_engine.active


async def test_missing_identity_requires_explicit_resolver(env: tuple[Any, ...]) -> None:
    mw = make(env, Reader())
    with pytest.raises(ValueError, match="complete RunIdentity"):
        await mw.abefore_agent({"messages": env[4].messages}, SimpleNamespace(context=None))
    assert not env[0].requests


async def test_injected_reference_is_not_the_current_user(env: tuple[Any, ...]) -> None:
    class FilteringWriter(SuppliedWriter):
        async def write(
            self,
            session_id: str,
            raw_messages: list[BaseMessage],
            write_index: int,
            run_id: str,
            *,
            identity=None,
            peer_id: str | None = None,
        ) -> WriteResult:
            messages = [
                m for m in raw_messages[write_index:] if not m.additional_kwargs.get(INJECTED)
            ]
            return WriteResult(messages, len(raw_messages), len(messages))

    reader = Reader()
    mw = make(env, reader, FilteringWriter())
    user = HumanMessage(id="u", content="Read mail")
    reference = HumanMessage(
        id="reference", content="Not a query", additional_kwargs={INJECTED: True}
    )
    state = {"messages": [user, reference]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    assert reader.queries == [("s", "Read mail")]
    assert state["ov_context"]["original_user"] is user
    assert user.content == "Read mail"
    assert [m.id for m in state["ov_context"]["messages"]] == ["u"]


async def test_checkpoint_resume_keeps_published_recall_and_skips_initialization(
    env: tuple[Any, ...],
) -> None:
    reader, writer = Reader(), SuppliedWriter()
    mw = make(env, reader, writer)
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    serde = JsonPlusSerializer(allowed_msgpack_modules=[])
    restored = serde.loads_typed(serde.dumps_typed(state))
    fresh = make(env, reader, writer)
    restored.update(await fresh.abefore_agent(restored, runtime(env)))
    await call(fresh, restored, env[4], runtime(env))
    assert reader.queries == [("s", "Read the mail")]
    assert reader.exchanges.count(("s", None)) == 1
    assert len(writer.calls) == 1
    assert restored["ov_context"]["messages"][0].content.count("[参考记忆]") == 1


async def test_fallback_forces_full_reload_without_archive_change(env: tuple[Any, ...]) -> None:
    reader = Reader()
    reader.error = OVHTTPError(503)
    mw = make(env, reader)
    engine = mw._ov_session_engine
    engine.config = replace(engine.config, context_failure="host")
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    assert state["ov_context"]["context_status"] == "reload_required"
    reader.error = None
    past = HumanMessage(id="past", content="Persisted remote history")
    reader.snapshots["s"] = (None, [past, *env[4].messages])
    output = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in output] == ["past", "m1"]
    assert reader.exchanges == [("s", None), ("s", None)]
    assert state["ov_context"]["archive_id"] is None
    assert not state["ov_context"]["context_status"] == "reload_required"
    await call(mw, state, env[4], runtime(env))
    assert reader.exchanges == [("s", None), ("s", None)]


@pytest.mark.parametrize("status", [401, 403])
async def test_recall_auth_failure_never_invokes_model(env: tuple[Any, ...], status: int) -> None:
    class Denied(Reader):
        async def recall(self, query: str, *, identity=None, session_id: str) -> RecallResult:
            raise OVHTTPError(status)

    mw = make(env, Denied())
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    before = deepcopy(state)
    with pytest.raises(OVHTTPError):
        await call(mw, state, env[4], runtime(env))
    assert state == before


class SessionClient:
    _initialized = True

    def __init__(
        self, gets: list[dict[str, str] | Exception], create: dict[str, str] | Exception
    ) -> None:
        self.gets, self.created = gets, create
        self.calls: list[str] = []

    async def get_session(
        self, session_id: str, auto_create: bool = False, *, identity=None
    ) -> dict[str, str]:
        self.calls.append("get")
        value = self.gets.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def list_tasks(
        self, *, task_type: str, resource_id: str, status: str, limit: int, identity=None
    ) -> list[dict[str, Any]]:
        return []

    async def create_session(
        self, session_id: str, memory_policy: dict[str, Any] | None = None, *, identity=None
    ) -> dict[str, str]:
        self.calls.append("create")
        if isinstance(self.created, Exception):
            raise self.created
        return self.created


@pytest.mark.parametrize(
    "mode", ["exists", "created", "race", "wrong_get", "wrong_create", "conflict", "unavailable"]
)
async def test_session_creation_and_competition(env: tuple[Any, ...], mode: str) -> None:
    reader, writer = Reader(), SuppliedWriter()
    mw = make(env, reader, writer)
    client = SessionClient([NotFoundError("s")], {"session_id": "s"})
    if mode == "exists":
        client.gets = [{"session_id": "s"}]
    elif mode == "race":
        client.created = AlreadyExistsError("s")
        client.gets.append({"session_id": "s"})
    elif mode == "wrong_get":
        client.gets = [{"session_id": "other"}]
    elif mode == "wrong_create":
        client.created = {"session_id": "other"}
    elif mode == "conflict":
        client.created = ConflictError("not an already-exists error")
    elif mode == "unavailable":
        client.created = OVHTTPError(503)
    mw._ov_session_engine.client.client = client
    state = {"messages": env[4].messages}
    if mode in ("exists", "created", "race"):
        state.update(await mw.abefore_agent(state, runtime(env)))
        assert state["ov_context"]["context_status"] == "ready"
        assert (
            client.calls
            == {"exists": ["get"], "created": ["get", "create"], "race": ["get", "create", "get"]}[
                mode
            ]
        )
    else:
        with pytest.raises((ValueError, ConflictError, OVHTTPError)):
            await mw.abefore_agent(state, runtime(env))
        assert not writer.calls and not reader.exchanges
        assert not mw._ov_session_engine.active and "ov_context" not in state


async def test_shared_reader_handles_parallel_sessions_and_rejects_same_session_run(
    env: tuple[Any, ...],
) -> None:
    reader = Reader()
    mw = make(env, reader)

    async def invoke(sid: str, rid: str) -> dict[str, Any]:
        rt = SimpleNamespace(
            context=SimpleNamespace(identity=replace(env[3], session_id=sid, run_id=rid))
        )
        state = {"messages": [HumanMessage(id="same-id", content="Same question")]}
        state.update(await mw.abefore_agent(state, rt))
        await call(mw, state, env[4], rt)
        return state

    a, b = await asyncio.gather(invoke("a", "r-a"), invoke("b", "r-b"))
    assert set(reader.queries) == {("a", "Same question"), ("b", "Same question")}
    assert "Reference for a" in a["ov_context"]["messages"][0].content
    assert "Reference for b" in b["ov_context"]["messages"][0].content
    with pytest.raises(RuntimeError, match="Serialize runs"):
        await invoke("a", "new-run")
    assert mw._ov_session_engine.active == {
        ("a", "u", "p", "a"): "r-a",
        ("a", "u", "p", "b"): "r-b",
    }


async def test_other_agent_checkpoint_is_rejected_before_io(env: tuple[Any, ...]) -> None:
    a = make(env, Reader())
    state = {"messages": env[4].messages}
    state.update(await a.abefore_agent(state, runtime(env)))
    b = make(env, Reader())
    b.identity_resolver = lambda state, runtime: replace(env[3], actor_peer_id="other-agent")
    before = len(env[0].requests)
    with pytest.raises(ValueError, match="different session/identity"):
        await b.abefore_agent(state, runtime(env))
    assert len(env[0].requests) == before


@pytest.mark.parametrize("extended", [False, True])
async def test_response_merge_preserves_model_and_command_fields(
    env: tuple[Any, ...], extended: bool
) -> None:
    mw, state = await start(env)
    normal = ModelResponse(result=[AIMessage(content="ok")], structured_response={"answer": 42})
    command = Command(
        graph="parent", goto="next", resume={"request": 1}, update=MappingProxyType({"counter": 2})
    )

    async def handler(request: ModelRequest[Any]) -> Any:
        return ExtendedModelResponse(normal, command) if extended else normal

    response = await mw.awrap_model_call(
        env[4].override(state=state, runtime=runtime(env)), handler
    )
    assert response.model_response is normal
    assert normal.structured_response == {"answer": 42}
    assert "ov_context" in response.command.update
    if extended:
        assert response.command.graph == "parent" and response.command.goto == "next"
        assert response.command.resume == {"request": 1}
        assert response.command.update["counter"] == 2
        assert command.update == {"counter": 2}


@pytest.mark.parametrize("update", [{"ov_context": {}}, [("key", 1)]])
async def test_conflicting_or_non_mapping_command_update_fails(
    env: tuple[Any, ...], update: Any
) -> None:
    mw, state = await start(env)
    before = deepcopy(state)

    async def handler(request: ModelRequest[Any]) -> ExtendedModelResponse[Any]:
        return ExtendedModelResponse(ModelResponse(result=[]), Command(update=update))

    with pytest.raises((ValueError, TypeError)):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), handler)
    assert state == before


async def test_handler_failure_keeps_graph_state_unchanged_without_engine_lock(
    env: tuple[Any, ...],
) -> None:
    reader = Reader()
    mw = make(env, reader)
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    before = deepcopy(state)

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        assert not hasattr(mw._ov_session_engine, "locks")
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await asyncio.wait_for(
            mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), handler), 1
        )
    assert state == before
    await call(mw, state, env[4], runtime(env))
    assert len(reader.queries) == 2  # First attempt did not publish its recall state.


async def test_model_request_parameters_survive_ov_only_assembly(env: tuple[Any, ...]) -> None:
    mw, state = await start(env, reader_config=MemoryReaderConfig(enabled=False))
    request = env[4].override(
        state=state,
        runtime=runtime(env),
        messages=[HumanMessage(id="poison", content="ignored")],
        model_settings={"temperature": 0.2, "max_tokens": 512},
        response_format={"type": "json_object"},
    )

    async def handler(updated: ModelRequest[Any]) -> ModelResponse[Any]:
        assert updated.model is request.model
        assert updated.tools == request.tools
        assert updated.model_settings == request.model_settings
        assert updated.response_format == request.response_format
        assert [m.id for m in updated.messages] == ["m1"]
        return ModelResponse(result=[])

    await mw.awrap_model_call(request, handler)


@pytest.mark.parametrize("failure", [False, True])
async def test_observation_samples_have_identity_status_and_no_content(
    env: tuple[Any, ...], failure: bool
) -> None:
    samples: list[tuple[str, dict[str, Any]]] = []
    metrics = Metrics(lambda event, **fields: samples.append((event, fields)))
    mw = make(env, Reader(), metrics=metrics)
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        if failure:
            raise RuntimeError("failure")
        return ModelResponse(result=[])

    request = env[4].override(state=state, runtime=runtime(env))
    if failure:
        with pytest.raises(RuntimeError):
            await mw.awrap_model_call(request, handler)
    else:
        await mw.awrap_model_call(request, handler)
    by_event = dict(samples)
    for event in (
        "before_agent.total",
        "session.ensure",
        "context.initial_load",
        "message.write",
        "context.refresh",
        "memory.recall",
        "context.assemble",
        "model.handler",
    ):
        assert by_event[event]["duration_ms"] >= 0
        assert by_event[event]["session_id"] == "s"
        assert by_event[event]["actor_peer_id"] == "p"
        assert by_event[event]["run_id"] == "r"
    assert by_event["model.handler"]["status"] == ("error" if failure else "ok")
    assert "Read the mail" not in str(samples)
    latency = LatencySamples()
    for event, fields in samples:
        latency(event, **fields)
    assert all(row["avg_ms"] <= row["max_ms"] for row in latency.summary())
    assert all(row["p90_ms"] >= 0 for row in latency.summary())


class CommitClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def commit_session(
        self, session_id: str, *, keep_recent_count: int, identity=None
    ) -> dict[str, Any]:
        self.calls.append((session_id, keep_recent_count))
        return {"archived": True, "task_id": "compress-task"}


@pytest.mark.parametrize("pending_tools", [False, True])
async def test_after_model_commits_only_after_a_final_response(
    env: tuple[Any, ...], pending_tools: bool
) -> None:
    reader = Reader()
    reader.tokens = 100
    client = CommitClient()
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 2), client)
    mw = make(env, reader, commit_policy=policy)
    state = {
        "messages": [
            HumanMessage(id="old", content="Earlier question"),
            HumanMessage(id="current", content="Current question"),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime(env)))
    state["messages"].append(
        AIMessage(
            id="response",
            content="" if pending_tools else "Final answer",
            tool_calls=([{"id": "call-1", "name": "lookup", "args": {}}] if pending_tools else []),
        )
    )

    state.update(await mw.aafter_model(state, runtime(env)))

    assert state["ov_context"]["write_index"] == 3
    assert client.calls == ([] if pending_tools else [("s", 2)])
    assert await mw.aafter_model(state, runtime(env)) == {}
    assert client.calls == ([] if pending_tools else [("s", 2)])


@pytest.mark.parametrize(
    "tokens,keep,expected",
    [(100, 1, True), (None, 1, False), (0, 1, False), (100, 0, True), (100, 2, False)],
)
async def test_single_policy_requires_known_tokens_safe_tail_and_history(
    env: tuple[Any, ...], tokens: int | None, keep: int, expected: bool
) -> None:
    client = CommitClient()
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, keep), client)
    reader = Reader()
    reader.tokens = tokens
    mw = make(
        env,
        reader,
        session_commit_policy=policy,
        commit_policy=OpenVikingCommitPolicy(mode="always"),
    )
    raw = [
        HumanMessage(id="old", content="history"),
        HumanMessage(id="current", content="question"),
    ]
    state = {"messages": raw}
    state.update(await mw.abefore_agent(state, runtime(env)))
    # Any accidental fallthrough to the legacy commit path would hit this endpoint.
    await call(mw, state, env[4], runtime(env))
    assert bool(client.calls) is expected
    assert not env[0].counts["commit"]
    if expected:
        assert state["ov_context"]["pending_commit"] == {
            "task_id": "compress-task",
            "archive_id": None,
        }
        await call(mw, state, env[4], runtime(env))
        assert len(client.calls) == 1
    state.update(await mw.aafter_agent(state, runtime(env)))
    assert not env[0].counts["commit"]


async def test_policy_expands_actual_retention_to_keep_tool_group_complete(
    env: tuple[Any, ...],
) -> None:
    env[0].sessions.add("s")
    client = CommitClient()
    mw = make(
        env,
        Reader(),
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 2), client),
    )
    context = OVContext(
        "s",
        messages=[
            HumanMessage(id="old", content="old"),
            AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}]),
            ToolMessage(id="t", tool_call_id="c", content="done"),
            HumanMessage(id="u", content="current"),
        ],
        owner=("a", "u", "p", "s"),
    )
    context.begin_run("r", context.messages[-1])
    assert await mw._ov_session_engine.commit(
        context, pending_tokens=100, identity=context.identity
    )
    assert client.calls == [("s", 3)]
    assert mw._ov_session_engine.session_commit_policy.config.keep_recent_count == 2


async def test_policy_budget_error_never_uses_legacy_commit(env: tuple[Any, ...]) -> None:
    client = CommitClient()
    mw = make(
        env,
        Reader(),
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 0), client),
    )
    state = {"messages": [HumanMessage(id="u", content="current")]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    mw._ov_session_engine.assembler.token_counter = lambda request: 1_000_000
    with pytest.raises(ContextBudgetExceeded):
        await call(mw, state, env[4], runtime(env))
    assert not client.calls and not env[0].counts["commit"]


async def test_policy_compression_waits_for_published_snapshot_and_reassembles_once(
    env: tuple[Any, ...],
) -> None:
    user = HumanMessage(id="current", content="question")

    class PublishingClient(CommitClient):
        async def commit_session(
            self, session_id: str, *, keep_recent_count: int, identity=None
        ) -> dict[str, Any]:
            result = await super().commit_session(
                session_id, keep_recent_count=keep_recent_count, identity=identity
            )
            reader.snapshots[session_id] = (
                "archive-1",
                [HumanMessage(id="ov:summary", content="summary"), user],
            )
            return result

    reader = Reader()
    reader.tokens = 100
    client = PublishingClient()
    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env,
        reader,
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), client),
        metrics=Metrics(lambda event, **fields: samples.append((event, fields))),
    )
    state = {"messages": [HumanMessage(id="past", content="past"), user]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    engine = mw._ov_session_engine
    engine.assembler.config = replace(
        engine.config, context_window=100, output_reserve=0, safety_margin=0
    )
    engine.assembler.token_counter = lambda request: (
        20
        if request.messages and request.messages[0].id == "ov:summary"
        else (200 if request.messages else 0)
    )
    output = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in output] == ["ov:summary", "current"]
    assert len(client.calls) == 1 and not env[0].counts["commit"]
    assert state["ov_context"]["archive_id"] == "archive-1"
    assert state["ov_context"]["pending_commit"] is None
    assert output[1].content.count("[参考记忆]") == 1
    assert len([event for event, _ in samples if event == "context.assemble"]) == 2
    assert dict(samples)["context.compress_wait"]["status"] == "ok"


async def test_official_sdk_clients_keep_fixed_identity_across_parallel_sessions(
    env: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking_cli.client.http import AsyncHTTPClient

    requests: list[httpx.Request] = []
    sessions: set[str] = set()

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/tasks":
            return httpx.Response(200, json={"status": "ok", "result": []})
        if request.method == "GET":
            sid = request.url.path.rsplit("/", 1)[-1]
            if sid not in sessions:
                return httpx.Response(
                    404,
                    json={"status": "error", "error": {"code": "NOT_FOUND", "message": "missing"}},
                )
        else:
            sid = json.loads(request.content)["session_id"]
            sessions.add(sid)
        return httpx.Response(200, json={"status": "ok", "result": {"session_id": sid}})

    real_http_client = httpx.AsyncClient

    def http_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_http_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", http_client)
    clients = []
    for peer in ("mail-agent", "doc-agent"):
        client = AsyncHTTPClient(
            auth_mode="trusted", url="http://ov.test", account="a", user="u", actor_peer_id=peer
        )
        await client.initialize()
        clients.append(client)

    async def run(client: AsyncHTTPClient, peer: str, sid: str) -> None:
        mw = OpenVikingContextMiddleware(
            client=client,
            account="a",
            user="u",
            actor_peer_id=peer,
            identity_resolver=lambda state, runtime: replace(
                env[3], actor_peer_id=peer, session_id=sid, run_id=f"{sid}-run"
            ),
            context_reader=Reader(),
            message_writer=SuppliedWriter(),
        )
        state = {"messages": [HumanMessage(id="u", content="question")]}
        await mw.abefore_agent(state, runtime(env))
        await mw.aclose()
        assert client._http is not None  # Middleware does not close a borrowed Client.

    try:
        await asyncio.gather(
            *(
                run(client, peer, f"{peer}-{index}")
                for client, peer in zip(clients, ("mail-agent", "doc-agent"), strict=True)
                for index in range(2)
            )
        )
        assert len(sessions) == 4
        for request in requests:
            assert request.headers["X-OpenViking-Account"] == "a"
            assert request.headers["X-OpenViking-User"] == "u"
            peer = request.headers["X-OpenViking-Actor-Peer"]
            if request.url.path == "/api/v1/tasks":
                assert request.url.params["resource_id"].startswith(peer)
                continue
            sid = (
                request.url.path.rsplit("/", 1)[-1]
                if request.method == "GET"
                else json.loads(request.content)["session_id"]
            )
            assert sid.startswith(peer)
    finally:
        for client in clients:
            await client.close()


async def test_cancel_before_agent_releases_ownership_and_emits_cancelled_sample(
    env: tuple[Any, ...],
) -> None:
    entered = asyncio.Event()

    class BlockingWriter(SuppliedWriter):
        async def write(
            self,
            session_id: str,
            raw_messages: list[BaseMessage],
            write_index: int,
            run_id: str,
            *,
            identity=None,
            peer_id: str | None = None,
        ) -> WriteResult:
            entered.set()
            await asyncio.Event().wait()
            return WriteResult([], write_index, 0)

    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env,
        Reader(),
        BlockingWriter(),
        metrics=Metrics(lambda event, **fields: samples.append((event, fields))),
    )
    state = {"messages": env[4].messages}
    task = asyncio.create_task(mw.abefore_agent(state, runtime(env)))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not mw._ov_session_engine.active and "ov_context" not in state
    assert dict(samples)["before_agent.total"]["status"] == "cancelled"
    assert dict(samples)["message.write"]["status"] == "cancelled"


async def test_wrap_rejects_different_run_before_reader_or_model(env: tuple[Any, ...]) -> None:
    reader = Reader()
    mw = make(env, reader)
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    with pytest.raises(ValueError, match="different Run"):
        await call(mw, state, env[4], runtime(env, "other-run"))
    assert reader.exchanges == [("s", None)] and not reader.queries
