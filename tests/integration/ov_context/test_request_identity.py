"""Invocation identity through real hooks, native caches and SDK request headers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy
from openviking.integrations.langchain.ov_context import (
    OVContext,
    OVContextConfig,
    OVIdentity,
    OVSessionClient,
    RecallResult,
    RunIdentity,
    WriteResult,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking_cli.client.http import AsyncHTTPClient

pytestmark = pytest.mark.asyncio


class FakeOVService:
    """Offline canonical HTTP routes behind the native SDK and a shared pool."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.sessions: set[tuple[str, str, str, str]] = set()
        self.messages: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        self.http = httpx.AsyncClient(
            base_url="http://ov.test", transport=httpx.MockTransport(self.handle)
        )
        self.client = AsyncHTTPClient(
            auth_mode="trusted", url="http://ov.test", profile_enabled=False
        )
        self.client._http = self.http

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        await asyncio.sleep(0)
        account = request.headers["X-OpenViking-Account"]
        user = request.headers["X-OpenViking-User"]
        peer = request.headers["X-OpenViking-Actor-Peer"]
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        sid = (
            path.split("/")[4]
            if path.startswith("/api/v1/sessions/")
            else body.get("session_id", "")
        )
        scope = account, user, peer, sid
        result: Any
        if path == "/api/v1/sessions":
            self.sessions.add(scope)
            result = {"session_id": sid}
        elif path.endswith("/messages/batch"):
            self.messages.setdefault(scope, []).extend(body["messages"])
            result = {"session_id": sid, "added": len(body["messages"])}
        elif path.endswith("/context"):
            result = {
                "messages": self.messages.get(scope, []),
                "latest_finished_archive_id": None,
                "latest_archive_overview": "",
            }
        elif path in ("/api/v1/search/search", "/api/v1/search/find"):
            result = {
                "memories": [
                    {
                        "uri": "viking://user/memories/reference",
                        "context_type": "memory",
                        "score": 0.9,
                    }
                ]
            }
        elif path == "/api/v1/content/read":
            result = f"reference:{account}/{user}/{peer}"
        elif path.endswith("/commit"):
            result = {"archived": True, "task_id": f"task:{account}:{user}:{peer}"}
        elif path == "/api/v1/tasks":
            result = []
        elif path.startswith("/api/v1/tasks/"):
            assert path.rsplit("/", 1)[-1] == f"task:{account}:{user}:{peer}"
            result = {"status": "running"}
        elif "/tool-results/" in path and path.endswith("/search"):
            result = {"tool_result_id": path.split("/")[-2], "matches": []}
        elif "/tool-results" in path:
            result = {}
        elif path.startswith("/api/v1/sessions/"):
            if scope not in self.sessions:
                return httpx.Response(
                    404,
                    json={
                        "status": "error",
                        "error": {"code": "NOT_FOUND", "message": "session missing"},
                    },
                )
            result = {"session_id": sid, "latest_finished_archive_id": None, "pending_tokens": 100}
        else:
            raise AssertionError(f"Unexpected route: {request.method} {path}")
        return httpx.Response(200, json={"status": "ok", "result": result})


class Writer:
    def __init__(self, sdk: FakeOVService) -> None:
        self.client = sdk.client
        self.identities: list[OVIdentity] = []

    async def write(
        self,
        session_id: str,
        raw_messages: list[BaseMessage],
        write_index: int,
        run_id: str,
        *,
        identity: OVIdentity | None = None,
        peer_id: str | None = None,
    ) -> WriteResult:
        assert identity is not None
        self.identities.append(identity)
        delta = raw_messages[write_index:]
        # Test I/O stores the full native payload as text in canonical OV parts.
        messages = [
            {
                "role": "assistant" if message["type"] == "ai" else "user",
                "parts": [{"type": "text", "text": json.dumps(message)}],
                "peer_id": peer_id,
            }
            for message in messages_to_dict(delta)
        ]
        await self.client.batch_add_messages(session_id, messages, identity=identity)
        return WriteResult(list(delta), len(raw_messages), len(delta))


class Reader:
    def __init__(self, sdk: FakeOVService) -> None:
        self.client = sdk.client
        self.identities: list[OVIdentity] = []

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        self.identities.append(identity)
        result = await self.client.get_session_context(session_id, identity=identity)
        messages = [json.loads(message["parts"][0]["text"]) for message in result["messages"]]
        return ContextSnapshot(messages_from_dict(messages), None)

    async def recall(self, query: str, *, identity: OVIdentity, session_id: str) -> RecallResult:
        self.identities.append(identity)
        result = await self.client.search(query, session_id=session_id, identity=identity)
        content = await self.client.read(result.memories[0].uri, identity=identity)
        return RecallResult(content, "success", 1)


def runtime(identity: RunIdentity) -> SimpleNamespace:
    return SimpleNamespace(context=SimpleNamespace(identity=identity))


def build(sdk: FakeOVService, **kwargs: Any) -> tuple[OpenVikingContextMiddleware, Writer, Reader]:
    kwargs.setdefault("commit_config", None)  # Partition tests submit directly, without thresholds.
    writer, reader = Writer(sdk), Reader(sdk)
    kwargs.setdefault("commit_config", None)  # Identity tests use explicit legacy commits.
    middleware = OpenVikingContextMiddleware(
        client=sdk.client,
        message_writer=writer,
        context_reader=reader,
        context_config=OVContextConfig(context_window=100000),
        token_counter=lambda request: 10,
        **kwargs,
    )
    return middleware, writer, reader


async def test_shared_sdk_same_session_id_different_accounts_users_and_agents(
    env: tuple[Any, ...],
) -> None:
    sdk = FakeOVService()
    mw, writer, reader = build(sdk)
    identities = [
        RunIdentity("a", "u", "mail", "same-session", "r"),
        RunIdentity("b", "u", "mail", "same-session", "r"),
        RunIdentity("a", "v", "doc", "same-session", "r"),
    ]

    async def invoke(identity: RunIdentity) -> dict[str, Any]:
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(id="old", content="history"),
                HumanMessage(id="same-id", content="question"),
            ],
            "account_id": "poison",
            "user_id": "poison",
            "peer_id": "poison",
        }
        state.update(await mw.abefore_agent(state, runtime(identity)))

        async def handler(request: Any) -> ModelResponse[Any]:
            assert request.messages[-1].content.startswith(
                f"[参考记忆]\nreference:{identity.account_id}/{identity.user_id}/{identity.actor_peer_id}"
            )
            return ModelResponse(result=[AIMessage(content="done")])

        result = await mw.awrap_model_call(
            env[4].override(state=state, runtime=runtime(identity)), handler
        )
        state.update(result.command.update)
        context = OVContext.from_state(state["ov_context"])
        assert await mw._ov_session_engine.commit(context, identity=context.identity)
        # Pending task identity and task metadata must remain in the same partition.
        assert not await mw._ov_session_engine._compression_failed(
            context, identity=context.identity
        )
        assert context.pending_commit is not None
        assert (
            context.pending_commit["task_id"]
            == f"task:{identity.account_id}:{identity.user_id}:{identity.actor_peer_id}"
        )
        assert context.owner == identity.session_key
        return state

    try:
        states = await asyncio.gather(*(invoke(identity) for identity in identities))
        assert set(mw._ov_session_engine.active) == {
            identity.session_key for identity in identities
        }
        assert set(mw._ov_session_engine._commit_progress) == {
            identity.session_key for identity in identities
        }
        assert not hasattr(mw._ov_session_engine, "locks")
        assert set(writer.identities) == {identity.sdk_identity for identity in identities}
        assert set(reader.identities) == set(writer.identities)
        for request in sdk.requests:
            assert "identity" not in (json.loads(request.content) if request.content else {})
            assert request.headers["X-OpenViking-Account"] != "poison"
            if request.url.path.endswith("/messages/batch"):
                assert (
                    json.loads(request.content)["messages"][0]["peer_id"]
                    == request.headers["X-OpenViking-Actor-Peer"]
                )
        assert "X-OpenViking-Account" not in sdk.http.headers
        for state, identity in zip(states, identities, strict=True):
            # A different Run for the same partition must wait for the first one.
            with pytest.raises(RuntimeError, match="Serialize"):
                await mw.abefore_agent(state, runtime(replace(identity, run_id="next")))
    finally:
        await mw.aclose()
        assert not sdk.http.is_closed
        await sdk.http.aclose()


@pytest.mark.parametrize(
    "field", ["account_id", "user_id", "actor_peer_id", "session_id", "run_id"]
)
async def test_identity_change_rejected_by_every_followup_hook_before_io(
    field: str, env: tuple[Any, ...]
) -> None:
    sdk = FakeOVService()
    mw, _, _ = build(sdk)
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    state: dict[str, Any] = {"messages": [HumanMessage(id="u", content="question")]}
    try:
        state.update(await mw.abefore_agent(state, runtime(identity)))
        changed = runtime(replace(identity, **{field: "different"}))
        before = len(sdk.requests)
        for hook in (mw.abefore_model, mw.aafter_model, mw.aafter_agent):
            with pytest.raises(ValueError, match="different"):
                await hook(state, changed)
        with pytest.raises(ValueError, match="different"):
            await mw.awrap_model_call(
                env[4].override(state=state, runtime=changed), lambda request: None
            )
        assert len(sdk.requests) == before
    finally:
        await sdk.http.aclose()


@pytest.mark.parametrize("bad", [None, {}, "text"])
async def test_missing_identity_no_network_or_default(bad: Any) -> None:
    sdk = FakeOVService()
    mw, _, _ = build(sdk)
    try:
        with pytest.raises(ValueError, match="complete RunIdentity"):
            await mw.abefore_agent(
                {"messages": [HumanMessage(content="question")]}, SimpleNamespace(context=bad)
            )
        assert not sdk.requests
    finally:
        await sdk.http.aclose()


async def test_identity_resolver_and_explicit_message_peer() -> None:
    sdk = FakeOVService()
    mw, writer, _ = build(
        sdk,
        identity_resolver=lambda state, runtime: RunIdentity(**runtime.context),
        peer_id_resolver=lambda state, runtime: "authorized-peer",
    )
    identity = RunIdentity("a", "u", "caller", "same-session", "r")
    try:
        state: dict[str, Any] = {"messages": [HumanMessage(id="u", content="question")]}
        state.update(await mw.abefore_agent(state, SimpleNamespace(context=vars(identity))))
        assert state["ov_context"]["owner"] == identity.session_key
        assert writer.identities == [identity.sdk_identity]
        create = next(request for request in sdk.requests if request.url.path == "/api/v1/sessions")
        assert create.headers["X-OpenViking-Actor-Peer"] == "caller"
        batch = next(
            request for request in sdk.requests if request.url.path.endswith("/messages/batch")
        )
        assert batch.headers["X-OpenViking-Actor-Peer"] == "caller"
        assert json.loads(batch.content)["messages"][0]["peer_id"] == "authorized-peer"
    finally:
        await sdk.http.aclose()


@pytest.mark.parametrize("field", ["account_id", "user_id", "actor_peer_id"])
async def test_native_sdk_overrides_constructor_identity_per_request(field: str) -> None:
    requests: list[httpx.Request] = []
    client = AsyncHTTPClient(
        auth_mode="trusted",
        url="http://ov.test",
        account="a",
        user="u",
        actor_peer_id="p",
        profile_enabled=False,
    )

    def response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": {"session_id": "s"}})

    client._http = httpx.AsyncClient(
        base_url="http://ov.test",
        headers=OVIdentity("a", "u", "p").headers(),
        transport=httpx.MockTransport(response),
    )
    await client.initialize()
    handle = AsyncClientHandle(OpenVikingConnection(client=client))
    identity = replace(OVIdentity("a", "u", "p"), **{field: "different"})
    try:
        await handle.call("get_session", identity=identity, session_id="s")
        assert all(requests[0].headers[key] == value for key, value in identity.headers().items())
        assert all(
            client._http.headers[key] == value
            for key, value in OVIdentity("a", "u", "p").headers().items()
        )
    finally:
        await client.close()


async def test_native_sdk_request_identity_overrides_extra_header_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": {"session_id": "s"}})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(response), **kwargs),
    )
    client = AsyncHTTPClient(
        auth_mode="trusted",
        url="http://ov.test",
        account="a",
        user="u",
        actor_peer_id="p",
        extra_headers={"x-openviking-account": "other"},
        profile_enabled=False,
    )
    await client.initialize()
    handle = AsyncClientHandle(OpenVikingConnection(client=client))
    try:
        await handle.call("get_session", identity=OVIdentity("a", "u", "p"), session_id="s")
        assert requests[0].headers["X-OpenViking-Account"] == "a"
        assert client._http is not None
        assert "other" in client._http.headers["X-OpenViking-Account"]
    finally:
        await client.close()


async def test_policy_receives_and_forwards_identity_and_cleanup_recovers_owner() -> None:
    sdk = FakeOVService()
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), sdk.client)
    mw, _, _ = build(sdk, session_commit_policy=policy)
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(id="old", content="history"),
            HumanMessage(id="new", content="question"),
        ]
    }

    class Graph:
        async def aget_state(self, config: Any) -> SimpleNamespace:
            return SimpleNamespace(values=state)

        async def aupdate_state(self, config: Any, update: dict[str, Any]) -> None:
            state.update(update)

    try:
        state.update(await mw.abefore_agent(state, runtime(identity)))
        context = OVContext.from_state(state["ov_context"])
        assert await mw._ov_session_engine.commit(
            context, pending_tokens=100, identity=context.identity
        )
        await mw.acleanup_run(Graph(), {})
        assert state["ov_context"]["run_id"] is None
        assert state["ov_context"]["pending_commit"]["task_id"] == "task:a:u:p"
        assert tuple(state["ov_context"]["owner"]) == identity.session_key
        assert all(request.headers["X-OpenViking-Actor-Peer"] == "p" for request in sdk.requests)
    finally:
        await sdk.http.aclose()


async def test_tool_read_search_list_use_runtime_identity_and_reject_wrong_owner() -> None:
    sdk = FakeOVService()
    mw, _, _ = build(sdk)
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    state: dict[str, Any] = {"messages": [HumanMessage(id="u", content="question")]}
    try:
        state.update(await mw.abefore_agent(state, runtime(identity)))
        rt = SimpleNamespace(state=state, context=runtime(identity).context)
        read, search, listing = mw.tools
        ref = "viking://user/u/sessions/same-session/tool-results/result"
        await read.coroutine(ref=ref, runtime=rt)
        await search.coroutine(ref=ref, queries=["needle", "alternative"], runtime=rt)
        await listing.coroutine(runtime=rt)
        before = len(sdk.requests)
        rt.context = runtime(replace(identity, user_id="wrong")).context
        with pytest.raises(ValueError, match="different"):
            await read.coroutine(ref=ref, runtime=rt)
        assert len(sdk.requests) == before
        assert {request.url.path for request in sdk.requests}.issuperset(
            {
                "/api/v1/sessions/same-session/tool-results/result",
                "/api/v1/sessions/same-session/tool-results/result/search",
                "/api/v1/sessions/same-session/tool-results",
            }
        )
        searches = [
            request for request in sdk.requests if request.url.path.endswith("/result/search")
        ]
        assert {request.url.params["q"] for request in searches} == {"needle", "alternative"}
        assert all(request.headers["X-OpenViking-User"] == "u" for request in searches)
    finally:
        await sdk.http.aclose()


@pytest.mark.parametrize("partition", ["account", "user"])
async def test_two_deepagents_share_sdk_with_independent_checkpoints(
    env: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, partition: str
) -> None:
    from deepagents import graph as deepagents_graph
    from deepagents.profiles.harness import harness_profiles

    from examples.deepagents_ov_context.chat.agent import (
        RequestContext,
        create_demo_agent,
    )

    from .test_graph import ToolLoopModel, read_mail

    monkeypatch.setattr(
        harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES)
    )
    models: list[ToolLoopModel] = []

    def resolve_model(value: Any) -> ToolLoopModel:
        class UniqueMessageModel(ToolLoopModel):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                result = super()._generate(*args, **kwargs)
                result.generations[0].message.id = f"model-{len(self.inputs)}"
                return result

        model = UniqueMessageModel()
        models.append(model)
        return model

    monkeypatch.setattr(deepagents_graph, "resolve_model", resolve_model)
    sdk = FakeOVService()
    client: OVSessionClient = sdk.client
    writer, reader = Writer(sdk), Reader(sdk)
    agents = [
        create_demo_agent(
            checkpointer=InMemorySaver(),
            model=f"ov-demo:identity-{index}",
            connection=OpenVikingConnection(client=client),
            context_config=OVContextConfig(context_window=100000),
            message_writer=writer,
            context_reader=reader,
            tools=[read_mail],
        )
        for index in range(2)
    ]
    identities = [
        RunIdentity("a", "u", "mail", "same-session", "r1"),
        RunIdentity(
            "b" if partition == "account" else "a",
            "v" if partition == "user" else "u",
            "doc",
            "same-session",
            "r1",
        ),
    ]

    async def invoke(index: int) -> None:
        agent, mw = agents[index]
        identity = identities[index]
        config: RunnableConfig = {"configurable": {"thread_id": json.dumps(identity.session_key)}}
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(id="u", content="Read the mail")]},
                config=config,
                context=RequestContext(identity),
            )
            assert result["messages"][-1].content == "Mail summarized"
            state = (await agent.aget_state(config)).values["ov_context"]
            assert tuple(state["owner"]) == identity.session_key
            assert state["run_id"] is None
            assert not mw._ov_session_engine.active
            await agent.ainvoke(
                {"messages": [HumanMessage(id="u2", content="Second query")]},
                config=config,
                context=RequestContext(replace(identity, run_id="r2")),
            )
            restored = (await agent.aget_state(config)).values["ov_context"]
            assert restored["completed_runs"] == 2
            assert restored["write_index"] == 6
            assert "reference:" not in str(restored["messages"])
            assert [m.type for m in restored["messages"]][:4] == ["human", "ai", "tool", "ai"]
        except BaseException:
            await mw.acleanup_run(agent, config)
            raise

    try:
        await asyncio.gather(invoke(0), invoke(1))
        assert len(models) == 2
        for model, identity in zip(models, identities, strict=True):
            assert any(
                f"reference:{identity.account_id}/{identity.user_id}/{identity.actor_peer_id}"
                in str(messages)
                for messages in model.inputs
            )
            other = identities[1] if identity == identities[0] else identities[0]
            assert all(
                f"reference:{other.account_id}/{other.user_id}/{other.actor_peer_id}"
                not in str(messages)
                for messages in model.inputs
            )
    finally:
        for _, mw in agents:
            await mw.aclose()
        assert not sdk.http.is_closed
        await sdk.http.aclose()


@pytest.mark.parametrize(
    "snapshot_reader", [False, True], ids=["legacy-refresh", "snapshot-exchange"]
)
async def test_two_users_share_two_agent_instances_and_sdk_across_four_sessions(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_reader: bool,
) -> None:
    from deepagents import graph as deepagents_graph
    from deepagents.profiles.harness import harness_profiles

    from examples.deepagents_ov_context.chat.agent import (
        RequestContext,
        create_demo_agent,
    )
    from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
    from openviking.integrations.langchain.ov_context.reader import (
        OVContextReader as SnapshotReader,
    )

    from .test_graph import ToolLoopModel, read_mail

    class SessionModel(ToolLoopModel):
        def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
            result: ChatResult = super()._generate(*args, **kwargs)
            current = next(m for m in reversed(args[0]) if isinstance(m, HumanMessage))
            question = str(current.content).split("[用户原文]\n")[-1]
            response = result.generations[0].message
            response.id = f"{question}-{'call' if getattr(response, 'tool_calls', []) else 'final'}"
            return result

    models: list[SessionModel] = []

    def resolve_model(value: Any) -> SessionModel:
        model = SessionModel()
        models.append(model)
        return model

    monkeypatch.setattr(
        harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES)
    )
    monkeypatch.setattr(deepagents_graph, "resolve_model", resolve_model)
    sdk = FakeOVService()
    close_client = AsyncMock(wraps=sdk.client.close)
    monkeypatch.setattr(sdk.client, "close", close_client)
    writer = Writer(sdk)
    reader = (
        SnapshotReader(sdk.client, MemoryReaderConfig(search_mode="find"))
        if snapshot_reader
        else Reader(sdk)
    )
    agents = [
        create_demo_agent(
            checkpointer=InMemorySaver(),
            model=f"ov-demo:shared-{actor}",
            connection=OpenVikingConnection(client=sdk.client),
            context_config=OVContextConfig(context_window=100000),
            message_writer=writer,
            context_reader=reader,
            tools=[read_mail],
            token_counter=lambda request: 10,
        )
        for actor in ("mail", "doc")
    ]
    identities = [
        (index, RunIdentity("acct-a", user, actor, f"s-{user}-{actor}", "r1"))
        for index, actor in enumerate(("mail", "doc"))
        for user in ("user-a", "user-b")
    ]
    sdk_headers = dict(sdk.http.headers)

    async def invoke(index: int, identity: RunIdentity, round_number: int) -> None:
        agent, middleware = agents[index]
        config: RunnableConfig = {"configurable": {"thread_id": json.dumps(identity.session_key)}}
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            id=f"u{round_number}",
                            content=f"{identity.session_id}-round-{round_number}",
                        )
                    ]
                },
                config=config,
                context=RequestContext(replace(identity, run_id=f"r{round_number}")),
            )
            assert result["messages"][-1].content == "Mail summarized"
        except BaseException:
            await middleware.acleanup_run(agent, config)
            raise

    try:
        # Exactly two graph/Middleware instances serve both users concurrently.
        assert agents[0][0] is not agents[1][0]
        engines = [middleware._ov_session_engine for _, middleware in agents]
        assert engines[0].client is not engines[1].client
        for engine in engines:
            assert engine.writer is writer and engine.context_reader is reader
            assert engine.client.client is sdk.client
        for round_number in (1, 2):
            await asyncio.gather(
                *(invoke(index, identity, round_number) for index, identity in identities)
            )
        assert len(models) == 2
        assert set(sdk.messages) == {identity.session_key for _, identity in identities}
        assert len(sdk.sessions) == 4
        snapshots = []
        for index, identity in identities:
            agent, _ = agents[index]
            config: RunnableConfig = {
                "configurable": {"thread_id": json.dumps(identity.session_key)}
            }
            snapshot = (await agent.aget_state(config)).values["ov_context"]
            snapshots.append(snapshot)
            assert tuple(snapshot["owner"]) == identity.session_key
            assert snapshot["session_id"] == identity.session_id
            assert snapshot["write_index"] == 6 and snapshot["completed_runs"] == 2
            assert snapshot["run_id"] is None and not snapshot["recall_done"]
            assert [m.type for m in snapshot["messages"]] == [
                "human",
                "ai",
                "tool",
                "ai",
                "human",
                "ai",
            ]
            assert [m.content for m in snapshot["messages"] if isinstance(m, HumanMessage)] == [
                f"{identity.session_id}-round-1",
                f"{identity.session_id}-round-2",
            ]
            stored = messages_from_dict(
                [
                    json.loads(record["parts"][0]["text"])
                    for record in sdk.messages[identity.session_key]
                ]
            )
            assert [m.model_dump() for m in stored] == [
                m.model_dump() for m in snapshot["messages"]
            ]
            assert "[参考记忆]" not in str(snapshot["messages"])
        assert len({id(snapshot["messages"]) for snapshot in snapshots}) == 4
        for index, model in enumerate(models):
            assert len(model.inputs) == 6  # Two tool loops plus two second-round calls.
            for messages in model.inputs:
                current = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
                matching = [
                    identity
                    for agent_index, identity in identities
                    if agent_index == index and identity.session_id in str(current.content)
                ]
                assert len(matching) == 1
                identity = matching[0]
                reference = (
                    f"reference:{identity.account_id}/{identity.user_id}/{identity.actor_peer_id}"
                )
                assert str(current.content).count(reference) == 1
                for _, other in identities:
                    if other != identity:
                        assert other.session_id not in str(messages)
                        assert (
                            f"reference:{other.account_id}/{other.user_id}/{other.actor_peer_id}"
                            not in str(messages)
                        )
        expected_scopes = {identity.session_key for _, identity in identities}
        searches = [
            request
            for request in sdk.requests
            if request.url.path
            == ("/api/v1/search/find" if snapshot_reader else "/api/v1/search/search")
        ]
        assert len(searches) == 8  # One recall per Session per Run, despite tool loops.
        assert sum(request.url.path.endswith("/context") for request in sdk.requests) == 4
        for request in sdk.requests:
            body = json.loads(request.content) if request.content else {}
            account = request.headers["X-OpenViking-Account"]
            user = request.headers["X-OpenViking-User"]
            actor = request.headers["X-OpenViking-Actor-Peer"]
            assert (account, user, actor) in {scope[:3] for scope in expected_scopes}
            if request.url.path.startswith("/api/v1/sessions/"):
                assert (account, user, actor, request.url.path.split("/")[4]) in expected_scopes
            if request in searches and snapshot_reader:
                assert "session_id" not in body
                assert any(
                    scope[:3] == (account, user, actor) and scope[3] in body["query"]
                    for scope in expected_scopes
                )
            elif request in searches or request.url.path == "/api/v1/sessions":
                assert (account, user, actor, body["session_id"]) in expected_scopes
        assert dict(sdk.http.headers) == sdk_headers
        assert all(not engine.active for engine in engines)
        for _, middleware in agents:
            await middleware.aclose()
        close_client.assert_not_awaited()
        assert not sdk.http.is_closed
    finally:
        for _, middleware in agents:
            await middleware.aclose()
        await sdk.client.close()
        close_client.assert_awaited_once()


@pytest.mark.parametrize(
    "field", ["account_id", "user_id", "actor_peer_id", "session_id", "run_id"]
)
@pytest.mark.parametrize("bad", ["", " ", "bad\nheader", "bad/path"])
async def test_resolver_rejects_invalid_identity_fields_before_io(field: str, bad: str) -> None:
    sdk = FakeOVService()
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    mw, _, _ = build(
        sdk, identity_resolver=lambda state, runtime: replace(identity, **{field: bad})
    )
    try:
        with pytest.raises(ValueError, match="Identity fields"):
            await mw.abefore_agent(
                {"messages": [HumanMessage(content="question")]}, runtime(identity)
            )
        assert not sdk.requests
    finally:
        await sdk.http.aclose()


async def test_snapshot_reader_receives_session_and_identity_without_mutable_context(
    env: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeOVService()
    mw, _, reader = build(sdk)
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    original = reader.exchange_context
    calls: list[tuple[str, str | None, OVIdentity]] = []

    async def exchange_context(
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        calls.append((session_id, expected_archive_id, identity))
        return await original(session_id, expected_archive_id, identity=identity)

    monkeypatch.setattr(reader, "exchange_context", exchange_context)

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage(content="done")])

    try:
        state: dict[str, Any] = {"messages": [HumanMessage(id="u", content="question")]}
        state.update(await mw.abefore_agent(state, runtime(identity)))
        await mw.awrap_model_call(
            env[4].override(state=state, runtime=runtime(identity)),
            handler,
        )
        assert calls == [(identity.session_id, None, identity.sdk_identity)]
        assert state["ov_context"]["owner"] == identity.session_key
    finally:
        await sdk.http.aclose()


async def test_completed_after_agent_retry_is_noop_with_matching_identity() -> None:
    sdk = FakeOVService()
    mw, _, _ = build(sdk)
    identity = RunIdentity("a", "u", "p", "same-session", "r")
    state: dict[str, Any] = {"messages": [HumanMessage(id="u", content="question")]}
    try:
        state.update(await mw.abefore_agent(state, runtime(identity)))
        state.update(await mw.aafter_agent(state, runtime(identity)))
        before = len(sdk.requests)
        assert await mw.aafter_agent(state, runtime(identity)) == {}
        assert state["ov_context"]["completed_runs"] == 1
        assert len(sdk.requests) == before
        with pytest.raises(ValueError, match="different session/identity"):
            await mw.aafter_agent(state, runtime(replace(identity, user_id="wrong")))
    finally:
        await sdk.http.aclose()
