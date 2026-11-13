"""Discard every Run's graph/SDK; retain only checkpoint and OV service state."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from deepagents import graph as deepagents_graph
from deepagents.profiles.harness import harness_profiles
from langchain_core.messages import HumanMessage, messages_from_dict
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from examples.deepagents_ov_context.chat.agent import invoke_demo_run
from openviking.integrations.langchain.ov_context import OVContextConfig, RunIdentity
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import (
    ContextSnapshot,
    MemoryReaderConfig,
    OVContextReader,
)
from openviking_cli.exceptions import UnavailableError

from .test_graph import ToolLoopModel, read_mail
from .test_request_identity import FakeOVService, Writer

pytestmark = pytest.mark.asyncio


class Service(FakeOVService):
    """Canonical metadata/task routes and a controllable archive publication."""

    def __init__(self) -> None:
        super().__init__()
        self.commits: dict[tuple[str, ...], int] = {}
        self.policy_triggers: dict[tuple[str, ...], bool] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.terminal: dict[tuple[str, ...], str] = {}
        self.archived_ids: dict[tuple[str, ...], list[str]] = {}

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        scope = tuple(
            request.headers[name]
            for name in ("X-OpenViking-Account", "X-OpenViking-User", "X-OpenViking-Actor-Peer")
        )
        sid = (
            request.url.params.get("resource_id", "")
            if path == "/api/v1/tasks"
            else path.split("/")[4]
            if path.startswith("/api/v1/sessions/")
            else ""
        )
        key = (*scope, sid)
        if path == "/api/v1/tasks":
            self.requests.append(request)
            result: Any = [
                task
                for task in self.tasks.values()
                if task["scope"] == scope
                and task["resource_id"] == sid
                and task["status"] == request.url.params["status"]
            ]
        elif path.startswith("/api/v1/tasks/"):
            self.requests.append(request)
            result = self.tasks[path.rsplit("/", 1)[-1]]
            assert result["scope"] == scope
        elif path.endswith("/commit"):
            self.requests.append(request)
            count = self.commits.get(key, 0) + 1
            self.commits[key] = count
            keep = json.loads(request.content)["keep_recent_count"]
            records = self.messages.get(key, [])
            old, live = (records[:-keep], records[-keep:]) if keep else (records, [])
            self.archived_ids[key] = [json.loads(m["parts"][0]["text"])["data"]["id"] for m in old]
            self.messages[key] = live
            task_id = f"{sid}-task-{count}"
            self.tasks[task_id] = {
                "task_id": task_id,
                "task_type": "session_commit",
                "resource_id": sid,
                "scope": scope,
                "status": "running",
                "result": {"archive_id": f"archive_{count:03d}"},
            }
            result = {
                "archived": True,
                "task_id": task_id,
                "archive_uri": f"viking://session/{sid}/history/archive_{count:03d}",
            }
        else:
            response = await super().handle(request)
            if not response.is_success:
                return response
            result = response.json()["result"]
            if path.endswith("/context"):
                native_ids = [
                    json.loads(m["parts"][0]["text"])["data"]["id"] for m in result["messages"]
                ]
                assert len(native_ids) == len(set(native_ids)), native_ids
                result["latest_finished_archive_id"] = self.terminal.get(key)
                result["latest_archive_overview"] = (
                    "Archived history" if key in self.terminal else ""
                )
            elif path.startswith("/api/v1/sessions/") and isinstance(result, dict):
                result.update(
                    commit_count=self.commits.get(key, 0),
                    latest_finished_archive_id=self.terminal.get(key),
                    pending_tokens=100
                    if self.policy_triggers.get(key) and not self.commits.get(key)
                    else 0,
                )
        return httpx.Response(200, json={"status": "ok", "result": result})


class NativeReader(OVContextReader):
    """External adapter paired with Writer's complete native JSON fixture format."""

    async def exchange_context(
        self,
        session_id,
        expected_archive_id=None,
        *,
        identity=None,
        token_budget: int | None = None,
    ):
        result = await self.client.get_session_context(session_id, identity=identity)
        archive = result["latest_finished_archive_id"]
        if expected_archive_id is not None and expected_archive_id != archive:
            raise ValueError("Archive is not published")
        messages = messages_from_dict(
            [json.loads(record["parts"][0]["text"]) for record in result["messages"]]
        )
        if archive is not None:
            messages.insert(
                0,
                HumanMessage(id=f"ov_summary:{archive}", content=result["latest_archive_overview"]),
            )
        return ContextSnapshot(messages, archive)


class RoundModel(ToolLoopModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        user = next(m for m in reversed(args[0]) if isinstance(m, HumanMessage))
        query = str(user.content).split("[用户原文]\n")[-1]
        response = result.generations[0].message
        response.id = query + ("-call" if getattr(response, "tool_calls", []) else "-final")
        return result


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> list[RoundModel]:
    from examples.deepagents_ov_context.chat import agent as create_deep_agent_demo

    # This matrix exercises the legacy capacity settings. The default
    # capacity/Run policy is covered by the main Hook and live Chat tests.
    monkeypatch.setattr(
        create_deep_agent_demo,
        "OpenVikingContextMiddleware",
        partial(create_deep_agent_demo.OpenVikingContextMiddleware, commit_config=None),
    )
    values: list[RoundModel] = []
    monkeypatch.setattr(
        harness_profiles, "_HARNESS_PROFILES", dict(harness_profiles._HARNESS_PROFILES)
    )

    def resolve(value: Any) -> RoundModel:
        model = RoundModel()
        values.append(model)
        return model

    monkeypatch.setattr(deepagents_graph, "resolve_model", resolve)
    return values


async def invoke(
    service: Service,
    saver: InMemorySaver,
    identity: RunIdentity,
    query: str | None,
    *,
    tokens: int = 10,
) -> dict[str, Any]:
    service.policy_triggers[identity.session_key] = tokens >= 90000
    return await invoke_demo_run(
        model="ov-demo:run-scoped",
        ov_url="http://ov.test",
        api_key="test-only",
        auth_mode="trusted",
        http_client=service.http,
        checkpointer=saver,
        identity=identity,
        messages=[] if query is None else [HumanMessage(id=query, content=query)],
        message_writer_factory=lambda sdk: Writer(SimpleNamespace(client=sdk)),
        context_reader_factory=lambda sdk: NativeReader(
            sdk, MemoryReaderConfig(search_mode="find")
        ),
        tools=[read_mail],
        context_config=OVContextConfig(context_window=100000),
        commit_config=CommitConfig(100, 0.5, 999, 1),
        token_counter=lambda request: min(tokens, 1000) if request.messages else 0,
    )


async def snapshot(saver: InMemorySaver, identity: RunIdentity) -> dict[str, Any]:
    saved = await saver.aget_tuple(
        {"configurable": {"thread_id": json.dumps(identity.session_key)}}
    )
    assert saved is not None
    return saved.checkpoint["channel_values"]["ov_context"]


async def test_rebuild_every_run_preserves_four_sessions_native_cache_and_one_pool(models):
    service, saver = Service(), InMemorySaver()
    identities = [
        RunIdentity("acct", user, peer, f"s-{user}-{peer}", "r1")
        for user in ("u1", "u2")
        for peer in ("mail", "doc")
    ]
    defaults = dict(service.http.headers)
    try:
        for number in (1, 2):
            await asyncio.gather(
                *(
                    invoke(
                        service, saver, replace(i, run_id=f"r{number}"), f"{i.session_id}-q{number}"
                    )
                    for i in identities
                )
            )
            assert not service.http.is_closed
        assert len(models) == 8  # Every Run constructs a new graph and model.
        for identity in identities:
            state = await snapshot(saver, identity)
            assert tuple(state["owner"]) == identity.session_key
            assert state["context_status"] == "ready" and state["completed_runs"] == 2
            assert state["run_id"] is None and state["write_index"] == 6
            assert [m.type for m in state["messages"]] == [
                "human",
                "ai",
                "tool",
                "ai",
                "human",
                "ai",
            ]
            assert "[参考记忆]" not in str(state["messages"])
            assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
                f"{identity.session_id}-q1",
                f"{identity.session_id}-q2",
            ]
            for other in identities:
                if other != identity:
                    assert other.session_id not in str(state["messages"])
        assert sum(r.url.path.endswith("/context") for r in service.requests) == 4
        assert sum(r.url.path == "/api/v1/search/find" for r in service.requests) == 8
        assert dict(service.http.headers) == defaults
    finally:
        await service.http.aclose()


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("terminal_task_status", ["completed", "running"])
async def test_new_instance_observes_remote_commit_without_task_discovery(
    models, monkeypatch, cancelled, terminal_task_status
):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")

        entered = asyncio.Event()

        class Failing(RoundModel):
            def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
                raise RuntimeError("model failed after commit")

            async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
                if cancelled:
                    entered.set()
                    await asyncio.Event().wait()
                return self._generate(*args, **kwargs)

        monkeypatch.setattr(deepagents_graph, "resolve_model", lambda value: Failing())
        current = replace(identity, run_id="r2")
        attempt = asyncio.create_task(
            invoke(service, saver, current, "current-question", tokens=90000)
        )
        if cancelled:
            await asyncio.wait_for(entered.wait(), timeout=5)
            attempt.cancel()
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await attempt
        saved = await snapshot(saver, current)
        assert saved["pending_commit"] is None  # Model node never published its update.
        assert saved["commit_count"] == 0
        assert service.commits[current.session_key] == 1
        # No cleanup is called and the failing graph/Middleware/SDK are discarded.
        monkeypatch.setattr(deepagents_graph, "resolve_model", lambda value: RoundModel())
        await invoke(service, saver, current, None, tokens=90000)
        saved = await snapshot(saver, current)
        assert saved["pending_commit"] is None  # Do not reconstruct a task ID.
        assert saved["commit_count"] == 1 and saved["completed_runs"] == 1
        assert service.commits[current.session_key] == 1
        service.tasks["s-task-1"]["status"] = terminal_task_status
        service.terminal[current.session_key] = "archive_001"
        await invoke(service, saver, replace(identity, run_id="r3"), "next-question")
        saved = await snapshot(saver, identity)
        assert saved["archive_id"] == "archive_001" and saved["pending_commit"] is None
        ids = [m.id for m in saved["messages"]]
        assert set(ids).isdisjoint(service.archived_ids[current.session_key])
        assert ids.count("current-question") == 1 and ids.count("next-question") == 1
        assert "[参考记忆]" not in str(saved["messages"])
        assert not service.http.is_closed
        await invoke(service, saver, replace(identity, run_id="r4"), "fourth-question")
        saved = await snapshot(saver, identity)
        assert saved["completed_runs"] == 3 and saved["pending_commit"] is None
        assert service.commits[identity.session_key] == 1
        assert not any(request.url.path == "/api/v1/tasks" for request in service.requests)
    finally:
        await service.http.aclose()


async def test_phase_one_archive_without_registered_task_blocks_another_commit(models):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")
        # Simulate a lost response after persisted Phase 1, before task registration.
        service.commits[identity.session_key] = 1
        await invoke(service, saver, replace(identity, run_id="r2"), "next-question", tokens=90000)
        saved = await snapshot(saver, identity)
        assert saved["pending_commit"] is None
        assert saved["commit_count"] == 1 and saved["archive_id"] is None
        assert service.commits[identity.session_key] == 1
        assert not any(r.url.path.endswith("/commit") for r in service.requests)
    finally:
        await service.http.aclose()


@pytest.mark.parametrize("failure", ["task-list", "metadata"])
async def test_task_list_is_unused_and_metadata_failure_stops_before_model(models, failure):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")
        saved = await snapshot(saver, identity)
        requests = []

        async def unavailable(request):
            requests.append(request)
            path = request.url.path
            if path == ("/api/v1/tasks" if failure == "task-list" else "/api/v1/sessions/s"):
                return httpx.Response(
                    503,
                    json={
                        "status": "error",
                        "error": {"code": "UNAVAILABLE", "message": "recovery unavailable"},
                    },
                )
            return await service.handle(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as pool:
            previous = service.http
            service.http = pool
            try:
                if failure == "metadata":
                    with pytest.raises(UnavailableError, match="(?i)recovery unavailable"):
                        await invoke(
                            service, saver, replace(identity, run_id="r2"), "next-question"
                        )
                else:
                    await invoke(service, saver, replace(identity, run_id="r2"), "next-question")
            finally:
                service.http = previous
        assert not any(r.url.path == "/api/v1/tasks" for r in requests)
        after = await snapshot(saver, identity)
        if failure == "metadata":
            assert not models[-1].inputs
            # The user write is confirmed before the existing model-time probe.
            assert after["write_index"] == saved["write_index"] + 1
            assert after["messages"][-1].id == "next-question"
        else:
            assert models[-1].inputs
    finally:
        await service.http.aclose()


async def test_new_run_refreshes_checkpoint_after_server_idle_archive(models):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    key = identity.session_key
    try:
        await invoke(service, saver, identity, "old-question")
        saved = await snapshot(saver, identity)
        # Server-side idle archiving runs independently of the discarded graph.
        service.archived_ids[key] = [message.id for message in saved["messages"]]
        service.messages[key] = []
        service.commits[key] = 1
        service.terminal[key] = "archive_001"
        service.tasks["s-task-1"] = {
            "task_id": "s-task-1",
            "task_type": "session_commit",
            "resource_id": "s",
            "scope": key[:3],
            "status": "completed",
            "result": {"archive_id": "archive_001"},
        }
        assert (await snapshot(saver, identity)) == saved
        requests_before = len(service.requests)
        await invoke(service, saver, replace(identity, run_id="r2"), "next-question")
        current = await snapshot(saver, identity)
        assert current["archive_id"] == "archive_001"
        assert current["commit_count"] == 1 and current["pending_commit"] is None
        assert current["completed_runs"] == 1  # Count only the Run after the observed idle commit.
        assert tuple(current["owner"]) == key
        ids = [message.id for message in current["messages"]]
        assert ids[0] == "ov_summary:archive_001"
        assert ids.count("next-question") == 1
        assert set(ids).isdisjoint(service.archived_ids[key])
        assert not any(
            request.url.path.endswith("/commit") for request in service.requests[requests_before:]
        )
        assert len(models) == 2 and not service.http.is_closed
    finally:
        await service.http.aclose()


@pytest.mark.parametrize("count", [True, -1])
async def test_invalid_remote_commit_count_rejects_model_after_user_write(models, count):
    from openviking.integrations.langchain.ov_context.errors import ContextReadError

    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")
        saved = await snapshot(saver, identity)
        service.commits[identity.session_key] = count
        with pytest.raises(ContextReadError, match="Invalid session commit_count"):
            await invoke(service, saver, replace(identity, run_id="r2"), "next-question")
        after = await snapshot(saver, identity)
        assert after["write_index"] == saved["write_index"] + 1
        assert after["messages"][-1].id == "next-question"
        assert after["commit_count"] == saved["commit_count"]
        assert not models[-1].inputs
    finally:
        await service.http.aclose()


async def test_task_records_do_not_gate_normal_dialogue(models):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")
        for index in range(2):
            task_id = f"t{index}"
            service.tasks[task_id] = {
                "task_id": task_id,
                "task_type": "session_commit",
                "resource_id": "s",
                "scope": identity.session_key[:3],
                "status": "running",
            }
        await invoke(service, saver, replace(identity, run_id="r2"), "next-question")
        assert models[-1].inputs
        assert not any(request.url.path == "/api/v1/tasks" for request in service.requests)
    finally:
        await service.http.aclose()


async def test_auto_commit_before_existing_model_probe_uses_fresh_count(models):
    service, saver = Service(), InMemorySaver()
    identity = RunIdentity("acct", "user", "mail", "s", "r1")
    try:
        await invoke(service, saver, identity, "old-question")
        discovered = False

        async def auto_commit(request):
            nonlocal discovered
            if request.url.path.endswith("/messages/batch") and not discovered:
                discovered = True
                service.commits[identity.session_key] = 1
                service.tasks["s-task-1"] = {
                    "task_id": "s-task-1",
                    "task_type": "session_commit",
                    "resource_id": "s",
                    "scope": identity.session_key[:3],
                    "status": "running",
                }
            return await service.handle(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(auto_commit)) as pool:
            previous = service.http
            service.http = pool
            try:
                await invoke(
                    service, saver, replace(identity, run_id="r2"), "next-question", tokens=90000
                )
            finally:
                service.http = previous
        saved = await snapshot(saver, identity)
        assert saved["commit_count"] == 1 and saved["completed_runs"] == 1
        assert discovered and saved["pending_commit"] is None
        assert not any(request.url.path == "/api/v1/tasks" for request in service.requests)
        assert service.commits[identity.session_key] == 1
    finally:
        await service.http.aclose()
