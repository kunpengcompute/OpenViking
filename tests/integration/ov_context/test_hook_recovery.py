"""Compression side effects, failed-node recovery and per-call preparation."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy
from openviking.integrations.langchain.ov_context import (
    ContextBudgetExceeded,
    Metrics,
    OVContext,
    WriteResult,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from tests.ov_context_support.transport import OVHTTPError

from .test_graph import CallContext, ToolLoopModel
from .test_hook_design import CommitClient, Reader, make
from .test_io_contracts import SuppliedWriter
from .test_session_hooks import call, runtime

pytestmark = pytest.mark.asyncio


def raw_messages() -> list[BaseMessage]:
    return [
        HumanMessage(id="old", content="history"),
        HumanMessage(id="current", content="question"),
    ]


class Tasks(CommitClient):
    def __init__(self) -> None:
        super().__init__()
        self.statuses: dict[str, str] = {}

    async def commit_session(
        self, session_id: str, *, keep_recent_count: int, identity=None
    ) -> dict[str, Any]:
        await super().commit_session(session_id, keep_recent_count=keep_recent_count)
        task_id = f"task-{len(self.calls)}"
        self.statuses[task_id] = "running"
        return {"archived": True, "task_id": task_id}

    async def get_task(self, task_id: str, *, identity=None) -> dict[str, str]:
        return {"task_id": task_id, "status": self.statuses[task_id]}


def attach_tasks(mw: Any, tasks: Tasks) -> None:
    # Use the same task endpoint as the policy's commit client.
    mw._connection.client.commit_session = tasks.commit_session
    mw._connection.client.get_task = tasks.get_task


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("new_policy", [False, True])
async def test_model_failure_keeps_submission_without_mutating_graph(
    env: tuple[Any, ...], cancelled: bool, new_policy: bool
) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100  # Policy submission stays asynchronous below model capacity.
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks) if new_policy else None
    mw = make(
        env,
        reader,
        session_commit_policy=policy,
        commit_config=None if new_policy else CommitConfig(100, 0.5, 5, 1),
    )
    attach_tasks(mw, tasks)
    engine = mw._ov_session_engine
    engine.config = replace(engine.config)
    engine.assembler.token_counter = lambda request: 1000 if request.messages else 0
    state = {"messages": raw_messages()}
    state.update(await mw.abefore_agent(state, runtime(env)))
    state["ov_context"]["completed_runs"] = 3
    before = deepcopy(state)

    async def failing(request: ModelRequest[Any]) -> ModelResponse[Any]:
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("model failed")

    for _ in range(2):
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), failing)
        assert state == before
    assert len(tasks.calls) == 1
    await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"
    assert state["ov_context"]["completed_runs"] == 0
    await call(mw, state, env[4], runtime(env))
    assert not engine._commit_progress and len(tasks.calls) == 1


async def test_compression_timeout_keeps_submission_and_cleanup_checkpoints_it(
    env: tuple[Any, ...],
) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks)
    mw = make(env, reader, session_commit_policy=policy)
    attach_tasks(mw, tasks)
    engine = mw._ov_session_engine
    engine.config = replace(engine.config, commit_wait_timeout=0.01, poll_interval=0.001)
    engine.assembler.token_counter = lambda request: 1_000_000
    state = {"messages": raw_messages()}
    state.update(await mw.abefore_agent(state, runtime(env)))
    for _ in range(2):
        with pytest.raises(ContextBudgetExceeded, match="Capacity compression failed") as raised:
            await call(mw, state, env[4], runtime(env))
        assert isinstance(raised.value.__cause__, TimeoutError)
    assert len(tasks.calls) == 1 and state["ov_context"]["pending_commit"] is None

    class Graph:
        async def aget_state(self, config: Any) -> Any:
            return SimpleNamespace(values=state)

        async def aupdate_state(self, config: Any, update: Any) -> None:
            state.update(update)

    await mw.acleanup_run(Graph(), {})
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"
    fresh = make(env, reader, session_commit_policy=policy)
    attach_tasks(fresh, tasks)
    state.update(await fresh.abefore_agent(state, runtime(env, "r2")))
    await call(fresh, state, env[4], runtime(env, "r2"))
    assert len(tasks.calls) == 1


@pytest.mark.parametrize("status", ["failed", "cancelled", "canceled"])
async def test_known_terminal_task_allows_another_submission(
    env: tuple[Any, ...], status: str
) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    mw = make(
        env,
        reader,
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks),
    )
    attach_tasks(mw, tasks)
    state = {"messages": raw_messages()}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    tasks.statuses["task-1"] = status
    with pytest.raises(ContextBudgetExceeded, match="compression task did not complete"):
        await call(mw, state, env[4], runtime(env))
    assert mw._context(state, runtime(env)).pending_commit is None
    await call(mw, state, env[4], runtime(env))
    assert len(tasks.calls) == 2
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-2"


@pytest.mark.parametrize("status", [401, 404, 503])
async def test_task_query_error_preserves_pending_submission(
    env: tuple[Any, ...], status: int
) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    mw = make(
        env,
        reader,
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks),
    )
    attach_tasks(mw, tasks)
    state = {"messages": raw_messages()}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))

    async def query_error(task_id: str, *, identity=None) -> dict[str, str]:
        raise OVHTTPError(status)

    mw._connection.client.get_task = query_error
    with pytest.raises(OVHTTPError):
        await call(mw, state, env[4], runtime(env))
    mw._connection.client.get_task = tasks.get_task
    await call(mw, state, env[4], runtime(env))
    assert len(tasks.calls) == 1
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"


async def test_failure_during_wait_is_retained_until_cleanup(env: tuple[Any, ...]) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks)
    mw = make(env, reader, session_commit_policy=policy)
    attach_tasks(mw, tasks)
    state = {"messages": raw_messages()}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    polls = 0

    async def failing_task(task_id: str, *, identity=None) -> dict[str, str]:
        nonlocal polls
        polls += 1
        return {"status": "running" if polls <= 2 else "failed"}

    mw._connection.client.get_task = failing_task
    mw._ov_session_engine.assembler.token_counter = lambda request: 1_000_000
    with pytest.raises(ContextBudgetExceeded, match="compression task did not complete"):
        await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"
    # The original snapshot remains untouched; the terminal outcome is recovered.
    recovered = mw._context(state, runtime(env))
    assert recovered.pending_commit is None
    mw._connection.client.get_task = tasks.get_task
    mw._ov_session_engine.assembler.token_counter = lambda request: 10
    await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-2"


async def test_published_archive_is_not_replaced_by_old_pending_task(env: tuple[Any, ...]) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    mw = make(
        env,
        reader,
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks),
    )
    attach_tasks(mw, tasks)
    raw = raw_messages()
    state = {"messages": raw}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    reader.snapshots["s"] = (
        "archive-1",
        [HumanMessage(id="ov:summary", content="summary"), raw[-1]],
    )
    reader.tokens = 0
    tasks.statuses["task-1"] = "completed"

    async def failed(request: ModelRequest[Any]) -> ModelResponse[Any]:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), failed)
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"
    output = await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["pending_commit"] is None
    assert state["ov_context"]["archive_id"] == "archive-1"
    assert [m.id for m in output] == ["ov:summary", "current"] and len(tasks.calls) == 1


@pytest.mark.parametrize("metrics_enabled", [False, True])
@pytest.mark.parametrize("new_policy", [False, True])
async def test_count_fixed_and_full_request_once_per_assembly(
    env: tuple[Any, ...], metrics_enabled: bool, new_policy: bool
) -> None:
    reader, tasks = Reader(), Tasks()
    reader.tokens = 0
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks) if new_policy else None
    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env,
        reader,
        session_commit_policy=policy,
        metrics=Metrics(lambda event, **fields: samples.append((event, fields)))
        if metrics_enabled
        else None,
    )
    counts: list[ModelRequest[Any]] = []

    def counter(request: ModelRequest[Any]) -> int:
        counts.append(request)
        return 10

    mw._ov_session_engine.assembler.token_counter = counter
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    assert len([r for r in counts if r.messages]) == 1
    assert len([r for r in counts if not r.messages]) == 1
    if metrics_enabled:
        assert dict(samples)["context.estimate"]["estimated_input_tokens"] == 10


@pytest.mark.parametrize("same_id", [False, True])
async def test_latest_writer_tool_view_and_native_fields_win(
    env: tuple[Any, ...], same_id: bool
) -> None:
    latest = ToolMessage(
        id="native-tool",
        tool_call_id="c",
        content="new short view",
        artifact={"new": True},
        additional_kwargs={"trace": "new"},
        status="error",
    )

    class Writer(SuppliedWriter):
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
            return WriteResult([latest], len(raw_messages), 1)

    mw = make(env, Reader(), Writer())
    assistant = AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}])
    old = latest.model_copy(
        update={"id": latest.id if same_id else "other-id", "content": "old view"}
    )
    context = OVContext(
        "s", messages=[assistant, old], write_index=1, run_id="r", owner=("a", "u", "p", "s")
    )
    await mw._ov_session_engine.write(context, [assistant, latest], identity=context.identity)
    assert len(context.messages) == 2 and context.messages[-1].model_dump() == latest.model_dump()


async def test_real_graph_cleanup_preserves_submission_after_failed_model(
    env: tuple[Any, ...],
) -> None:
    class BrokenModel(ToolLoopModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model failed")

    reader, tasks = Reader(), Tasks()
    reader.tokens = 100
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), tasks)
    mw = make(env, reader, session_commit_policy=policy)
    attach_tasks(mw, tasks)
    saver = InMemorySaver()
    graph = create_agent(
        model=BrokenModel(),
        middleware=[mw],
        checkpointer=saver,
        context_schema=CallContext,
    )
    config = {"configurable": {"thread_id": "s"}}
    try:
        for attempt in range(2):
            with pytest.raises(RuntimeError, match="model failed"):
                await graph.ainvoke(
                    {"messages": raw_messages()} if attempt == 0 else None,
                    config=config,
                    context=runtime(env).context,
                )
        assert len(tasks.calls) == 1
    finally:
        await mw.acleanup_run(graph, config)
    saved = (await graph.aget_state(config)).values["ov_context"]
    assert saved["pending_commit"]["task_id"] == "task-1"
    assert saved["run_id"] is None and len(tasks.calls) == 1
    fresh = make(env, reader, session_commit_policy=policy)
    attach_tasks(fresh, tasks)
    state = {"messages": raw_messages(), "ov_context": saved}
    state.update(await fresh.abefore_agent(state, runtime(env, "r2")))
    await call(fresh, state, env[4], runtime(env, "r2"))
    assert len(tasks.calls) == 1
