"""Archive probes must save reads without acknowledging an unapplied snapshot."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot

from .test_hook_design import Reader, make
from .test_session_hooks import call, runtime
from .test_token_receipts import ReceiptWriter, reads


async def modern(env, *, commits=0, automatic=False, interval=2, tokens=17):
    writer, reader = ReceiptWriter(tokens), Reader()
    reader.tokens = 17
    mw = make(env, reader, writer, commit_config=CommitConfig(100000, 0.7, 100, 0))
    engine = mw._ov_session_engine
    # The default is exercised without referencing the new configuration field.
    if interval != 2:
        engine.config = replace(engine.config, summary_check_interval=interval)
    meta = {
        "commit_count": commits,
        "auto_commit_policy": {} if automatic else None,
        "latest_finished_archive_id": None,
        "pending_tokens": 17,
    }
    if automatic:
        meta["auto_commit_policy"] = {"message_count_threshold": 100}
    env[0].sessions.add("s")
    env[0].session_meta_hook = lambda _: dict(meta)
    state = {"messages": [HumanMessage(id="u", content="hello")]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    return mw, writer, reader, state, meta


async def test_no_archive_or_external_policy_never_reads_boundary_in_model_loop(env):
    mw, writer, _, state, _ = await modern(env)
    before = reads(env)
    for i in range(4):
        assert any(m.id == "u" for m in await call(mw, state, env[4], runtime(env)))
        state["messages"].append(AIMessage(id=f"a{i}", content="hello"))
        state.update(await mw.aafter_model(state, runtime(env)))
    assert reads(env) == before
    assert state["ov_context"]["archive_id"] is None


async def test_legacy_token_policy_still_reads_counts_without_writer_receipt(env):
    from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy

    reader = Reader()
    mw = make(
        env,
        reader,
        commit_policy=OpenVikingCommitPolicy(mode="pending_tokens", pending_token_threshold=50),
    )
    env[0].sessions.add("s")
    env[0].session_meta_hook = lambda _: {
        "commit_count": 0,
        "auto_commit_policy": None,
        "latest_finished_archive_id": None,
        "pending_tokens": 100,
    }
    state = {
        "messages": [
            HumanMessage(id="old", content="history"),
            AIMessage(id="old-a", content="answer"),
            HumanMessage(id="u", content="new"),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    assert env[0].counts["commit"] == 1


async def test_pending_archive_checks_first_then_every_two_model_calls(env):
    mw, _, _, state, _ = await modern(env, commits=2)
    start = reads(env)
    counts = []
    for _ in range(5):
        await call(mw, state, env[4], runtime(env))
        counts.append(reads(env) - start)
    assert counts == [1, 1, 2, 2, 3]
    assert state["ov_context"]["archive_id"] is None


async def test_configurable_probe_interval_is_counted_in_model_calls(env):
    mw, _, _, state, _ = await modern(env, commits=1, interval=3)
    before = reads(env)
    observed = []
    for _ in range(7):
        await call(mw, state, env[4], runtime(env))
        observed.append(reads(env) - before)
    assert observed == [1, 1, 1, 2, 2, 2, 3]


async def test_failed_model_does_not_persist_an_applied_version_or_stop_retry(env):
    mw, _, reader, state, meta = await modern(env, commits=1)
    meta["latest_finished_archive_id"] = "archive_001"
    reader.snapshots["s"] = ("archive_001", [state["messages"][0]])
    previous = deepcopy(state)

    async def fail(request):
        raise RuntimeError("model failed after context switch")

    with pytest.raises(RuntimeError, match="model failed"):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), fail)
    assert state == previous
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before + 1
    assert state["ov_context"]["archive_id"] == "archive_001"


@pytest.mark.parametrize(
    "count,boundary,tokens", [(3, "archive_002", 100000), (1, "archive_001", 0)]
)
async def test_retry_keeps_newer_observed_progress_instead_of_old_recovery_cache(
    env, count, boundary, tokens
):
    reader, writer = Reader(), ReceiptWriter(0)
    mw = make(env, reader, writer, commit_config=CommitConfig(100000, 0.7, 100, 1))
    meta = {
        "commit_count": 1,
        "auto_commit_policy": {"message_count_threshold": 100},
        "latest_finished_archive_id": None,
        "pending_tokens": 0,
    }
    env[0].sessions.add("s")
    env[0].session_meta_hook = lambda _: dict(meta)
    tail = [
        HumanMessage(id="old", content="history"),
        AIMessage(id="old-a", content="answer"),
        HumanMessage(id="u", content="new"),
    ]
    state = {"messages": tail[:]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    meta["latest_finished_archive_id"] = "archive_001"
    reader.snapshots["s"] = ("archive_001", [HumanMessage(id="sum1", content="summary1"), *tail])

    async def fail(request):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), fail)
    assert state["ov_context"]["archive_id"] is None
    meta.update(commit_count=count, latest_finished_archive_id=boundary, pending_tokens=tokens)
    reader.snapshots["s"] = ("archive_002", [HumanMessage(id="sum2", content="summary2"), *tail])
    commit = AsyncMock(
        return_value={
            "archived": True,
            "task_id": "task-4",
            "archive_uri": "viking://s/history/archive_004",
        }
    )
    mw._connection.client.commit_session = commit
    await call(mw, state, env[4], runtime(env))
    assert commit.await_count == 0
    assert state["ov_context"]["archive_id"] == "archive_002"
    assert state["ov_context"]["commit_count"] == max(count, 2)


async def test_recovered_new_commit_is_probed_immediately_after_failed_model(env):
    reader, writer = Reader(), ReceiptWriter(80000)
    mw = make(env, reader, writer, commit_config=CommitConfig(100000, 0.7, 100, 1))
    meta = {
        "commit_count": 0,
        "auto_commit_policy": None,
        "latest_finished_archive_id": None,
        "pending_tokens": 80000,
    }
    env[0].sessions.add("s")
    env[0].session_meta_hook = lambda _: dict(meta)
    state = {
        "messages": [
            HumanMessage(id="old", content="history"),
            AIMessage(id="old-a", content="answer"),
            HumanMessage(id="u", content="new"),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime(env)))
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    await engine.check_summary(context, identity=(context).identity)
    state["ov_context"] = context.to_state()
    engine.assembler.token_counter = lambda _: 1000

    async def commit(*args, **kwargs):
        meta["commit_count"] = 1
        return {
            "archived": True,
            "task_id": "task-1",
            "archive_uri": "viking://s/history/archive_001",
        }

    mw._connection.client.commit_session = commit
    mw._connection.client.get_task = AsyncMock(return_value={"status": "running"})

    async def fail(request):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), fail)
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before + 1
    assert state["ov_context"]["pending_commit"]["task_id"] == "task-1"


async def test_missing_commit_count_disables_optimization_for_known_checkpoint(env):
    mw, _, _, state, meta = await modern(env)
    context = OVContext.from_state(state["ov_context"])
    meta.pop("commit_count")
    await mw._ov_session_engine.check_summary(context, identity=(context).identity)
    state["ov_context"] = context.to_state()
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before + 1


async def test_probe_cadence_survives_checkpoint_and_new_middleware(env):
    mw, writer, reader, state, meta = await modern(env, commits=1)
    await call(mw, state, env[4], runtime(env))
    serde = JsonPlusSerializer(allowed_msgpack_modules=[])
    state = serde.loads_typed(serde.dumps_typed(state))
    fresh = make(env, reader, writer, commit_config=CommitConfig(100000, 0.7, 100, 0))
    env[0].session_meta_hook = lambda _: dict(meta)
    state.update(await fresh.abefore_agent(state, runtime(env)))
    before = reads(env)
    await call(fresh, state, env[4], runtime(env))
    assert reads(env) == before
    await call(fresh, state, env[4], runtime(env))
    assert reads(env) == before + 1


async def test_automatic_commits_are_discovered_even_without_local_pending(env):
    mw, _, reader, state, meta = await modern(env, automatic=True)
    await call(mw, state, env[4], runtime(env))
    meta.update(commit_count=1, latest_finished_archive_id="archive_001")
    reader.snapshots["s"] = (
        "archive_001",
        [HumanMessage(id="sum", content="summary"), state["messages"][0]],
    )
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before and state["ov_context"]["archive_id"] is None
    await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["archive_id"] == "archive_001"
    assert state["ov_context"]["commit_count"] == 1


async def test_completed_snapshot_may_advance_past_metadata_and_old_task_target(env):
    mw, _, reader, state, meta = await modern(env, commits=2)
    context = OVContext.from_state(state["ov_context"])
    context.pending_commit = {
        "task_id": "task-1",
        "archive_id": None,
        "target_archive_id": "archive_001",
    }
    meta["latest_finished_archive_id"] = "archive_001"
    reader.snapshots["s"] = (
        "archive_002",
        [HumanMessage(id="sum", content="latest"), state["messages"][0]],
    )
    await mw._ov_session_engine.check_summary(context, identity=(context).identity)
    assert context.archive_id == "archive_002" and context.pending_commit is None
    assert context.commit_count == 2
    state["ov_context"] = context.to_state()
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before


@pytest.mark.parametrize("include_target", [True, False])
async def test_newer_snapshot_does_not_reuse_an_older_refresh_plan_after_retry(env, include_target):
    mw, _, reader, state, meta = await modern(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    user = state["messages"][0]
    context.messages[:0] = [
        HumanMessage(id="past", content="past"),
        AIMessage(id="past-a", content="past answer"),
    ]
    state["messages"].append(AIMessage(id="stale-tail", content="already covered by archive two"))
    await engine.write(context, state["messages"], identity=context.identity)
    receipt = {"archived": True, "task_id": "task-1"}
    if include_target:
        receipt["archive_uri"] = "viking://s/history/archive_001"
    mw._connection.client.commit_session = AsyncMock(return_value=receipt)
    assert await engine.commit(
        context, pending_tokens=17, trigger="capacity", identity=context.identity
    )
    before_switch = deepcopy(context.to_state())
    meta.update(commit_count=1, latest_finished_archive_id="archive_001")
    reader.snapshots["s"] = (
        "archive_002",
        [HumanMessage(id="ov_summary:archive_002", content="latest"), user],
    )
    await engine.check_summary(context, identity=context.identity)
    assert context.archive_id == "archive_002" and context.pending_commit is None
    assert "stale-tail" not in [m.id for m in context.messages]
    # A failed model did not publish the switched cache. Recovery must retain
    # the newer count without reapplying the older plan's already-covered tail.
    retry_context = OVContext.from_state(before_switch)
    engine.recover_commit(retry_context, persisted=True)
    meta.update(commit_count=2, latest_finished_archive_id="archive_002")
    await engine.check_summary(retry_context, identity=retry_context.identity)
    assert retry_context.archive_id == "archive_002"
    assert "stale-tail" not in [m.id for m in retry_context.messages]


async def test_same_version_reload_keeps_the_pending_refresh_plan(env):
    mw, _, reader, state, meta = await modern(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.messages[:0] = [
        HumanMessage(id="past", content="past"),
        AIMessage(id="past-a", content="answer"),
    ]
    mw._connection.client.commit_session = AsyncMock(
        return_value={
            "archived": True,
            "task_id": "task-1",
            "archive_uri": "viking://s/history/archive_001",
        }
    )
    mw._connection.client.get_task = AsyncMock(return_value={"status": "running"})
    assert await engine.commit(
        context, pending_tokens=17, trigger="capacity", identity=context.identity
    )
    meta.update(commit_count=1, latest_finished_archive_id=None)
    context.context_status = "reload_required"
    reader.snapshots["s"] = (None, list(context.messages))
    await engine.check_summary(context, identity=context.identity)
    assert context.context_status == "ready" and context.archive_id is None
    assert context.pending_commit["target_archive_id"] == "archive_001"
    assert context.session_key in engine._refresh_progress


@pytest.mark.parametrize("failure", ["error", "deadline", "cancel"])
async def test_required_probe_cannot_fall_back_to_an_uncompressed_model_call(env, failure):
    import asyncio

    from openviking.integrations.langchain.ov_context import ContextBudgetExceeded

    from .test_revised_design import SDK, capacity_request
    from .test_revised_design import make as make_revised

    class ReceiptSDK(SDK):
        async def get_session(self, *args, **kwargs):
            return {**await super().get_session(*args, **kwargs), "auto_commit_policy": None}

        async def batch_add_messages(self, *args, **kwargs):
            return {**await super().batch_add_messages(*args, **kwargs), "pending_tokens": 10000}

    sdk = ReceiptSDK()
    mw = make_revised(sdk, resilient=True, io_timeout=0.05, commit_wait_timeout=0.1)
    request = await capacity_request(mw, env[4])
    started = asyncio.Event()

    async def read(*args, **kwargs):
        started.set()
        if failure == "error":
            raise RuntimeError("forced capacity probe unavailable")
        await asyncio.Event().wait()

    sdk.get_session = read
    handler = AsyncMock(return_value=AIMessage(content="unsafe fallback"))
    if failure == "cancel":
        task = asyncio.create_task(mw.awrap_model_call(request, handler))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ContextBudgetExceeded):
            await mw.awrap_model_call(request, handler)
    assert handler.await_count == 0 and sdk.commits == 0
    assert mw.failure is None and mw._preparation_deadline.get() is None


async def test_intermediate_snapshot_does_not_clear_a_newer_pending_commit(env):
    mw, _, reader, state, meta = await modern(env, commits=2)
    context = OVContext.from_state(state["ov_context"])
    context.pending_commit = {
        "task_id": None,
        "archive_id": None,
        "target_archive_id": "archive_002",
    }
    meta["latest_finished_archive_id"] = "archive_001"
    reader.snapshots["s"] = ("archive_001", [state["messages"][0]])
    await mw._ov_session_engine.check_summary(context, identity=(context).identity)
    assert context.archive_id == "archive_001"
    assert context.pending_commit["target_archive_id"] == "archive_002"
    meta["latest_finished_archive_id"] = "archive_002"
    reader.snapshots["s"] = ("archive_002", [state["messages"][0]])
    await mw._ov_session_engine.check_summary(context, identity=(context).identity)
    assert context.archive_id == "archive_002" and context.pending_commit is None


async def test_new_commit_during_snapshot_read_remains_pending(env):
    mw, _, reader, state, meta = await modern(env, commits=1)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.pending_commit = {
        "task_id": "task-1",
        "archive_id": None,
        "target_archive_id": "archive_001",
    }
    meta["latest_finished_archive_id"] = "archive_001"

    async def exchange(*args, **kwargs):
        context.commit_count = 2
        context.pending_commit = {
            "task_id": "task-2",
            "archive_id": None,
            "target_archive_id": "archive_002",
        }
        return ContextSnapshot([state["messages"][0]], "archive_001")

    reader.exchange_context = exchange
    await engine.check_summary(context, identity=(context).identity)
    assert context.archive_id == "archive_001" and context.commit_count == 2
    assert context.pending_commit["task_id"] == "task-2"


@pytest.mark.parametrize("failure", ["old", "duplicate", "tool", "timeout"])
async def test_failed_snapshot_never_advances_applied_version_or_acknowledges(env, failure):
    mw, _, reader, state, meta = await modern(env, commits=2)
    context = OVContext.from_state(state["ov_context"])
    context.archive_id = "archive_001"
    context.pending_commit = {
        "task_id": None,
        "archive_id": "archive_001",
        "target_archive_id": "archive_002",
    }
    meta["latest_finished_archive_id"] = "archive_002"
    user = state["messages"][0]
    messages = [user]
    archive = "archive_002"
    if failure == "old":
        archive = "archive_001"
    if failure == "duplicate":
        messages = [user, user]
    if failure == "tool":
        messages = [user, ToolMessage(id="t", tool_call_id="missing", content="result")]
    if failure == "missing_user":
        messages = []
    if failure == "timeout":
        reader.error = TimeoutError("injected")
    reader.snapshots["s"] = (archive, messages)
    previous = deepcopy(context.to_state())
    with pytest.raises((ContextReadError, ValueError, TimeoutError)):
        await mw._ov_session_engine.check_summary(context, identity=(context).identity)
    assert context.archive_id == previous["archive_id"]
    assert context.messages == previous["messages"]
    assert context.pending_commit == previous["pending_commit"]


async def test_terminal_failure_preserves_applied_version_and_clears_failed_task(env):
    mw, _, reader, state, meta = await modern(env, commits=2)
    context = OVContext.from_state(state["ov_context"])
    context.archive_id = "archive_001"
    context.pending_commit = {
        "task_id": "failed-task",
        "archive_id": "archive_001",
        "target_archive_id": "archive_002",
    }
    meta["latest_finished_archive_id"] = "archive_002"
    mw._connection.client.get_task = AsyncMock(
        return_value={"status": "failed", "result": {"archive_id": "archive_002"}}
    )
    reader.snapshots["s"] = (
        "archive_002",
        [HumanMessage(id="sum", content="older valid overview"), state["messages"][0]],
    )
    from openviking.integrations.langchain.ov_context import ContextBudgetExceeded

    previous = list(context.messages)
    with pytest.raises(ContextBudgetExceeded, match="compression task did not complete"):
        await mw._ov_session_engine.check_summary(context, identity=context.identity)
    assert context.archive_id == "archive_001" and context.pending_commit is None
    assert context.messages == previous
    assert context.commit_count == 2


async def test_budget_exceeded_forces_probe_even_on_skipped_interval(env):
    mw, _, reader, state, meta = await modern(env, commits=1)
    engine = mw._ov_session_engine
    engine.config = replace(
        engine.config, context_window=10000, output_reserve=100, safety_margin=0
    )
    engine.assembler.config = engine.config
    await call(mw, state, env[4], runtime(env))
    meta["latest_finished_archive_id"] = "archive_001"
    user = state["messages"][0]
    reader.snapshots["s"] = ("archive_001", [HumanMessage(id="sum", content="short"), user])
    original = engine.assembler.prepare

    def prepare(request, context, **kwargs):
        from openviking.integrations.langchain.ov_context import ContextBudgetExceeded

        if context.archive_id is None:
            raise ContextBudgetExceeded("needs ready summary")
        return original(request, context, **kwargs)

    engine.assembler.prepare = prepare
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) >= before + 1
    assert state["ov_context"]["archive_id"] == "archive_001"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_summary_probe_interval_rejected(value):
    with pytest.raises(ValueError):
        OVContextConfig(summary_check_interval=value)
