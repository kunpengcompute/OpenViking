"""Normal hooks need no task discovery; commit accounting uses existing reads."""

from langchain_core.messages import AIMessage, HumanMessage

from openviking.integrations.langchain.ov_context import OVContext
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot

from .test_session_hooks import call, runtime
from .test_summary_probes import modern


async def test_initialized_next_run_does_not_read_session_or_list_tasks(env):
    mw, _, _, state, _ = await modern(env)
    state["messages"].append(AIMessage(id="answer", content="answer"))
    state.update(await mw.aafter_model(state, runtime(env)))
    state.update(await mw.aafter_agent(state, runtime(env)))
    prior = len(env[0].requests)
    state["messages"].append(HumanMessage(id="next", content="next"))
    state.update(await mw.abefore_agent(state, runtime(env, "r2")))
    requests = env[0].requests[prior:]
    assert not any(req.method == "GET" for req, _ in requests)
    assert env[0].counts["tasks"] == 0
    assert not hasattr(mw._ov_session_engine, "restore_progress")
    assert not hasattr(mw._ov_session_engine, "_restored")


async def test_new_server_commit_resets_runs_once_at_existing_probe(env):
    mw, _, reader, state, meta = await modern(env, automatic=True)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.completed_runs = 4
    meta.update(commit_count=1, latest_finished_archive_id="archive_001")
    reader.snapshots["s"] = ("archive_001", [state["messages"][0]])
    await engine.check_summary(context, identity=context.identity)
    assert context.completed_runs == 0 and context.commit_count == 1
    context.completed_runs = 1
    await engine.check_summary(context, identity=context.identity)
    assert context.completed_runs == 1
    assert env[0].counts["tasks"] == 0


async def test_archive_finishing_does_not_reset_runs_counted_after_submission(env):
    mw, _, reader, state, meta = await modern(env, commits=2, automatic=True)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.completed_runs = 2
    context.pending_commit = {
        "task_id": None,
        "archive_id": None,
        "target_archive_id": "archive_002",
    }
    meta["latest_finished_archive_id"] = "archive_002"
    reader.snapshots["s"] = ("archive_002", [state["messages"][0]])
    await engine.check_summary(context, identity=context.identity)
    assert context.archive_id == "archive_002" and context.pending_commit is None
    assert context.completed_runs == 2


async def test_snapshot_ahead_of_metadata_counts_new_commit_once(env):
    mw, _, reader, state, meta = await modern(env, commits=1, automatic=True)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.completed_runs = 3
    meta["latest_finished_archive_id"] = "archive_001"
    reader.snapshots["s"] = ("archive_002", [state["messages"][0]])
    await engine.check_summary(context, identity=context.identity)
    assert context.commit_count == 2 and context.completed_runs == 0
    context.completed_runs = 1
    meta.update(commit_count=2, latest_finished_archive_id="archive_002")
    await engine.check_summary(context, identity=context.identity)
    assert context.completed_runs == 1


async def test_existing_snapshot_with_unknown_count_keeps_completed_runs(env):
    mw, _, _, state, _ = await modern(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.archive_id = "archive_002"
    context.commit_count = None
    context.completed_runs = 3
    engine._apply_snapshot(context, ContextSnapshot(context.messages, "archive_002"))
    assert context.commit_count == 2 and context.completed_runs == 3


async def test_receipt_confirmed_commit_is_not_counted_twice(env):
    mw, _, reader, state, meta = await modern(env, automatic=True)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.messages = [
        HumanMessage(id="old", content="old"),
        AIMessage(id="old-a", content="a"),
        *context.messages,
    ]
    context.completed_runs = 4
    context.commit_count = None  # Legacy state, but a modern receipt provides the target.

    async def committed(*args, **kwargs):
        return {"archived": True, "archive_uri": "viking://s/history/archive_002"}

    engine._send_commit = committed
    engine.session_commit_policy = None
    assert await engine.commit(context, identity=context.identity)
    assert context.completed_runs == 0 and context.commit_count == 2
    context.completed_runs = 1
    engine._remember_commit(context)
    meta.update(commit_count=2, latest_finished_archive_id="archive_002")
    reader.snapshots["s"] = (
        "archive_002",
        [HumanMessage(id="ov_summary:archive_002", content="Earlier work"), state["messages"][0]],
    )
    await engine.check_summary(context, identity=context.identity)
    assert context.completed_runs == 1


async def test_independent_commit_observed_by_existing_token_read_blocks_resubmit(env):
    mw, _, _, state, meta = await modern(env, automatic=True)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    context.completed_runs = 4
    engine.invalidate_tokens(context)
    meta["commit_count"] = 1
    assert await engine.pending_tokens(context, identity=context.identity) is None
    assert context.commit_count == 1 and context.completed_runs == 0
    assert context.summary_check_countdown == 0


async def test_sdk_task_list_failure_does_not_block_dialogue(env):
    mw, _, _, state, _ = await modern(env)
    env[0].error["tasks"] = 503
    assert any(m.id == "u" for m in await call(mw, state, env[4], runtime(env)))
    assert env[0].counts["tasks"] == 0
