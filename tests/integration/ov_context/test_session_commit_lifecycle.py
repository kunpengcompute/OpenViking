"""Main middleware persists responses and commits only at safe lifecycle points."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy
from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.commit import (
    CommitConfig,
    CommitResult,
    load_commit_config,
)
from openviking_cli.exceptions import UnavailableError
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_graph import CallContext, ToolLoopModel, read_mail
from .test_hook_design import CommitClient, Reader
from .test_io_contracts import SuppliedWriter
from .test_session_hooks import middleware as legacy_middleware
from .test_session_hooks import runtime

pytestmark = pytest.mark.asyncio


def build_middleware(
    env: tuple[Any, ...], *, keep: int = 2, tokens: int = 100, runs: int = 5
) -> tuple[OpenVikingContextMiddleware, SuppliedWriter, CommitClient]:
    reader, writer, client = Reader(), SuppliedWriter(), CommitClient()
    env[0].session_meta_hook = lambda sid: {
        "latest_finished_archive_id": None,
        "pending_tokens": tokens,
    }
    middleware = OpenVikingContextMiddleware(
        message_writer=writer,
        client=TransportSDKAdapter(env[1]),
        context_reader=reader,
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, runs, keep), client),
        context_config=OVContextConfig(context_window=100000),
    )
    return middleware, writer, client


def initial() -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(id="old", content="Previous question"),
            AIMessage(id="old-answer", content="Previous answer"),
            HumanMessage(id="current", content="Current task"),
        ]
    }


async def test_final_response_is_written_then_committed_once(env: tuple[Any, ...]) -> None:
    middleware, writer, client = build_middleware(env)
    state = initial()
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(AIMessage(id="final", content="Answer"))
    state.update(await middleware.aafter_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 4
    assert client.calls == [("s", 2)]
    assert await middleware.aafter_model(state, runtime(env)) == {}
    assert len(writer.calls) == 2 and len(client.calls) == 1


async def test_pending_tools_are_written_without_commit(env: tuple[Any, ...]) -> None:
    middleware, _, client = build_middleware(env)
    state = initial()
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(
        AIMessage(id="call", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}])
    )
    state.update(await middleware.aafter_model(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 4
    assert client.calls == []


async def test_return_direct_results_are_flushed_and_run_counted_once(env: tuple[Any, ...]) -> None:
    middleware, writer, client = build_middleware(env, keep=3, tokens=1, runs=1)
    state = initial()
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(
        AIMessage(id="call", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}])
    )
    state.update(await middleware.aafter_model(state, runtime(env)))
    state["messages"].append(ToolMessage(id="result", content="Direct result", tool_call_id="c"))
    state.update(await middleware.aafter_agent(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 5
    assert state["ov_context"]["run_id"] is None
    assert state["ov_context"]["completed_runs"] == 0
    assert state["ov_context"]["last_completed_run"] == "r"
    assert client.calls == [("s", 0)]
    assert writer.calls[-1][1] == 4
    assert not middleware._ov_session_engine.active
    assert await middleware.aafter_agent(state, runtime(env)) == {}
    assert len(client.calls) == 1


async def test_completed_run_threshold_is_checked_at_after_agent(env: tuple[Any, ...]) -> None:
    middleware, _, client = build_middleware(env, tokens=1, runs=1)
    state = initial()
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(AIMessage(id="final", content="Answer"))
    state.update(await middleware.aafter_model(state, runtime(env)))
    assert client.calls == []
    state.update(await middleware.aafter_agent(state, runtime(env)))
    assert client.calls == [("s", 0)]
    assert state["ov_context"]["completed_runs"] == 0


@pytest.mark.parametrize("keep", [0, 8])
@pytest.mark.parametrize("configured", [True, False], ids=["commit-policy", "legacy-config"])
async def test_after_agent_can_archive_a_long_first_run(
    env: tuple[Any, ...], keep: int, configured: bool
) -> None:
    if configured:
        middleware, _, client = build_middleware(env, keep=keep, tokens=1, runs=1)
    else:
        middleware = legacy_middleware(env, keep_recent_count=keep, commit_every_runs=1)
        client = None
    user = HumanMessage(id="current", content="Process the mailbox")
    state = {"messages": [user]}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    context = OVContext.from_state(state["ov_context"])
    context.inject_recall("Reference memory", "success")
    state["ov_context"] = context.to_state()
    for index in range(19):
        calls = [{"id": f"c-{index}-{i}", "name": "read_mail", "args": {}} for i in range(2)]
        state["messages"].append(AIMessage(id=f"a-{index}", content="", tool_calls=calls))
        state["messages"].extend(
            ToolMessage(id=f"t-{index}-{i}", content="Mail body", tool_call_id=call["id"])
            for i, call in enumerate(calls)
        )
    state["messages"].extend(
        [AIMessage(id="review", content="Checked"), AIMessage(id="final", content="Done")]
    )
    assert len(state["messages"]) == 60
    raw = deepcopy(state["messages"])
    state.update(await middleware.aafter_agent(state, runtime(env)))
    saved = state["ov_context"]
    engine = middleware._ov_session_engine
    assert engine.session_commit_policy.config.keep_recent_count == keep
    plan = engine._refresh_progress[OVContext.from_state(saved).session_key].plan
    assert plan.keep_recent_count == 0
    assert plan.archive_message_count == 60
    assert not plan.recent_messages
    if client:
        assert client.calls == [("s", 0)]
    else:
        payloads = [body for req, body in env[0].requests if req.url.path.endswith("/commit")]
        assert payloads == [{"keep_recent_count": 0}]
    assert saved["pending_commit"] is not None
    assert saved["write_index"] == 60
    assert saved["run_id"] is None and saved["original_user"] is None
    assert saved["recall_status"] is None and not saved["recall_done"]
    assert saved["last_completed_run"] == "r"
    assert saved["messages"][0] == user
    assert state["messages"] == raw
    assert not middleware._ov_session_engine.active
    assert await middleware.aafter_agent(state, runtime(env)) == {}


async def test_after_agent_always_policy_explicitly_uses_zero_retention(
    env: tuple[Any, ...],
) -> None:
    middleware = OpenVikingContextMiddleware(
        client=TransportSDKAdapter(env[1]),
        context_reader=Reader(),
        message_writer=SuppliedWriter(),
        commit_policy=OpenVikingCommitPolicy(mode="always"),
        context_config=OVContextConfig(context_window=100000),
    )
    assert middleware._ov_session_engine.session_commit_policy is None
    state = {"messages": [HumanMessage(id="current", content="Task")]}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(AIMessage(id="final", content="Done"))
    commit = AsyncMock(return_value=True)
    middleware._ov_session_engine.commit = commit

    await middleware.aafter_agent(state, runtime(env))

    assert commit.await_count == 1
    assert commit.await_args.kwargs["keep_recent_count"] == 0
    assert commit.await_args.args[0].run_id is None


async def test_completed_first_run_is_committed_and_checkpointed_by_real_graph(
    env: tuple[Any, ...],
) -> None:
    middleware, writer, client = build_middleware(env, keep=8, tokens=1, runs=1)
    graph = create_agent(
        model=ToolLoopModel(),
        tools=[read_mail],
        middleware=[middleware],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "s"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(id="current", content="Read the mail")]},
        config=config,
        context=CallContext(env[3]),
    )
    saved = (await graph.aget_state(config)).values["ov_context"]
    assert result["messages"][-1].content == "Mail summarized"
    assert client.calls == [("s", 0)]
    assert saved["write_index"] == len(result["messages"]) == 4
    assert saved["pending_commit"]["task_id"] == "compress-task"
    assert saved["run_id"] is None and saved["original_user"] is None
    assert saved["messages"][0].content == "Read the mail"
    assert all(call[2] == "r" for call in writer.calls)


@pytest.mark.parametrize("pending_tools", [False, True])
async def test_after_agent_retains_commit_safety_checks(
    env: tuple[Any, ...], pending_tools: bool
) -> None:
    middleware, _, client = build_middleware(env, keep=0, tokens=1 if pending_tools else 0, runs=1)
    state = {"messages": [HumanMessage(id="current", content="Task")]}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(
        AIMessage(
            id="answer",
            content="" if pending_tools else "Done",
            tool_calls=[{"id": "c", "name": "read_mail", "args": {}}] if pending_tools else [],
        )
    )
    state.update(await middleware.aafter_agent(state, runtime(env)))
    assert client.calls == []
    assert state["ov_context"]["completed_runs"] == 1
    assert state["ov_context"]["run_id"] is None


async def test_after_agent_transient_commit_failure_preserves_completed_run(
    env: tuple[Any, ...],
) -> None:
    middleware, _, client = build_middleware(env, keep=0, tokens=1, runs=1)
    engine = middleware._ov_session_engine
    engine.config = replace(engine.config, retry_attempts=1)
    state = {"messages": [HumanMessage(id="current", content="Task")]}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(AIMessage(id="final", content="Done"))
    engine.pending_tokens = AsyncMock(side_effect=UnavailableError("OV unavailable"))
    state.update(await middleware.aafter_agent(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 2
    assert state["ov_context"]["completed_runs"] == 1
    assert state["ov_context"]["last_completed_run"] == "r"
    assert state["ov_context"]["run_id"] is None
    assert client.calls == [] and not engine.active
    assert await middleware.aafter_agent(state, runtime(env)) == {}


async def test_failed_write_does_not_publish_a_new_cursor(env: tuple[Any, ...]) -> None:
    middleware, writer, client = build_middleware(env)
    state = initial()
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"].append(AIMessage(id="final", content="Answer"))
    before = deepcopy(state)
    writer.error = RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        await middleware.aafter_model(state, runtime(env))
    assert state == before
    assert client.calls == []


async def test_commit_config_loader_returns_validated_config(tmp_path: Path) -> None:
    path = tmp_path / "commit.toml"
    path.write_text(
        "[commit]\ntoken_budget=100\ncommit_token_threshold_ratio=0.5\n"
        "completed_run_threshold=5\nkeep_recent_count=2\n"
    )
    assert load_commit_config(path) == CommitConfig(100, 0.5, 5, 2)
    assert CommitResult(status="skipped").status == "skipped"
