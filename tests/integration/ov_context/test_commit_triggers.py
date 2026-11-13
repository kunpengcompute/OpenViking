"""Capacity triggers use the new policy without bypassing session safety."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.middleware import (
    OpenVikingCommitPolicy,
    OpenVikingContextMiddleware,
)
from openviking.integrations.langchain.ov_context import ContextBudgetExceeded, OVContext
from openviking.integrations.langchain.ov_context.commit import CommitConfig, CommitTrigger
from tests.ov_context_support.commit_settings import EXAMPLE_COMMIT_CONFIG
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_hook_design import CommitClient, Reader, make
from .test_io_contracts import SuppliedWriter
from .test_session_hooks import call, runtime

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("trigger", ["policy", "capacity"])
@pytest.mark.parametrize("tokens", [0, 1, 100])
async def test_explicit_triggers_bypass_only_thresholds(
    env: tuple[Any, ...], trigger: CommitTrigger, tokens: int
) -> None:
    sdk = CommitClient()
    policy = OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), sdk)
    result = await policy.commit("s", tokens, 0, identity=env[3].sdk_identity, trigger=trigger)
    expected = tokens > 0 and (tokens >= 50 or trigger != "policy")
    assert sdk.calls == ([("s", 1)] if expected else [])
    assert result.status == ("archived" if expected else "skipped")
    if expected:
        assert result.reason == ("pending_capacity" if trigger == "policy" else trigger)


async def test_configured_policy_still_obeys_thresholds(env: tuple[Any, ...]) -> None:
    reader = Reader()
    reader.tokens = 1
    env[0].session_meta_hook = lambda sid: {"pending_tokens": reader.tokens}
    mw = OpenVikingContextMiddleware(
        client=TransportSDKAdapter(env[1]),
        context_reader=reader,
        message_writer=SuppliedWriter(),
        token_counter=lambda request: 10,
        commit_config=EXAMPLE_COMMIT_CONFIG,
    )
    assert mw._ov_session_engine.session_commit_policy.config == EXAMPLE_COMMIT_CONFIG
    state = {"messages": [HumanMessage(id=str(i), content="question") for i in range(10)]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    state.update(await mw.aafter_agent(state, runtime(env)))
    assert not env[0].counts["commit"]
    assert state["ov_context"]["completed_runs"] == 1


@pytest.mark.parametrize("tokens,runs,expected", [(1, 4, False), (1, 5, True), (0, 5, False)])
async def test_completed_run_threshold_remains_effective(
    env: tuple[Any, ...], tokens: int, runs: int, expected: bool
) -> None:
    sdk = CommitClient()
    policy = OpenVikingCommitPolicy(EXAMPLE_COMMIT_CONFIG, sdk)
    result = await policy.commit("s", tokens, runs, identity=env[3].sdk_identity)
    assert sdk.calls == ([("s", 8)] if expected else [])
    assert result.status == ("archived" if expected else "skipped")
    if expected:
        assert result.reason == "completed_runs"


@pytest.mark.parametrize("input_tokens", [90, 200], ids=["capacity-ratio", "over-budget"])
@pytest.mark.parametrize("defaults", [False, True], ids=["custom-retention", "default-retention"])
async def test_model_budget_triggers_compression_below_pending_threshold(
    env: tuple[Any, ...], input_tokens: int, defaults: bool
) -> None:
    reader = Reader()
    reader.tokens = 1  # Model/system overhead can exhaust the window before OV's threshold.
    user = HumanMessage(id="current", content="question")
    config = EXAMPLE_COMMIT_CONFIG if defaults else CommitConfig(128_000, 0.7, 5, 1)
    raw = [HumanMessage(id=f"old-{i}", content="old") for i in range(10 if defaults else 1)] + [
        user
    ]

    class Publisher(CommitClient):
        async def commit_session(
            self, session_id: str, *, keep_recent_count: int, identity: Any = None
        ) -> dict[str, Any]:
            assert identity == env[3].sdk_identity
            result = await super().commit_session(session_id, keep_recent_count=keep_recent_count)
            reader.snapshots[session_id] = (
                "archive-1",
                [
                    HumanMessage(id="ov:summary", content="history summary"),
                    *raw[-keep_recent_count:],
                ],
            )
            return result

    sdk = Publisher()
    mw = make(env, reader, commit_config=config)
    mw._ov_session_engine.session_commit_policy.ov_client = sdk
    engine = mw._ov_session_engine
    engine.assembler.config = replace(
        engine.config, context_window=100, output_reserve=0, safety_margin=0
    )
    engine.assembler.token_counter = lambda request: (
        20
        if request.messages and request.messages[0].id == "ov:summary"
        else (
            input_tokens
            if request.messages and request.messages[-1].id == user.id
            else (10 if request.messages else 0)
        )
    )
    state = {"messages": raw}
    state.update(await mw.abefore_agent(state, runtime(env)))
    output = await call(mw, state, env[4], runtime(env))
    assert sdk.calls == [("s", config.keep_recent_count)]
    assert not env[0].counts["commit"]  # No fallback into the legacy client.
    assert [m.id for m in output] == [
        "ov:summary",
        *[m.id for m in raw[-config.keep_recent_count :]],
    ]
    assert state["ov_context"]["pending_commit"] is None
    assert state["messages"] == raw


@pytest.mark.parametrize("tokens", [None, 0])
async def test_capacity_trigger_does_not_invent_pending_tokens(
    env: tuple[Any, ...], tokens: int | None
) -> None:
    reader, sdk = Reader(), CommitClient()
    reader.tokens = tokens
    mw = make(
        env, reader, session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 1), sdk)
    )
    state = {
        "messages": [
            HumanMessage(id="old", content="history"),
            HumanMessage(id="u", content="current"),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime(env)))
    mw._ov_session_engine.assembler.token_counter = lambda request: 1_000_000
    with pytest.raises(ContextBudgetExceeded):
        await call(mw, state, env[4], runtime(env))
    assert not sdk.calls and not env[0].counts["commit"]


@pytest.mark.parametrize("trigger", ["capacity"])
@pytest.mark.parametrize("boundary", ["tool-group", "current-run"])
async def test_explicit_trigger_keeps_tool_groups_and_can_archive_current_run_prefix(
    env: tuple[Any, ...], trigger: CommitTrigger, boundary: str
) -> None:
    env[0].sessions.add("s")
    sdk = CommitClient()
    mw = make(
        env,
        Reader(),
        session_commit_policy=OpenVikingCommitPolicy(CommitConfig(100, 0.5, 5, 2), sdk),
    )
    messages = [
        HumanMessage(id="old", content="history"),
        AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}]),
        ToolMessage(id="t", content="result", tool_call_id="c"),
        HumanMessage(id="u", content="current"),
    ]
    user = messages[-1]
    if boundary == "current-run":
        messages = [messages[0], user, messages[1], messages[2]]
    context = OVContext("s", messages=messages, owner=env[3].session_key)
    context.begin_run("r", user)
    before = deepcopy(context.to_state())
    assert await mw._ov_session_engine.commit(
        context, pending_tokens=1, trigger=trigger, identity=context.identity
    )
    assert sdk.calls == [("s", 3 if boundary == "tool-group" else 2)]
    assert context.messages == before["messages"]
    assert context.write_index == before["write_index"]
    assert context.pending_commit["task_id"] == "compress-task"
