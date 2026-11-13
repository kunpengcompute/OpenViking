"""Archive current-Run history without losing its task anchor or retained tail."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ContextBudgetExceeded,
    OVContext,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import (
    ContextSnapshot,
    MemoryReaderConfig,
    OVContextReader,
)
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter

from .test_revised_design import SDK, runtime


class RefreshSDK(SDK):
    def __init__(self):
        super().__init__()
        self.keeps = []
        self.summary = "s" * 2200
        self.next_message = 0
        self.transport_groups = []

    async def batch_add_messages(self, session_id, messages, *, identity):
        self.transport_groups.extend(deepcopy(messages))
        stored = []
        for spec in messages:
            parts = spec["parts"]
            if (
                spec["role"] == "user"
                and len(parts) > 1
                and all(p["type"] == "tool" for p in parts)
            ):
                stored.extend({**spec, "parts": [part]} for part in parts)
            else:
                stored.append(spec)
        result = await super().batch_add_messages(session_id, stored, identity=identity)
        for row in self.rows[-len(stored) :]:
            row["id"] = f"remote-{self.next_message}"
            self.next_message += 1
        return result

    async def commit_session(self, session_id, *, keep_recent_count, identity):
        self.keeps.append(keep_recent_count)
        result = await super().commit_session(
            session_id, keep_recent_count=keep_recent_count, identity=identity
        )
        result["archive_uri"] = f"viking://session/s/archive_{self.commits:03d}"
        result["task_id"] = f"task-{self.commits}"
        self.polls = 0
        return result

    async def get_task(self, task_id, *, identity):
        result = await super().get_task(task_id, identity=identity)
        if result["status"] == "completed":
            self.archive = f"archive_{self.commits:03d}"
            result["result"]["archive_id"] = self.archive
        return result

    async def get_session_context(self, session_id, token_budget=128000, *, identity):
        result = await super().get_session_context(session_id, token_budget, identity=identity)
        if self.archive:
            result["latest_archive_overview"] = self.summary
        return result


def middleware(sdk, keep=2):
    return OpenVikingContextMiddleware(
        client=sdk,
        message_writer=OVMessageWriter(sdk),
        context_reader=OVContextReader(sdk, MemoryReaderConfig(enabled=False)),
        commit_config=CommitConfig(1000000, 0.7, 100, keep),
        context_config=OVContextConfig(
            context_window=10000,
            output_reserve=1000,
            safety_margin=100,
            capacity_ratio=0.7,
            poll_interval=0.001,
            commit_wait_timeout=0.1,
        ),
        token_counter=lambda req: 100 + sum(len(m.content) for m in req.messages),
    )


def conversation(rounds=4, size=1000, parallel=1):
    messages = [HumanMessage(id="current", content="current task")]
    for index in range(rounds):
        calls = [
            {"name": "browser", "id": f"c-{index}-{n}", "args": {"step": index}}
            for n in range(parallel)
        ]
        messages.append(
            AIMessage(
                id=f"a-{index}",
                content="a" * size,
                tool_calls=calls,
                response_metadata={"source": index},
            )
        )
        messages.extend(
            ToolMessage(
                id=f"t-{index}-{n}",
                tool_call_id=call["id"],
                content="t" * size,
                artifact={"source": n},
            )
            for n, call in enumerate(reversed(calls))
        )
    return messages


@pytest.mark.parametrize("keep", [0, 1, 2])
async def test_capacity_archives_earlier_interactions_within_one_run(env, keep):
    sdk = RefreshSDK()
    mw = middleware(sdk, keep)
    metrics = []
    mw._ov_metrics.sink = lambda event, **data: metrics.append((event, data))
    state = {"messages": conversation()}
    state.update(await mw.abefore_agent(state, runtime()))
    context = mw._context(state, runtime())
    context.inject_recall("retained memory", "success")
    state["ov_context"] = context.to_state()
    before = deepcopy(state)
    requests = []

    async def handler(request):
        requests.append(request)
        return AIMessage(content="done")

    response = await mw.awrap_model_call(env[4].override(state=state, runtime=runtime()), handler)
    cached = response.command.update["ov_context"]
    assert sdk.keeps == [2] and sdk.commits == 1
    committed = [
        data
        for event, data in metrics
        if event == "context.commit" and data["status"] == "accepted"
    ]
    assert committed[0]["ov_keep_recent_count"] == 2
    assert committed[0]["archive_message_count"] == 7
    assert committed[0]["retained_native_message_count"] == 2
    assert mw._ov_session_engine.session_commit_policy.config.keep_recent_count == keep
    assert [m.id for m in cached["messages"]] == [
        "ov_summary:archive_001",
        "current",
        "a-3",
        "t-3-0",
    ]
    assert "retained memory" in cached["messages"][1].content
    assert cached["messages"][-2].response_metadata == {"source": 3}
    assert cached["messages"][-1].artifact == {"source": 0}
    assert cached["write_index"] == len(state["messages"])
    assert cached["recall_done"] is True and cached["recall_status"] == "success"
    assert cached["archive_id"] == "archive_001"
    assert len(requests) == 1 and requests[0].system_message is not None
    assert state == before
    assert "retained memory" not in str(sdk.rows)


async def test_parallel_tool_group_counts_persisted_records_and_preserves_results(env):
    sdk = RefreshSDK()
    mw = middleware(sdk, keep=1)
    state = {"messages": conversation(rounds=2, size=1500, parallel=3)}
    state.update(await mw.abefore_agent(state, runtime()))
    # One user, two assistant records and six separately persisted tool results.
    assert len(sdk.rows) == 9
    assert len(sdk.rows[-1]["parts"]) == 1
    assert len(sdk.transport_groups[-1]["parts"]) == 3
    engine = mw._ov_session_engine
    plan = engine.assembler.plan_refresh(mw._context(state, runtime()), keep_recent_count=1)
    assert plan.keep_recent_count == 4 and plan.archive_message_count == 5
    assert len(plan.recent_messages) == 4
    result = await mw.awrap_model_call(
        env[4].override(state=state, runtime=runtime()),
        AsyncMock(return_value=AIMessage(content="done")),
    )
    messages = result.command.update["ov_context"]["messages"]
    assert sdk.keeps == [4] and len(messages) == 6
    assert [m.tool_call_id for m in messages if isinstance(m, ToolMessage)] == [
        "c-1-2",
        "c-1-1",
        "c-1-0",
    ]
    assert not any(m.id.startswith(("a-0", "t-0")) for m in messages)


async def test_inflight_archive_keeps_later_successful_writer_increments(env):
    sdk = RefreshSDK()
    mw = middleware(sdk)
    state = {"messages": conversation(rounds=2, size=500)}
    state.update(await mw.abefore_agent(state, runtime()))
    engine = mw._ov_session_engine
    context = mw._context(state, runtime())
    identity = runtime().context.identity.sdk_identity
    assert await engine.commit(context, pending_tokens=10000, identity=identity, trigger="capacity")
    added = [
        AIMessage(
            id="later-ai",
            content="later",
            tool_calls=[{"name": "browser", "id": "later", "args": {}}],
        ),
        ToolMessage(
            id="later-result", tool_call_id="later", content="later result", artifact={"late": True}
        ),
    ]
    state["messages"].extend(added)
    await engine.write(context, state["messages"], identity=identity)
    request = env[4].override(state=state, runtime=runtime())
    await engine.wait_summary(context, identity=identity, token_budget=2900, model_request=request)
    ids = [m.id for m in context.messages]
    assert ids == ["ov_summary:archive_001", "current", "a-1", "t-1-0", "later-ai", "later-result"]
    assert context.messages[-1].artifact == {"late": True}
    assert context.pending_commit is None and context.write_index == len(state["messages"])
    assert sdk.commits == 1


async def test_failed_handler_reuses_refresh_plan_until_graph_checkpoint_catches_up(env):
    sdk = RefreshSDK()
    mw = middleware(sdk)
    state = {"messages": conversation()}
    state.update(await mw.abefore_agent(state, runtime()))
    before = deepcopy(state)
    request = env[4].override(state=state, runtime=runtime())
    with pytest.raises(RuntimeError, match="model failed"):
        await mw.awrap_model_call(request, AsyncMock(side_effect=RuntimeError("model failed")))
    assert state == before
    result = await mw.awrap_model_call(request, AsyncMock(return_value=AIMessage(content="done")))
    cached = result.command.update["ov_context"]
    assert sdk.commits == 1
    assert cached["messages"][-1].id == "t-3-0"
    assert cached["messages"][-1].artifact == {"source": 0}
    state.update(result.command.update)
    mw._context(state, runtime())
    assert not mw._ov_session_engine._refresh_progress


async def test_over_budget_candidate_keeps_old_cache_and_pending_progress(env):
    sdk = RefreshSDK()
    sdk.summary = "s" * 10000
    mw = middleware(sdk)
    state = {"messages": conversation()}
    state.update(await mw.abefore_agent(state, runtime()))
    before = deepcopy(state)
    handler = AsyncMock(return_value=AIMessage(content="done"))
    with pytest.raises(ContextBudgetExceeded):
        await mw.awrap_model_call(env[4].override(state=state, runtime=runtime()), handler)
    handler.assert_not_awaited()
    assert state == before
    recovered = mw._context(state, runtime())
    assert recovered.archive_id is None
    assert recovered.pending_commit["task_id"] == "task-1"
    assert recovered.messages[-1].id == "t-3-0"
    assert sdk.commits == 1


async def test_base_goal_excess_does_not_start_an_extra_refresh(env):
    sdk = RefreshSDK()
    mw = middleware(sdk)
    state = {
        "messages": [
            HumanMessage(id="old", content="x" * 3500),
            HumanMessage(id="current", content="task"),
        ]
    }
    state.update(await mw.abefore_agent(state, runtime()))
    reads = len(sdk.budgets)
    await mw.awrap_model_call(
        env[4].override(state=state, runtime=runtime()),
        AsyncMock(return_value=AIMessage(content="done")),
    )
    assert len(sdk.budgets) == reads and sdk.commits == 0


def test_refresh_plan_and_candidate_are_isolated_and_keep_user_exactly_once():
    mw = middleware(RefreshSDK())
    context = OVContext(
        "s", messages=conversation(rounds=2), owner=("a", "u", "p", "s"), write_index=5
    )
    context.begin_run("r", context.messages[0])
    context.inject_recall("memory", "success")
    assembler = mw._ov_session_engine.assembler
    plan = assembler.plan_refresh(context, keep_recent_count=1)
    before = deepcopy(context.to_state())
    snapshot = ContextSnapshot(
        [HumanMessage(id="ov_summary:archive_001", content="summary")], "archive_001"
    )
    candidate = assembler.refresh(context, snapshot, plan=plan)
    assert context.to_state() == before
    assert candidate.archive_id == "archive_001"
    assert [m.id for m in candidate.messages] == [
        "ov_summary:archive_001",
        "current",
        "a-1",
        "t-1-0",
    ]
    candidate.messages[-1].content = "changed"
    assert (
        plan.recent_messages[-1].content != "changed" and context.messages[-1].content != "changed"
    )
    with pytest.raises(ContextReadError, match="no longer matches"):
        assembler.refresh(context, snapshot, plan=replace(plan, run_id="other"))
    with pytest.raises(ContextReadError, match="usable summary"):
        assembler.refresh(context, ContextSnapshot([], "archive_001"), plan=plan)


def test_current_user_stays_in_place_when_it_is_part_of_the_retained_range():
    mw = middleware(RefreshSDK())
    rows = [HumanMessage(id="old", content="old"), HumanMessage(id="current", content="task")]
    context = OVContext("s", messages=rows, owner=("a", "u", "p", "s"))
    context.begin_run("r", rows[-1])
    context.inject_recall("memory", "success")
    assembler = mw._ov_session_engine.assembler
    plan = assembler.plan_refresh(context, keep_recent_count=1)
    snapshot = ContextSnapshot([HumanMessage(id="ov:summary", content="summary")], "archive_001")
    candidate = assembler.refresh(context, snapshot, plan=plan)
    assert [m.id for m in candidate.messages] == ["ov:summary", "current"]
    assert "memory" in candidate.messages[-1].content


def test_repeated_refresh_counts_archived_user_only_as_a_model_view():
    mw = middleware(RefreshSDK())
    assembler = mw._ov_session_engine.assembler
    context = OVContext(
        "s", messages=conversation(rounds=3), owner=("a", "u", "p", "s"), write_index=7
    )
    context.begin_run("r", context.messages[0])
    plan = assembler.plan_refresh(context, keep_recent_count=2)
    snapshot = ContextSnapshot(
        [HumanMessage(id="ov:summary", content="first summary")], "archive_001"
    )
    current = assembler.refresh(context, snapshot, plan=plan)
    # Reconstruction from checkpoint keeps the distinction from active OV records.
    current = OVContext.from_state(current.to_state())
    without_new_interaction = assembler.plan_refresh(current, keep_recent_count=2)
    assert without_new_interaction.archive_message_count == 0
    assert without_new_interaction.keep_recent_count == 2
    current.append(
        [
            AIMessage(
                id="next-ai",
                content="next",
                tool_calls=[{"name": "browser", "id": "next", "args": {}}],
            ),
            ToolMessage(id="next-result", tool_call_id="next", content="result"),
        ]
    )
    current.write_index += 2
    next_plan = assembler.plan_refresh(current, keep_recent_count=2)
    assert next_plan.archive_message_count == 2 and next_plan.keep_recent_count == 2
    snapshot = ContextSnapshot(
        [HumanMessage(id="ov:summary", content="next summary")], "archive_002"
    )
    second = assembler.refresh(current, snapshot, plan=next_plan)
    assert [m.id for m in second.messages] == ["ov:summary", "current", "next-ai", "next-result"]
    second.end_run()
    ended_plan = assembler.plan_refresh(second, keep_recent_count=0)
    assert ended_plan.archive_message_count == 2 and ended_plan.keep_recent_count == 0


async def test_engine_refresh_builds_candidate_before_explicit_budgeted_publication(env):
    sdk = RefreshSDK()
    mw = middleware(sdk)
    state = {"messages": [HumanMessage(id="current", content="task")]}
    state.update(await mw.abefore_agent(state, runtime()))
    engine = mw._ov_session_engine
    context = mw._context(state, runtime())
    before = deepcopy(context.to_state())
    sdk.archive = "archive_001"
    candidate = await engine.refresh(
        context, identity=context.identity, expected_archive_id="archive_001", token_budget=2900
    )
    assert context.to_state() == before and candidate.archive_id == "archive_001"
    assert candidate.messages[-1].id == "current"
    request = env[4].override(state=state, runtime=runtime())
    engine.apply_context(context, candidate, model_request=request)
    assert context.archive_id == "archive_001" and context.context_status == "ready"


async def test_real_graph_compacts_twice_and_rebuilt_agent_reuses_checkpoint():
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    @tool
    def long_result(step: int) -> str:
        """Return a deterministic long tool result for this step."""
        return str(step) + "t" * 2000

    replies = [
        AIMessage(
            id=f"a-{n}",
            content="a" * 1000,
            tool_calls=[{"name": "long_result", "id": f"call-{n}", "args": {"step": n}}],
        )
        for n in range(4)
    ] + [AIMessage(id="done", content="done")]
    sdk = RefreshSDK()
    first = middleware(sdk)
    saver = InMemorySaver()
    graph = create_agent(
        Model(responses=replies),
        tools=[long_result],
        middleware=[first],
        checkpointer=saver,
        context_schema=SimpleNamespace,
    )
    config = {"configurable": {"thread_id": "s"}, "recursion_limit": 100}
    await graph.ainvoke(
        {"messages": [HumanMessage(id="current", content="task")]},
        config=config,
        context=runtime().context,
    )
    saved = (await graph.aget_state(config)).values
    assert sdk.commits == 2 and sdk.keeps == [2, 2]
    assert len(saved["messages"]) == 10
    assert saved["ov_context"]["write_index"] == 10
    assert saved["ov_context"]["archive_id"] == "archive_002"
    assert len(saved["ov_context"]["messages"]) < len(saved["messages"])
    assert sum(m.id == "current" for m in saved["ov_context"]["messages"]) == 1
    await first.aclose()

    second = middleware(sdk)
    rebuilt = create_agent(
        Model(responses=[AIMessage(id="next-answer", content="continued")]),
        tools=[long_result],
        middleware=[second],
        checkpointer=saver,
        context_schema=SimpleNamespace,
    )
    await rebuilt.ainvoke(
        {"messages": [HumanMessage(id="next-user", content="continue")]},
        config=config,
        context=runtime("r2").context,
    )
    restored = (await rebuilt.aget_state(config)).values
    assert len(restored["messages"]) == restored["ov_context"]["write_index"] == 12
    assert restored["ov_context"]["archive_id"] == "archive_002" and sdk.commits == 2
    assert restored["ov_context"]["messages"][-1].id == "next-answer"
    await second.aclose()
