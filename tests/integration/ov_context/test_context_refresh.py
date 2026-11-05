"""Pure assembly refresh tests; Reader/Commit and Hook orchestration are separate."""

from copy import deepcopy
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context import (
    ContextAssembler,
    ContextSnapshot,
    OVContext,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.errors import ContextReadError


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


def test_refresh_plan_and_candidate_are_isolated_and_keep_user_exactly_once():
    context = OVContext(
        "s", messages=conversation(rounds=2), owner=("a", "u", "p", "s"), write_index=5
    )
    context.begin_run("r", context.messages[0])
    context.inject_recall("memory", "success")
    assembler = ContextAssembler(OVContextConfig())
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
    rows = [HumanMessage(id="old", content="old"), HumanMessage(id="current", content="task")]
    context = OVContext("s", messages=rows, owner=("a", "u", "p", "s"))
    context.begin_run("r", rows[-1])
    context.inject_recall("memory", "success")
    assembler = ContextAssembler(OVContextConfig())
    plan = assembler.plan_refresh(context, keep_recent_count=1)
    snapshot = ContextSnapshot([HumanMessage(id="ov:summary", content="summary")], "archive_001")
    candidate = assembler.refresh(context, snapshot, plan=plan)
    assert [m.id for m in candidate.messages] == ["ov:summary", "current"]
    assert "memory" in candidate.messages[-1].content


def test_repeated_refresh_counts_archived_user_only_as_a_model_view():
    assembler = ContextAssembler(OVContextConfig())
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
