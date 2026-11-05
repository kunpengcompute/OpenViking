"""Session data survives graph reconstruction and isolates independent threads."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from openviking.integrations.langchain.ov_context import OVContext, OVPluginState, RunIdentity


def graph(saver):
    builder = StateGraph(OVPluginState)
    builder.add_node("keep", lambda state: {})
    builder.add_edge(START, "keep")
    builder.add_edge("keep", END)
    return builder.compile(checkpointer=saver)


async def test_checkpoint_restores_native_messages_and_progress_after_graph_rebuild():
    saver = InMemorySaver()
    user = HumanMessage(id="u", content="task", additional_kwargs={"source": "app"})
    rows = [
        user,
        AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "lookup", "args": {}}]),
        ToolMessage(id="t", tool_call_id="c", content="result", artifact={"raw": True}),
    ]
    context = OVContext(
        "s",
        messages=rows,
        owner=("acct", "user", "peer", "s"),
        write_index=3,
        archive_id="archive_001",
        completed_runs=2,
        context_status="ready",
        pending_commit={"task_id": "job", "archive_id": "archive_001"},
    )
    context.begin_run("r", user)
    config = {"configurable": {"thread_id": "acct-user-peer-s"}}
    await graph(saver).ainvoke({"messages": rows, "ov_context": context.to_state()}, config)
    restored = OVContext.from_state((await graph(saver).aget_state(config)).values["ov_context"])
    assert restored.to_state() == context.to_state()
    assert isinstance(restored.messages[1], AIMessage)
    assert isinstance(restored.messages[2], ToolMessage)
    assert restored.messages[2].artifact == {"raw": True}
    restored.messages.append(HumanMessage(id="next", content="new"))
    assert len((await graph(saver).aget_state(config)).values["ov_context"]["messages"]) == 3


async def test_four_threads_keep_users_and_agents_separate():
    saver = InMemorySaver()
    for user in ("alice", "bob"):
        for peer in ("mail", "doc"):
            sid = f"{user}-{peer}"
            identity = RunIdentity("acct", user, peer, sid, "r")
            context = OVContext(
                sid, messages=[HumanMessage(content=sid)], owner=identity.session_key
            )
            await graph(saver).ainvoke(
                {"messages": [], "ov_context": context.to_state()},
                {"configurable": {"thread_id": sid}},
            )
    for user in ("alice", "bob"):
        for peer in ("mail", "doc"):
            sid = f"{user}-{peer}"
            saved = (await graph(saver).aget_state({"configurable": {"thread_id": sid}})).values
            context = OVContext.from_state(saved["ov_context"])
            assert context.owner == ("acct", user, peer, sid)
            assert [m.content for m in context.messages] == [sid]


def test_new_run_cleans_recall_and_keeps_session_state():
    first = HumanMessage(id="u1", content="first")
    context = OVContext("s", messages=[first], write_index=1, archive_id="archive_001")
    context.begin_run("r1", first)
    context.inject_recall("remember", "success")
    assert "remember" in context.messages[0].content
    second = HumanMessage(id="u2", content="second")
    context.append([second])
    context.begin_run("r2", second)
    assert context.messages[0] == first
    assert (context.run_id, context.recall_done, context.original_user) == ("r2", False, second)
    assert (context.archive_id, context.write_index) == ("archive_001", 1)


@pytest.mark.parametrize(
    "field", ["account_id", "user_id", "actor_peer_id", "session_id", "run_id"]
)
@pytest.mark.parametrize("bad", ["", " ", "bad/path", "bad\nheader"])
def test_identity_rejects_invalid_fields(field, bad):
    values = {
        "account_id": "a",
        "user_id": "u",
        "actor_peer_id": "p",
        "session_id": "s",
        "run_id": "r",
    }
    with pytest.raises(ValueError):
        RunIdentity(**{**values, field: bad})


def test_identity_and_state_mapping_are_independent_of_mutable_sdk_defaults():
    identity = RunIdentity("a", "u", "p", "s", "r")
    with pytest.raises(FrozenInstanceError):
        identity.user_id = "other"
    context = OVContext("s", owner=identity.session_key, messages=[HumanMessage(content="task")])
    restored = OVContext.from_state(deepcopy(context.to_state()))
    assert restored.identity == identity.sdk_identity
    restored.messages.append(HumanMessage(content="extra"))
    assert len(context.messages) == 1
