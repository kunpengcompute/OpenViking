"""Reuse only the current Run's graph when reading its final checkpoint view."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from examples.deepagents_ov_context.chat import server as chat_server
from openviking.integrations.langchain.ov_context import OVContext

from .test_chat_api import new_session, parse_events


def values(session, mode):
    messages = [HumanMessage(id="u1", content="hello"), AIMessage(id="a1", content="done")]
    result = {"messages": messages}
    if mode == "current":
        result["ov_context"] = OVContext(
            session_id=session.session_id,
            messages=messages,
            owner=session.identity("r").session_key,
            write_index=2,
        ).to_state()
    return result


@pytest.mark.parametrize("mode", ["baseline", "current"])
async def test_existing_graph_reads_same_view_without_rebuild_or_sdk(chat, monkeypatch, mode):
    client, rt = chat
    session = rt.store.get(await new_session(client, execution={"mode": mode}))
    graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(values=values(session, mode)))
    )
    sdk = SimpleNamespace(close=AsyncMock())
    middleware = SimpleNamespace(aclose=AsyncMock())
    build = Mock(return_value=(graph, middleware))
    create_sdk = Mock(return_value=sdk)
    monkeypatch.setattr(rt, "session_graph", build)
    monkeypatch.setattr(rt, "sdk", create_sdk)

    inspected = await rt.context(session)
    build.assert_called_once()
    middleware.aclose.assert_awaited_once()
    sdk.close.assert_awaited_once()
    build.reset_mock()
    create_sdk.reset_mock()
    reused = await rt.context(session, graph=graph)

    assert reused == inspected
    build.assert_not_called()
    create_sdk.assert_not_called()
    assert middleware.aclose.await_count == sdk.close.await_count == 1


async def test_produce_constructs_one_graph_and_closes_caller_resources_once(chat, monkeypatch):
    client, rt = chat
    sid = await new_session(client, execution={"mode": "baseline"})
    session = rt.store.get(sid)
    sdk = SimpleNamespace(close=AsyncMock())

    class Graph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"model": {"messages": [AIMessage(content="done")]}}

        async def aget_state(self, config):
            assert config == session.graph_config
            return SimpleNamespace(values=values(session, "baseline"))

    build = Mock(return_value=(Graph(), None))
    monkeypatch.setattr(rt, "graph", build)
    monkeypatch.setattr(rt, "sdk", lambda session=None: sdk)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done"
    assert events[-1][1]["context"]["messages"][-1]["content"] == "done"
    build.assert_called_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("mode", ["baseline", "current"])
async def test_each_run_rebuilds_graph_for_same_session_and_shares_only_saver(
    chat, monkeypatch, mode
):
    client, rt = chat
    sid = await new_session(client, execution={"mode": mode})
    session = rt.store.get(sid)
    created = []

    class Graph:
        def __init__(self, settings):
            self.settings = settings
            self.run_ids = []
            self.inspections = 0

        async def astream(self, inputs, *, context, **kwargs):
            self.run_ids.append(context.identity.run_id)
            yield "updates", {"model": {"messages": [AIMessage(content="done")]}}

        async def aget_state(self, config):
            assert config == session.graph_config
            self.inspections += 1
            return SimpleNamespace(values=values(session, mode))

    def create(**settings):
        graph = Graph(settings)
        created.append(graph)
        return graph

    monkeypatch.setattr(chat_server, "create_deep_agent", create)
    run_ids = []
    for prompt in ("first turn", "next turn"):
        response = await client.post("/api/chat", json={"session_id": sid, "message": prompt})
        events = parse_events(response.text)
        assert events[-1][0] == "done"
        run_ids.append(next(data["run_id"] for event, data in events if event == "start"))

    assert len(created) == 2 and created[0] is not created[1]
    assert run_ids[0] != run_ids[1]
    for graph, run_id in zip(created, run_ids, strict=True):
        assert graph.run_ids == [run_id]
        assert graph.inspections == 1
        assert graph.settings["checkpointer"] is rt.saver
    if mode == "current":
        first = next(
            m for m in created[0].settings["middleware"] if hasattr(m, "_ov_session_engine")
        )
        second = next(
            m for m in created[1].settings["middleware"] if hasattr(m, "_ov_session_engine")
        )
        assert first is not second


async def test_failed_existing_graph_read_preserves_fail_open_and_caller_ownership(
    chat, monkeypatch
):
    client, rt = chat
    session = rt.store.get(await new_session(client))
    graph = SimpleNamespace(aget_state=AsyncMock(side_effect=RuntimeError("read failed")))
    create_sdk = Mock(side_effect=AssertionError("No remote fallback on checkpoint read failure"))
    monkeypatch.setattr(rt, "sdk", create_sdk)
    view = await rt.context(session, graph=graph)
    assert view.ov_degraded and view.ov_error_type == "RuntimeError"
    create_sdk.assert_not_called()


async def test_parallel_sessions_read_only_their_own_existing_graph(chat, monkeypatch):
    client, rt = chat
    sessions = [rt.store.get(await new_session(client, user_id=user)) for user in ("alice", "bob")]
    graphs = [
        SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values=values(session, "current")))
        )
        for session in sessions
    ]
    monkeypatch.setattr(rt, "sdk", Mock(side_effect=AssertionError("Unexpected SDK creation")))
    views = await asyncio.gather(
        *(rt.context(session, graph=graph) for session, graph in zip(sessions, graphs, strict=True))
    )
    for session, graph, view in zip(sessions, graphs, views, strict=True):
        assert view.owner == list(session.identity("r").session_key)
        graph.aget_state.assert_awaited_once_with(session.graph_config)


@pytest.mark.parametrize("mode", ["baseline", "current"])
async def test_existing_and_rebuilt_graph_restore_same_sqlite_checkpoint(chat, mode):
    client, rt = chat
    session = rt.store.get(await new_session(client, execution={"mode": mode}))
    sdk = rt.sdk()
    graph, middleware = rt.session_graph(
        session, sdk, SimpleNamespace(emit_metric=lambda *a, **k: None)
    )
    try:
        await graph.aupdate_state(
            session.graph_config,
            values(session, mode),
            as_node="OpenVikingContextMiddleware.after_agent" if mode == "current" else "model",
        )
        reused = await rt.context(session, graph=graph)
        inspected = await rt.context(session)
        assert reused == inspected
        assert [message["id"] for message in reused.messages] == ["u1", "a1"]
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()
