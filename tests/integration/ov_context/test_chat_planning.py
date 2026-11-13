"""Planning switches affect real graph tools and persisted session configuration."""

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from examples.deepagents_ov_context.chat.config import load_todo_enabled
from examples.deepagents_ov_context.chat.server import ActiveRun
from examples.deepagents_ov_context.chat.session import SessionStore

from .chat_model_support import NoToolModel


@pytest.fixture(autouse=True)
def office_assets(monkeypatch):
    """Planning tests exercise graph tools; document renderers are validated separately."""
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )


@pytest.mark.parametrize("mode", ["baseline", "current", "ordinary", "experience", "context_only"])
@pytest.mark.parametrize("enabled", [True, False])
async def test_planning_tools_and_prompt_in_every_mode(chat: Any, mode: str, enabled: bool) -> None:
    client, rt = chat
    rt.settings = replace(rt.settings, office_skills=True)
    response = await client.post(
        "/api/sessions", json={"execution": {"mode": mode, "enable_todos": enabled}}
    )
    assert response.status_code == 201
    session = rt.store.get(response.json()["session_id"])
    assert session.config_snapshot.execution["profile"]["enable_todos"] is enabled
    prompt = session.config_snapshot.execution["system_prompt"]
    assert ("[任务计划]" in prompt) is enabled
    sdk = rt.sdk()
    graph, middleware = rt.session_graph(session, sdk, ActiveRun())
    try:
        assert ("write_todos" in graph.nodes["tools"].bound._tools_by_name) is enabled
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()


@pytest.mark.parametrize("enabled", [True, False])
async def test_actual_model_bound_tools_and_system_prompt(chat: Any, enabled: bool) -> None:
    client, rt = chat
    seen_tools = []
    seen_messages = []

    class Model(NoToolModel):
        def bind_tools(self, tools, **kwargs):
            seen_tools.extend(
                getattr(t, "name", None) or t.get("function", {}).get("name", t.get("name"))
                for t in tools
            )
            return self

        def _generate(self, messages, *args, **kwargs):
            seen_messages.extend(messages)
            return super()._generate(messages, *args, **kwargs)

    rt.model = Model()
    rt.settings = replace(rt.settings, office_skills=True)
    response = await client.post(
        "/api/sessions", json={"execution": {"mode": "baseline", "enable_todos": enabled}}
    )
    session = rt.store.get(response.json()["session_id"])
    sdk = rt.sdk()
    try:
        graph, middleware = rt.session_graph(session, sdk, ActiveRun())
        await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, session.graph_config)
        assert ("write_todos" in seen_tools) is enabled
        assert any("[任务计划]" in str(m.content) for m in seen_messages) is enabled
    finally:
        await sdk.close()


async def test_default_override_and_sqlite_snapshot_survive_rebuild(chat: Any) -> None:
    client, rt = chat
    rt.settings = replace(rt.settings, office_skills=True, enable_todos=False)
    response = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    sid = response.json()["session_id"]
    session = rt.store.get(sid)
    assert session.config_snapshot.execution["profile"]["enable_todos"] is False
    rt.settings = replace(rt.settings, enable_todos=True)
    store = SessionStore(rt.settings.data_dir / "sessions.sqlite3")
    restored = store.get(sid)
    sdk = rt.sdk()
    try:
        graph, _ = rt.session_graph(restored, sdk, ActiveRun())
        assert "write_todos" not in graph.nodes["tools"].bound._tools_by_name
    finally:
        store.db.close()
        await sdk.close()
    override = await client.post("/api/sessions", json={"execution": {"enable_todos": False}})
    assert override.status_code == 201
    defaults = (await client.get("/api/execution-profiles")).json()
    assert defaults["agent_defaults"]["enable_todos"] is True


async def test_old_snapshot_retains_enabled_planning(chat: Any) -> None:
    client, rt = chat
    rt.settings = replace(rt.settings, office_skills=True)
    response = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    session = rt.store.get(response.json()["session_id"])
    session.config_snapshot.execution["profile"].pop("enable_todos")
    rt.settings = replace(rt.settings, enable_todos=False)
    sdk = rt.sdk()
    try:
        graph, _ = rt.session_graph(session, sdk, ActiveRun())
        assert "write_todos" in graph.nodes["tools"].bound._tools_by_name
    finally:
        await sdk.close()


def test_environment_file_and_process_override(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAT_AGENT__ENABLE_TODOS", raising=False)
    assert load_todo_enabled() is True
    p = tmp_path / "chat.env"
    p.write_text("CHAT_AGENT__ENABLE_TODOS=false\n")
    assert load_todo_enabled(p) is False
    monkeypatch.setenv("CHAT_AGENT__ENABLE_TODOS", "true")
    assert load_todo_enabled(p) is True


async def test_api_rejects_non_boolean_planning_switch(chat: Any) -> None:
    client, _ = chat
    for value in ["false", 0, 1]:
        response = await client.post("/api/sessions", json={"execution": {"enable_todos": value}})
        assert response.status_code == 422
