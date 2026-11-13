"""Page and batch clients exercise the same Chat routes, graph and SQLite state."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from benchmark.office.suites.common.chat import chat_events
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events


async def test_browser_only_profile_survives_graph_rebuild(chat: Any, monkeypatch: Any) -> None:
    from dataclasses import replace

    client, runtime = chat
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )
    runtime.settings = replace(runtime.settings, office_skills=True)
    bound: list[set[str]] = []

    class BrowserModel(NoToolModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> NoToolModel:
            names = {tool.name for tool in tools}
            bound.append(names)
            assert "execute" not in names
            assert {"read_file", "write_todos"} <= names
            return self

    runtime.model = BrowserModel()
    response = await client.post(
        "/api/sessions",
        json={"execution": {"mode": "baseline", "browser_only": True}},
    )
    assert response.status_code == 201
    sid = response.json()["session_id"]
    for _ in range(2):
        response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
        assert parse_events(response.text)[-1][0] == "done"
    assert len(bound) >= 2
    saved = runtime.store.get(sid)
    assert saved.execution.browser_only
    assert saved.config_snapshot.execution["profile"]["browser_only"]


@pytest.mark.parametrize("mode", ["context_only", "ordinary", "experience"])
async def test_native_profiles_run_through_chat_and_export_state(
    chat: Any, env: Any, mode: str
) -> None:
    client, runtime = chat
    server = env[0]
    await runtime.ov_http.aclose()

    async def current_contract(request: httpx.Request) -> httpx.Response:
        response = await server.handle(request)
        if response.status_code == 404:
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "error": {"code": "NOT_FOUND", "message": "missing"},
                },
            )
        if request.url.path.endswith("/context"):
            body = response.json()
            body["result"]["latest_finished_archive_id"] = server.archive_id
            return httpx.Response(200, json=body)
        return response

    runtime.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(current_contract))
    runtime.model = NoToolModel()
    created = await client.post("/api/sessions", json={"execution": {"mode": mode}})
    assert created.status_code == 201
    sid = created.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", events[-1]
    state = (await client.get(f"/api/sessions/{sid}/state")).json()
    assert state["ov_context"]["write_index"] == len(state["messages"]) == 2
    assert state["ov_context"]["run_id"] is None
    runs = (await client.get(f"/api/sessions/{sid}/runs")).json()
    assert runs[0]["statistics"] == events[-1][1]["statistics"]
    assert runtime.store.get(sid).execution.mode == mode
    assert server.counts["batch"] > 0


async def test_session_execution_settings_are_validated_and_immutable(chat: Any) -> None:
    client, runtime = chat
    for execution in (
        {"mode": "unknown"},
        {"memory_budget_chars": 0},
        {"target_uris": ["https://not-a-viking-target"]},
        {"context_window": 1024, "max_tokens": 2048},
        {"context_window": 1024},
    ):
        assert (
            await client.post("/api/sessions", json={"execution": execution})
        ).status_code == 422
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    session = runtime.store.get(sid)
    with pytest.raises(Exception, match="会话中途"):
        runtime.store.save(
            session.model_copy(update={"execution": ExecutionProfile(mode="ordinary")})
        )
    assert runtime.store.get(sid).execution.mode == "baseline"


async def test_sse_parser_keeps_multibyte_text_and_rejects_missing_terminal() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, text=': keepalive\n\nevent: done\ndata: {"answer":"中文"}\n\n'
            )
        )
    ) as client:
        async with client.stream("GET", "http://localhost/") as response:
            assert (await chat_events(response))[-1]["data"]["answer"] == "中文"
    with pytest.raises(RuntimeError, match="terminal"):
        await chat_events(
            httpx.Response(
                200,
                request=httpx.Request("GET", "http://localhost/"),
                text="event: token\ndata: {}\n\n",
            )
        )


async def test_profile_timeout_is_persisted_as_error(chat: Any) -> None:
    client, runtime = chat

    class WaitingModel(NoToolModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    runtime.model = WaitingModel()
    created = await client.post(
        "/api/sessions", json={"execution": {"mode": "baseline", "timeout_seconds": 0.05}}
    )
    sid = created.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "wait"})
    events = parse_events(response.text)
    assert events[-1][0] == "error" and events[-1][1]["error_type"] == "TimeoutError"
    assert runtime.store.get(sid).last_run.status == "error"
    assert not runtime.active


async def test_download_exports_are_session_run_scoped(chat: Any, tmp_path: Path) -> None:
    client, runtime = chat
    runtime.model = NoToolModel()
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    run_id = parse_events(response.text)[0][1]["run_id"]
    folder = runtime.settings.data_dir / "downloads" / sid / run_id
    folder.mkdir(parents=True)
    pdf = b"synthetic-pdf-for-http-integrity-check"
    (folder / "case.pdf").write_bytes(pdf)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    (folder / "link.pdf").symlink_to(outside)
    prefix = f"/api/sessions/{sid}/runs/{run_id}/downloads"
    listing = (await client.get(prefix)).json()
    assert listing == [
        {"filename": "case.pdf", "bytes": len(pdf), "sha256": hashlib.sha256(pdf).hexdigest()}
    ]
    assert (await client.get(prefix + "/case.pdf")).content == pdf
    assert (await client.get(prefix + "/link.pdf")).status_code == 404
    assert (await client.get(f"/api/sessions/{sid}/runs/unknown/downloads")).status_code == 404


def test_method_memory_controls_keep_reader_defaults_and_are_explicit():
    from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig

    defaults = MemoryReaderConfig(recall_max_tokens=1600, recall_detail=None)
    assert ExecutionProfile(mode="ordinary").memory_config(defaults).recall_detail is None
    profile = ExecutionProfile(
        mode="ordinary", memory_detail="full", memory_budget_tokens=3200, memory_budget_chars=8000
    )
    saved = ExecutionProfile.model_validate_json(profile.model_dump_json())
    cfg = saved.memory_config(defaults)
    assert (
        cfg.recall_detail == "full"
        and cfg.recall_max_tokens == 3200
        and cfg.max_injected_chars == 8000
    )
    assert cfg.experience_enabled is False
    with pytest.raises(ValueError):
        ExecutionProfile(memory_detail="invalid")


async def test_removed_official_profile_is_hidden_and_rejected_before_sdk_access(
    chat: Any, monkeypatch: Any
) -> None:
    client, runtime = chat
    initialize = AsyncMock()
    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.AsyncHTTPClient.initialize", initialize
    )
    profiles = (await client.get("/api/execution-profiles")).json()
    assert profiles["modes"] == ["current", "baseline", "context_only", "ordinary", "experience"]
    assert '<option value="official">' not in (await client.get("/")).text
    rejected = await client.post("/api/sessions", json={"execution": {"mode": "official"}})
    assert rejected.status_code == 422
    assert runtime.store.list() == []
    initialize.assert_not_awaited()


async def test_baseline_chat_never_accesses_ov(chat: Any, monkeypatch: Any) -> None:
    client, runtime = chat
    runtime.model = NoToolModel()
    initialize = AsyncMock(side_effect=AssertionError("Baseline contacted OV"))
    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.AsyncHTTPClient.initialize", initialize
    )
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    for message in ("整理会议安排", "继续"):
        response = await client.post("/api/chat", json={"session_id": sid, "message": message})
        assert parse_events(response.text)[-1][0] == "done"
    state = (await client.get(f"/api/sessions/{sid}/state")).json()
    assert len(state["messages"]) == 4 and "ov_context" not in state
    assert len((await client.get(f"/api/sessions/{sid}/runs")).json()) == 2
    assert (await client.post(f"/api/sessions/{sid}/restore")).status_code == 200
    assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 409
    initialize.assert_not_awaited()
