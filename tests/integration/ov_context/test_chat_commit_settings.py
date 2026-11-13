"""Commit settings come from .env, with durable per-session page overrides."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from examples.deepagents_ov_context.chat.config import commit_config_values, load_commit_defaults
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile
from examples.deepagents_ov_context.chat.server import ActiveRun, ChatRuntime
from examples.deepagents_ov_context.chat.session import ContextView
from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context.commit import CommitConfig

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events


@pytest.fixture(autouse=True)
def clean_commit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in os.environ:
        if name == "OV_COMMIT" or name.startswith("OV_COMMIT__"):
            monkeypatch.delenv(name)


def env_text() -> str:
    return (
        "OV_COMMIT__TOKEN_BUDGET=9000\n"
        "OV_COMMIT__COMMIT_TOKEN_THRESHOLD_RATIO=0.4\n"
        "OV_COMMIT__COMPLETED_RUN_THRESHOLD=6\n"
        "OV_COMMIT__KEEP_RECENT_COUNT=2\n"
    )


def test_env_file_and_process_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text(env_text())
    monkeypatch.setenv("OV_COMMIT__TOKEN_BUDGET", "12000")
    monkeypatch.setenv("OV_COMMIT__KEEP_RECENT_COUNT", "0")
    actual = load_commit_defaults(path)
    assert actual == CommitConfig(12000, Decimal("0.4"), 6, 0)
    assert commit_config_values(actual) == {
        "token_budget": 12000,
        "commit_token_threshold_ratio": 0.4,
        "completed_run_threshold": 6,
        "keep_recent_count": 0,
    }


@pytest.mark.parametrize("missing", range(4))
def test_each_environment_field_is_required(tmp_path: Path, missing: int) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(line for i, line in enumerate(env_text().splitlines()) if i != missing)
    )
    with pytest.raises(ValueError):
        load_commit_defaults(path)


def test_no_implicit_commit_defaults() -> None:
    with pytest.raises(ValueError):
        load_commit_defaults()
    middleware = OpenVikingContextMiddleware()
    assert middleware._ov_session_engine.session_commit_policy is None


@pytest.mark.parametrize(
    "setting",
    [
        "TOKEN_BUDGET=0",
        "COMMIT_TOKEN_THRESHOLD_RATIO=0",
        "COMMIT_TOKEN_THRESHOLD_RATIO=0.8",
        "COMMIT_TOKEN_THRESHOLD_RATIO=nan",
        "COMMIT_TOKEN_THRESHOLD_RATIO=inf",
        "COMPLETED_RUN_THRESHOLD=0",
        "KEEP_RECENT_COUNT=-1",
        "KEEP_RECENT_COUNT=10001",
    ],
)
def test_invalid_environment_rejected(tmp_path: Path, setting: str) -> None:
    path = tmp_path / ".env"
    name = setting.split("=", 1)[0]
    path.write_text(
        "\n".join(
            line
            for line in env_text().splitlines()
            if not line.startswith("OV_COMMIT__" + name + "=")
        )
        + "\nOV_COMMIT__"
        + setting
        + "\n"
    )
    with pytest.raises(ValueError):
        load_commit_defaults(path)


@pytest.mark.parametrize(
    "override",
    [
        {"token_budget": 0},
        {"token_budget": True},
        {"completed_run_threshold": 0},
        {"keep_recent_count": -1},
        {"keep_recent_count": 1.5},
        {"commit_token_threshold_ratio": 0.8},
        {"unknown": 1},
    ],
)
async def test_invalid_page_commit_overrides_are_http_422(
    chat: Any, override: dict[str, Any]
) -> None:
    client, _ = chat
    response = await client.post("/api/sessions", json={"execution": {"commit": override}})
    assert response.status_code == 422


async def test_saved_overrides_survive_agent_and_service_rebuild(chat: Any) -> None:
    client, first = chat
    first.model = NoToolModel()
    first.settings = replace(first.settings, commit_config=CommitConfig(9000, 0.4, 6, 2))
    response = await client.post(
        "/api/sessions",
        json={
            "execution": {"commit": {"commit_token_threshold_ratio": 0.3, "keep_recent_count": 0}}
        },
    )
    assert response.status_code == 201
    sid = response.json()["session_id"]
    restarted_settings = replace(first.settings, commit_config=CommitConfig(12000, 0.5, 9, 5))
    rebuilt = ChatRuntime(restarted_settings, first.model_config, checkpointer=first.saver)
    rebuilt.model = NoToolModel()
    try:
        for runtime, expected in (
            (first, CommitConfig(9000, 0.3, 6, 0)),
            (rebuilt, CommitConfig(9000, 0.3, 6, 0)),
        ):
            session = runtime.store.get(sid)
            for _ in range(2):
                sdk = runtime.sdk()
                _, middleware = runtime.session_graph(session, sdk, ActiveRun())
                assert middleware is not None
                try:
                    assert middleware._ov_session_engine.session_commit_policy.config == expected
                finally:
                    await middleware.aclose()
                    await sdk.close()
            assert session.execution.commit.keep_recent_count == 0
    finally:
        await rebuilt.close()


async def test_effective_commit_values_are_json_safe_and_recorded_in_chat_events(
    chat: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, rt = chat
    defaults = (await client.get("/api/execution-profiles")).json()["commit_defaults"]
    assert defaults == commit_config_values(rt.settings.commit_config)
    created = await client.post(
        "/api/sessions",
        json={
            "execution": {"commit": {"commit_token_threshold_ratio": 0.4, "keep_recent_count": 0}}
        },
    )
    sid = created.json()["session_id"]
    sdk = SimpleNamespace(initialize=AsyncMock(), close=AsyncMock())

    class Graph:
        async def astream(self, inputs: Any, config: Any, **kwargs: Any) -> Any:
            yield "updates", {"model": {"messages": [AIMessage(content="done")]}}

    async def context(session: Any, *, graph: Any = None) -> ContextView:
        return ContextView(source="checkpoint", messages=[], write_index=0)

    monkeypatch.setattr(rt, "sdk", lambda session=None: sdk)
    monkeypatch.setattr(rt, "graph", lambda *args, **kwargs: (Graph(), None))
    monkeypatch.setattr(rt, "context", context)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", events[-1]
    expected = {**defaults, "commit_token_threshold_ratio": 0.4, "keep_recent_count": 0}
    assert events[0][1]["commit_config"] == expected
    rid = events[0][1]["run_id"]
    saved = (await client.get(f"/api/sessions/{sid}/runs/{rid}")).json()
    assert saved["events"][0]["data"]["commit_config"] == expected


@pytest.mark.parametrize("mode", ["context_only", "ordinary", "experience"])
async def test_comparison_profiles_do_not_enable_automatic_commit(chat: Any, mode: str) -> None:
    _, rt = chat
    from examples.deepagents_ov_context.chat.session import SessionInfo

    session = SessionInfo(
        session_id="comparison",
        account_id="a",
        user_id="u",
        agent_id="p",
        created_at=0,
        execution=ExecutionProfile(mode=mode, commit={"keep_recent_count": 0}),
    )
    session.config_snapshot = rt.snapshot(session)
    sdk = rt.sdk()
    _, middleware = rt.session_graph(session, sdk, ActiveRun())
    assert middleware is not None
    try:
        assert middleware._ov_session_engine.session_commit_policy is None
    finally:
        await middleware.aclose()
        await sdk.close()


def test_page_submits_restores_and_clears_optional_commit_fields() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the page JavaScript contract")
    text = (
        Path(__file__).parents[3] / "examples/deepagents_ov_context/chat/static/index.html"
    ).read_text()
    functions = text.split("    function executionProfile()", 1)[1].split(
        "    async function createSession", 1
    )[0]
    script = (
        """
const fields = Object.fromEntries([
'execution-mode','ordinary-uris','experience-uris','memory-budget',
'profile-context-window','profile-base-ratio','profile-capacity-ratio','profile-max-tokens','profile-recursion','profile-timeout','profile-thinking','profile-todos','profile-prompt',
'commit-token-budget','commit-ratio','commit-run-threshold','commit-keep-recent','commit-settings'
].map(name=>[name,{value:''}]));
const $ = name=>fields[name];
fields['execution-mode'].value='current';
"""
        + "function executionProfile()"
        + functions
        + """
const inherited=executionProfile();
restoreCommitSettings({commit:{token_budget:6000,commit_token_threshold_ratio:0.25,completed_run_threshold:2,keep_recent_count:0}});
const explicit=executionProfile();
fields['execution-mode'].value='ordinary';syncCommitControls();
const comparison=executionProfile(), disabled=fields['commit-settings'].disabled;
fields['execution-mode'].value='current';restoreCommitSettings({});
const cleared=executionProfile();
console.log(JSON.stringify({inherited,explicit,comparison,disabled,cleared}));
"""
    )
    output = subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    values = json.loads(output.stdout)
    assert values["inherited"] == values["cleared"] == {"mode": "current"}
    assert values["explicit"]["commit"] == {
        "token_budget": 6000,
        "commit_token_threshold_ratio": 0.25,
        "completed_run_threshold": 2,
        "keep_recent_count": 0,
    }
    assert "commit" not in values["comparison"] and values["disabled"]


def test_init_config_adds_required_fields_without_overwriting_existing_values(
    tmp_path: Path,
) -> None:
    import os

    directory = tmp_path / "deploy"
    directory.mkdir()
    path = directory / ".env"
    path.write_text("export OV_COMMIT__TOKEN_BUDGET = 9000\nOV_READER__RECALL_LIMIT=2")
    script = Path(__file__).parents[3] / "examples/deepagents_ov_context/scripts/deploy.sh"
    environment = {**os.environ, "OV_CHAT_HOME": str(directory)}
    subprocess.run(
        ["bash", str(script), "init-config"], env=environment, check=True, capture_output=True
    )
    first = path.read_bytes()
    config = load_commit_defaults(path)
    assert config.token_budget == 9000
    assert config.completed_run_threshold > 0
    assert "OV_READER__RECALL_LIMIT=2" in path.read_text()
    subprocess.run(
        ["bash", str(script), "init-config"], env=environment, check=True, capture_output=True
    )
    assert path.read_bytes() == first
    assert path.stat().st_mode & 0o777 == 0o600
