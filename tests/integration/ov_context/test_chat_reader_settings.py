"""Environment defaults and saved page overrides survive per-Run Agent rebuilds."""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from examples.deepagents_ov_context.chat.config import load_reader_config
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile
from examples.deepagents_ov_context.chat.server import ActiveRun, ChatRuntime
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig

from .chat_model_support import NoToolModel


@pytest.fixture(autouse=True)
def clean_reader_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep .env precedence checks independent of imported deployment defaults."""
    for name in os.environ:
        if name == "OV_READER" or name.startswith("OV_READER__"):
            monkeypatch.delenv(name)


def test_reader_env_file_and_process_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OV_READER__CANDIDATE_LIMIT=30\nOV_READER__RECALL_LIMIT=3\nOV_READER__MAX_INJECTED_CHARS=900\nOV_READER__TARGET_URIS=["viking://user/u/memories"]\nOV_READER__EXPERIENCE_ENABLED=true\n'
    )
    monkeypatch.setenv("OV_READER__RECALL_LIMIT", "2")
    config = load_reader_config(env_file)
    assert config.candidate_limit == 30
    assert config.recall_limit == 2 and config.max_injected_chars == 900
    assert config.target_uris == ("viking://user/u/memories",)
    assert config.experience_enabled and config.recall_timeout == 30
    assert config == load_reader_config(env_file)


def test_context_defaults_and_explicit_null_targets(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        'OV_READER__TARGET_URIS=null\nOV_READER__RECALL_QUOTAS={"resources":1,"skills":1}\n'
        'OV_READER__RECALL_DETAIL={"resources":"overview","memories":"full"}\n'
    )
    config = load_reader_config(path)
    assert config.search_mode == "context" and config.target_uris is None
    assert config.recall_quotas == {"resources": 1, "skills": 1}
    assert config.recall_detail == {"resources": "overview", "memories": "full"}
    path.write_text("OV_READER__SEARCH_MODE=find\nOV_READER__RECALL_DETAIL=full\n")
    legacy = load_reader_config(path)
    assert legacy.search_mode == "find" and legacy.recall_detail == "full"
    assert legacy.candidate_limit == 25 and legacy.recall_limit == 5
    assert legacy.ranking == "score" and not legacy.prefer_leaf
    assert config.candidate_limit == 25 and config.ranking == "openclaw" and config.prefer_leaf


def test_chat_example_and_reader_defaults_use_25_candidates() -> None:
    from examples.deepagents_ov_context.chat.config import DEFAULT_ENV_FILE

    assert MemoryReaderConfig().candidate_limit == 25
    assert load_reader_config(DEFAULT_ENV_FILE).candidate_limit == 25


def test_find_defaults_preserve_explicit_environment_overrides(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "OV_READER__SEARCH_MODE=find\nOV_READER__CANDIDATE_LIMIT=13\n"
        "OV_READER__RECALL_LIMIT=3\nOV_READER__RANKING=openclaw\nOV_READER__PREFER_LEAF=true\n"
    )
    config = load_reader_config(path)
    assert config.candidate_limit == 13 and config.recall_limit == 3
    assert config.ranking == "openclaw" and config.prefer_leaf


@pytest.mark.parametrize("explicit_user_only", [False, True])
async def test_session_snapshot_saves_effective_user_and_peer_roots(chat, explicit_user_only):
    client, runtime = chat
    execution = {"mode": "current"}
    if explicit_user_only:
        execution["target_uris"] = ["viking://user/memories"]
    response = await client.post(
        "/api/sessions", json={"agent_id": "doc-agent", "execution": execution}
    )
    assert response.status_code == 201, response.text
    session = response.json()
    roots = session["config_snapshot"]["reader_config"]["target_uris"]
    expected = ["viking://user/memories"]
    if not explicit_user_only:
        expected.append("viking://user/peers/doc-agent/memories")
    assert roots == expected
    saved = runtime.store.get(session["session_id"])
    assert list(saved.config_snapshot.reader_config.target_uris) == expected


@pytest.mark.parametrize(
    "setting",
    [
        "RECALL_LIMIT=-1",
        "SCORE_THRESHOLD=nan",
        "RECALL_TIMEOUT=0",
        "TARGET_URIS=not-json",
        "RECALL_QUOTAS=not-json",
        'RECALL_QUOTAS={"memories":1}',
        'RECALL_QUOTAS={"events":201}',
        'RECALL_QUOTAS={"events":true}',
        'RECALL_QUOTAS={"events":"1"}',
    ],
)
def test_invalid_environment_fails_before_graph_creation(tmp_path: Path, setting: str) -> None:
    path = tmp_path / ".env"
    path.write_text(f"OV_READER__{setting}\n")
    with pytest.raises(ValueError):
        load_reader_config(path)


async def test_saved_page_overrides_are_reapplied_after_agent_and_runtime_rebuild(
    chat: Any,
) -> None:
    client, first = chat
    first.model = NoToolModel()
    defaults = MemoryReaderConfig(recall_limit=2, max_injected_chars=300, recall_timeout=7)
    first.settings = replace(first.settings, reader_config=defaults)
    created = await client.post(
        "/api/sessions",
        json={
            "execution": {
                "mode": "ordinary",
                "memory_budget_chars": 1234,
                "recall_timeout": 9,
                "target_uris": ["viking://user/saved/memories"],
                "max_tokens": 512,
            }
        },
    )
    assert created.status_code == 201
    sid = created.json()["session_id"]
    rebuilt = ChatRuntime(first.settings, first.model_config, checkpointer=first.saver)
    rebuilt.model = NoToolModel()
    try:
        for runtime in (first, rebuilt):
            session = runtime.store.get(sid)
            configs = []
            for _ in range(2):
                sdk = runtime.sdk()
                _, middleware = runtime.session_graph(session, sdk, ActiveRun())
                assert middleware is not None
                config = middleware._ov_session_engine.context_reader.config
                configs.append(config)
                assert config.recall_limit == 2
                assert config.max_injected_chars == 1234 and config.recall_timeout == 9
                assert config.target_uris == ("viking://user/saved/memories",)
                assert middleware._ov_session_engine.config.output_reserve == 512
                await middleware.aclose()
                await sdk.close()
            assert configs[0] == configs[1]
            assert session.execution.memory_budget_chars == 1234
    finally:
        await rebuilt.close()


def test_omitted_page_settings_inherit_environment_and_modes_remain_isolated() -> None:
    defaults = MemoryReaderConfig(
        recall_limit=4,
        max_injected_chars=777,
        experience_enabled=True,
        target_uris=("viking://user/env/memories",),
    )
    assert ExecutionProfile().memory_config(defaults) == defaults
    ordinary = ExecutionProfile(mode="ordinary").memory_config(defaults)
    assert ordinary.enabled and not ordinary.experience_enabled
    assert ordinary.max_injected_chars == 777 and ordinary.recall_limit == 4
    assert ordinary.target_uris == defaults.target_uris
    context_only = ExecutionProfile(mode="context_only").memory_config(defaults)
    assert not context_only.enabled and not context_only.experience_enabled


def test_page_preserves_explicit_settings_and_omits_inherited_values() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the page JavaScript contract")
    page = Path(__file__).parents[3] / "examples/deepagents_ov_context/chat/static/index.html"
    text = page.read_text()
    function = text.split("    function executionProfile()", 1)[1].split(
        "    async function createSession", 1
    )[0]
    script = (
        """
const fields = Object.fromEntries([
  'execution-mode','ordinary-uris','experience-uris','memory-budget',
  'profile-context-window','profile-base-ratio','profile-capacity-ratio','profile-max-tokens','profile-recursion','profile-timeout','profile-thinking','profile-todos','profile-prompt',
  'commit-token-budget','commit-ratio','commit-run-threshold','commit-keep-recent','commit-settings'
].map(name => [name, {value:''}]));
const $ = name => fields[name];
fields['execution-mode'].value='current';
"""
        + "function executionProfile()"
        + function
        + """
const inherited=executionProfile();
fields['ordinary-uris'].value='viking://user/page/memories';
fields['memory-budget'].value='1234';
fields['profile-max-tokens'].value='512';
fields['profile-thinking'].value='enabled';
fields['profile-todos'].value='disabled';
fields['profile-prompt'].value='saved instruction';
console.log(JSON.stringify({inherited, explicit:executionProfile()}));
"""
    )
    result = subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    values = json.loads(result.stdout)
    assert values["inherited"] == {"mode": "current"}
    assert values["explicit"] == {
        "mode": "current",
        "target_uris": ["viking://user/page/memories"],
        "memory_budget_chars": 1234,
        "max_tokens": 512,
        "thinking": True,
        "enable_todos": False,
        "system_prompt": "saved instruction",
    }
