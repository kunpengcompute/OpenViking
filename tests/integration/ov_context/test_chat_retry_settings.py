"""Retry configuration follows environment precedence and Session snapshots."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from examples.deepagents_ov_context.chat import server as chat_server
from examples.deepagents_ov_context.chat.config import load_context_config
from examples.deepagents_ov_context.chat.server import ActiveRun, ChatRuntime


def retry_env(tmp_path: Path) -> Path:
    template = (Path(chat_server.__file__).parents[1] / "config/chat.example.env").read_text()
    path = tmp_path / ".env"
    path.write_text(
        template.replace("OV_CONTEXT__RETRY_ATTEMPTS=2", "OV_CONTEXT__RETRY_ATTEMPTS=7")
    )
    return path


@pytest.fixture(autouse=True)
def clean_retry_environment(monkeypatch):
    monkeypatch.delenv("OV_CONTEXT", raising=False)
    monkeypatch.delenv("OV_CONTEXT__RETRY_ATTEMPTS", raising=False)


@pytest.mark.parametrize(
    ("process_value", "cli_value", "expected"),
    [(None, None, 7), ("4", None, 4), ("4", "0", 0), ("4", "3", 3)],
)
def test_chat_cli_retry_precedence(tmp_path, monkeypatch, process_value, cli_value, expected):
    import uvicorn

    env_file = retry_env(tmp_path)
    if process_value is not None:
        monkeypatch.setenv("OV_CONTEXT__RETRY_ATTEMPTS", process_value)
    argv = [
        "chat_server",
        "--model-config",
        str(tmp_path / "model.json"),
        "--data-dir",
        str(tmp_path / "chat"),
        "--env-file",
        str(env_file),
    ]
    if cli_value is not None:
        argv.extend(["--retry-attempts", cli_value])
    captured = []
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(chat_server, "create_app", lambda settings: settings)
    monkeypatch.setattr(uvicorn, "run", lambda settings, **kwargs: captured.append(settings))
    chat_server.main()
    assert len(captured) == 1
    assert captured[0].context_config.retry_attempts == expected


@pytest.mark.parametrize("retries", [0, 1])
async def test_chat_reads_use_context_retry_configuration(chat, monkeypatch, retries):
    _, runtime = chat
    runtime.settings = replace(
        runtime.settings,
        context_config=replace(runtime.settings.context_config, retry_attempts=retries),
    )
    monkeypatch.setattr(
        "openviking.integrations.langchain.ov_context.retry.retry_delay", lambda attempt: 0
    )
    operation = AsyncMock(side_effect=[httpx.ConnectError("OV temporarily unavailable"), "ready"])
    if retries:
        assert await runtime.ov_read(operation) == "ready"
    else:
        with pytest.raises(httpx.ConnectError):
            await runtime.ov_read(operation)
    assert operation.await_count == retries + 1


@pytest.mark.parametrize("fail_open", [False, True])
async def test_retry_env_is_frozen_for_session_and_reused_after_rebuild(chat, tmp_path, fail_open):
    client, first = chat
    first.settings = replace(
        first.settings,
        context_config=load_context_config(retry_env(tmp_path)),
        ov_policy=first.settings.ov_policy.model_copy(update={"fail_open": fail_open}),
    )
    response = await client.post("/api/sessions", json={})
    assert response.status_code == 201
    sid = response.json()["session_id"]
    snapshot = first.store.get(sid).config_snapshot
    assert snapshot.context_config.retry_attempts == 7
    changed = replace(
        first.settings,
        context_config=replace(first.settings.context_config, retry_attempts=1),
    )
    rebuilt = ChatRuntime(changed, first.model_config, checkpointer=first.saver)
    try:
        session = rebuilt.store.get(sid)
        for _ in range(2):
            sdk = rebuilt.sdk()
            middleware = None
            try:
                _, middleware = rebuilt.session_graph(session, sdk, ActiveRun())
                assert middleware._ov_session_engine.config.retry_attempts == (
                    0 if fail_open else 7
                )
                assert session.config_snapshot == snapshot
            finally:
                if middleware is not None:
                    await middleware.aclose()
                await sdk.close()
        first.settings = changed
        fresh = await client.post("/api/sessions", json={})
        assert fresh.status_code == 201
        assert (
            first.store.get(
                fresh.json()["session_id"]
            ).config_snapshot.context_config.retry_attempts
            == 1
        )
    finally:
        await rebuilt.close()


@pytest.mark.parametrize("fail_open", [False, True])
async def test_archive_wait_default_reaches_rebuilt_engine(chat, monkeypatch, fail_open):
    from openviking.integrations.langchain.ov_context import OVContextConfig

    monkeypatch.delenv("OV_CONTEXT__COMMIT_WAIT_TIMEOUT", raising=False)
    client, runtime = chat
    config = load_context_config()
    assert config.commit_wait_timeout == OVContextConfig().commit_wait_timeout == 300
    runtime.settings = replace(
        runtime.settings,
        context_config=config,
        ov_policy=runtime.settings.ov_policy.model_copy(update={"fail_open": fail_open}),
    )
    created = await client.post("/api/sessions", json={})
    assert created.status_code == 201
    session = runtime.store.get(created.json()["session_id"])
    # Later process settings affect new Sessions, while rebuilding this graph
    # preserves the timeout saved when the Session was created.
    runtime.settings = replace(
        runtime.settings, context_config=replace(config, commit_wait_timeout=3)
    )
    sdk = runtime.sdk()
    middleware = None
    try:
        _, middleware = runtime.session_graph(session, sdk, ActiveRun())
        assert middleware._ov_session_engine.config.commit_wait_timeout == 300
        assert session.config_snapshot.context_config.commit_wait_timeout == 300
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()
