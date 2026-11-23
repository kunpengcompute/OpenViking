# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Agent Thinking settings reach the wire and stay frozen across graph rebuilds."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from examples.deepagents_ov_context.chat import server
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile


@pytest.fixture(autouse=True)
def clean_thinking_environment(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL__THINKING", raising=False)


@pytest.mark.parametrize("thinking", [None, False, True])
@pytest.mark.parametrize("endpoint", ["https://gateway.invalid/v1", "https://ark.volces.com/v3"])
async def test_thinking_reaches_http_and_survives_runtime_rebuild(
    chat, monkeypatch, endpoint, thinking
):
    _, template = chat
    actual_client = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
        )

    monkeypatch.setattr(
        server,
        "httpx",
        SimpleNamespace(
            Timeout=httpx.Timeout,
            AsyncClient=lambda **kwargs: actual_client(
                **kwargs, transport=httpx.MockTransport(respond)
            ),
        ),
    )
    monkeypatch.setattr(server, "create_deep_agent", lambda **kwargs: kwargs["model"])
    config = replace(template.model_config, api_base=endpoint)
    first = server.ChatRuntime(template.settings, config, checkpointer=template.saver)
    rebuilt = None
    try:
        assert first.settings.thinking is True
        assert first.model.extra_body["thinking"] == {"type": "enabled"}
        session = server.SessionInfo(
            session_id="thinking-session",
            account_id=first.settings.account_id,
            user_id="user",
            agent_id="agent",
            created_at=0,
            execution=ExecutionProfile(mode="baseline", thinking=thinking),
        )
        session.config_snapshot = first.snapshot(session)
        first.store.save(session)
        expected = {"type": "disabled" if thinking is False else "enabled"}
        rebuilt = server.ChatRuntime(
            replace(first.settings, thinking=False), config, checkpointer=first.saver
        )
        assert rebuilt.model.extra_body["thinking"] == {"type": "disabled"}
        for runtime in (first, rebuilt):
            runtime.model = runtime.model.model_copy(
                update={"extra_body": {**(runtime.model.extra_body or {}), "custom_probe": 1}}
            )
            saved = runtime.store.get(session.session_id)
            assert saved.config_snapshot.execution["profile"]["thinking"] is (thinking is not False)
            for _ in range(2):
                model, middleware = runtime.session_graph(saved, None, server.ActiveRun())
                assert middleware is None
                result = await model.ainvoke("Return OK")
                assert result.content == "OK"
                assert requests[-1]["thinking"] == expected
                assert requests[-1]["custom_probe"] == 1
        assert len(requests) == 4
    finally:
        if rebuilt is not None:
            await rebuilt.close()
        await first.close()


@pytest.mark.parametrize(
    "file_value,process_value,cli,expected",
    [
        (None, None, [], True),
        ("false", None, [], False),
        ("false", "true", [], True),
        ("true", None, ["--no-thinking"], False),
        ("false", "false", ["--thinking"], True),
    ],
)
def test_thinking_cli_inherits_env_and_allows_explicit_override(
    tmp_path, monkeypatch, file_value, process_value, cli, expected
):
    import uvicorn

    env = tmp_path / ".env"
    from examples.deepagents_ov_context.chat.config import DEFAULT_ENV_FILE

    env.write_text(
        DEFAULT_ENV_FILE.read_text()
        + ("" if file_value is None else f"\nCHAT_MODEL__THINKING={file_value}\n")
    )
    if process_value is not None:
        monkeypatch.setenv("CHAT_MODEL__THINKING", process_value)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--model-config",
            str(tmp_path / "models.json"),
            "--data-dir",
            str(tmp_path / "chat"),
            "--env-file",
            str(env),
            *cli,
        ],
    )
    captured = []
    monkeypatch.setattr(server, "create_app", lambda settings: settings)
    monkeypatch.setattr(uvicorn, "run", lambda settings, **kwargs: captured.append(settings))
    server.main()
    assert captured[0].thinking is expected
