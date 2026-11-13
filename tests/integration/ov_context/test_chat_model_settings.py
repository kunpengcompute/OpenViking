"""Model-call deadline defaults are configurable and frozen per Session."""

import json
from dataclasses import replace

import httpx
import pytest

from examples.deepagents_ov_context.chat import server as chat_server
from examples.deepagents_ov_context.chat.config import load_model_call_timeout
from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from examples.deepagents_ov_context.chat.server import ActiveRun, ChatRuntime

from .test_chat_api import parse_events
from .test_model_call_deadline import StalledReasoningStream


def test_deadline_env_precedence_and_validation(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAT_MODEL__CALL_TIMEOUT_SECONDS", raising=False)
    p = tmp_path / ".env"
    p.write_text("CHAT_MODEL__CALL_TIMEOUT_SECONDS=75\n")
    assert load_model_call_timeout(p) == 75
    monkeypatch.setenv("CHAT_MODEL__CALL_TIMEOUT_SECONDS", "90")
    assert load_model_call_timeout(p) == 90
    for value in ("0", "-1", "nan", "inf", "3601"):
        monkeypatch.setenv("CHAT_MODEL__CALL_TIMEOUT_SECONDS", value)
        with pytest.raises(ValueError):
            load_model_call_timeout(p)


async def test_deadline_snapshot_survives_runtime_rebuild(chat, monkeypatch):
    client, first = chat
    first.settings = replace(first.settings, model_call_timeout_seconds=75)
    response = await client.post("/api/sessions", json={})
    sid = response.json()["session_id"]
    assert (
        first.store.get(sid).config_snapshot.execution["profile"]["model_call_timeout_seconds"]
        == 75
    )
    seen = []
    original = chat_server.create_deep_agent

    def capture(**kwargs):
        seen.append(kwargs["model"].async_call_timeout_seconds)
        return original(**kwargs)

    monkeypatch.setattr(chat_server, "create_deep_agent", capture)
    rebuilt = ChatRuntime(
        replace(first.settings, model_call_timeout_seconds=30),
        first.model_config,
        checkpointer=first.saver,
    )
    try:
        session = rebuilt.store.get(sid)
        sdk = rebuilt.sdk()
        middleware = None
        try:
            _, middleware = rebuilt.session_graph(session, sdk, ActiveRun())
            assert seen[-1] == 75
            assert rebuilt.model.async_call_timeout_seconds == 30
        finally:
            if middleware is not None:
                await middleware.aclose()
            await sdk.close()
        first.settings = replace(first.settings, model_call_timeout_seconds=30)
        response = await client.post(
            "/api/sessions",
            json={
                "execution": {"model_call_timeout_seconds": 12},
            },
        )
        assert response.status_code == 201
        assert (
            response.json()["config_snapshot"]["execution"]["profile"]["model_call_timeout_seconds"]
            == 12
        )
    finally:
        await rebuilt.close()


async def test_timeout_is_terminal_and_next_chat_still_works(chat):
    client, runtime = chat
    hanging = StalledReasoningStream()
    fast = False

    def respond(request):
        if not fast:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=hanging
            )
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as model_http:
        runtime.model = ArkChatOpenAI(
            api_key="offline",
            http_async_client=model_http,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        response = await client.post(
            "/api/sessions",
            json={
                "execution": {
                    "mode": "baseline",
                    "model_call_timeout_seconds": 0.15,
                    "timeout_seconds": 3,
                }
            },
        )
        sid = response.json()["session_id"]
        response = await client.post("/api/chat", json={"session_id": sid, "message": "q"})
        events = parse_events(response.text)
        assert events[-1][0] == "error"
        assert "ModelCallTimeoutError" in response.text
        assert "等待模型响应超时" in response.text
        assert not runtime.active and hanging.closed
        assert runtime.store.get(sid).last_run.status == "error"
        fast = True
        fresh = await client.post(
            "/api/sessions",
            json={
                "execution": {
                    "mode": "baseline",
                    "model_call_timeout_seconds": 2,
                    "timeout_seconds": 3,
                }
            },
        )
        response = await client.post(
            "/api/chat",
            json={
                "session_id": fresh.json()["session_id"],
                "message": "q",
            },
        )
        assert parse_events(response.text)[-1][0] == "done"
