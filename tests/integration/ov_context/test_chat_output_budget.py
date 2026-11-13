"""Larger per-session output budgets reach the model and reserve input space."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from examples.deepagents_ov_context.chat.server import ActiveRun


@pytest.mark.parametrize("mode", ["baseline", "ordinary"])
@pytest.mark.parametrize("limit", [32768, 65536])
async def test_large_output_budget_survives_graph_rebuild(chat, monkeypatch, mode, limit):
    client, runtime = chat
    runtime.settings = replace(runtime.settings, max_tokens=limit)
    runtime.model = ArkChatOpenAI(
        api_key="offline", model="deepseek-v4-flash", max_completion_tokens=2048
    )
    captured = {}
    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    response = await client.post(
        "/api/sessions",
        json={"execution": {"mode": mode, "context_window": 256000, "max_tokens": limit}},
    )
    assert response.status_code == 201
    session = runtime.store.get(response.json()["session_id"])
    assert session.config_snapshot.context_config.output_reserve == limit
    assert session.config_snapshot.execution["profile"]["max_tokens"] == limit
    sdk = runtime.sdk(session)
    middleware = None
    try:
        for _ in range(2):
            _, middleware = runtime.session_graph(session, sdk, ActiveRun())
            payload = captured["model"]._get_request_payload("Return OK.")
            assert payload["max_completion_tokens"] == limit
            assert "max_tokens" not in payload
            if middleware is not None:
                await middleware.aclose()
                middleware = None
        assert runtime.model.max_tokens == 2048
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()


async def test_output_budget_limits_and_window_are_still_checked(chat):
    client, runtime = chat
    for execution in (
        {"max_tokens": 65537},
        {"max_tokens": 0},
        {"context_window": 32768, "max_tokens": 32768},
    ):
        response = await client.post("/api/sessions", json={"execution": execution})
        assert response.status_code == 422
    with pytest.raises(ValueError, match="65536"):
        replace(runtime.settings, max_tokens=65537)
