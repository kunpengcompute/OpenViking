# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Chat rejects invalid OV identities and preserves parameter-error failures."""

from dataclasses import replace

import httpx
import pytest

from examples.deepagents_ov_context.chat.environments import ChatEnvironment
from examples.deepagents_ov_context.chat.server import ChatSettings
from examples.deepagents_ov_context.chat.session import NewSession
from openviking_cli.exceptions import InvalidArgumentError
from tests.ov_context_support.commit_settings import EXAMPLE_COMMIT_CONFIG

from .test_chat_api import new_session, parse_events


@pytest.mark.parametrize("field", ["user_id", "agent_id"])
@pytest.mark.parametrize("value", ["alice bob", ".", "..", "a@b@c", "user:1", "用户"])
async def test_invalid_session_identity_is_rejected_before_saving(chat, field, value):
    client, runtime = chat
    response = await client.post("/api/sessions", json={field: value})
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])
    assert runtime.store.list() == []


@pytest.mark.parametrize("field", ["user_id", "agent_id"])
@pytest.mark.parametrize("value", ["user-1", "user.name", "user_name", "user@example.com"])
def test_valid_session_identity_uses_ov_character_rules(field, value):
    assert getattr(NewSession(**{field: value}), field) == value


@pytest.mark.parametrize("value", ["bad account", ".", "..", "a@b@c", "_account", "用户"])
@pytest.mark.parametrize("source", ["settings", "environment"])
def test_invalid_account_is_rejected_in_service_configuration(tmp_path, source, value):
    with pytest.raises(ValueError):
        if source == "settings":
            ChatSettings(
                tmp_path / "models.json",
                tmp_path / "data",
                account_id=value,
                commit_config=EXAMPLE_COMMIT_CONFIG,
            )
        else:
            ChatEnvironment(id="test", label="test", ov_url="http://ov.test", account_id=value)


@pytest.mark.parametrize(
    "stage", ["prepare", "session", "write", "context", "recall_find", "recall_context"]
)
async def test_ov_parameter_error_is_terminal_instead_of_local_success(resilient_chat, stage):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        reader_config=replace(
            runtime.settings.reader_config,
            search_mode="find" if stage == "recall_find" else "context",
        ),
    )
    failed = []

    async def fault(request):
        path = request.url.path
        matches = (
            stage == "prepare"
            or (stage == "session" and path == "/api/v1/sessions" and request.method == "POST")
            or (stage == "write" and path.endswith("/messages/batch"))
            or (stage == "context" and path.endswith("/context"))
            or (stage.startswith("recall_") and "/search/" in path)
        )
        if matches:
            failed.append(path)
            return httpx.Response(
                400,
                json={
                    "status": "error",
                    "error": {
                        "code": "INVALID_ARGUMENT",
                        "message": "invalid request identity or parameter",
                    },
                },
            )

    server.fault = fault
    sid = await new_session(
        client, execution={"mode": "context_only" if stage == "prepare" else "current"}
    )
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    event, result = parse_events(response.text)[-1]
    assert failed, stage
    assert event == "error", result.get("error_type")
    assert result["error_type"] == "InvalidArgumentError"
    assert "请求参数无效" in result["message"]
    assert result["statistics"]["model_calls"] == 0
    assert not any(
        event == "stage" and data.get("name") == "ov.degraded"
        for event, data in parse_events(response.text)
    )
    assert not (await runtime.state(runtime.store.get(sid))).get("ov_failure")


async def test_parameter_error_bypasses_initial_context_fallback(env):
    from .test_hook_design import Reader, make, runtime

    reader = Reader()
    reader.error = InvalidArgumentError("invalid context parameters")
    middleware = make(env, reader)
    engine = middleware._ov_session_engine
    engine.config = replace(engine.config, context_failure="host")
    with pytest.raises(InvalidArgumentError):
        await middleware.abefore_agent({"messages": env[4].messages}, runtime(env))


async def test_capacity_compression_preserves_parameter_error(env):
    from unittest.mock import AsyncMock

    from openviking.integrations.langchain.ov_context import OVContext

    from .test_hook_design import Reader, make

    middleware = make(env, Reader())
    engine = middleware._ov_session_engine
    error = InvalidArgumentError("invalid commit parameters")
    engine.commit = AsyncMock(side_effect=error)
    with pytest.raises(InvalidArgumentError) as raised:
        await middleware._compress_context(
            OVContext("s"),
            identity=env[3].sdk_identity,
            pending_tokens=1,
            token_budget=1000,
            request=env[4],
        )
    assert raised.value is error


async def test_ov_tool_parameter_error_is_not_persisted_as_degradation():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from examples.deepagents_ov_context.chat.ov_resilience import ResilientOVMiddleware

    middleware = ResilientOVMiddleware(commit_config=None)
    request = SimpleNamespace(
        tool_call={"name": "openviking_tool_result_read", "id": "read"}, state={}
    )
    with pytest.raises(InvalidArgumentError):
        await middleware.awrap_tool_call(
            request, AsyncMock(side_effect=InvalidArgumentError("invalid ref"))
        )
    assert middleware.failure is None
