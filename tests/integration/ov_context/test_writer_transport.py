from dataclasses import replace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context import RunIdentity
from openviking.integrations.langchain.ov_context.metrics import LatencySamples
from tests.ov_context_support.models import ResolvedCaller
from tests.ov_context_support.transport import NativeHTTPTransport, OVHTTPError

pytestmark = pytest.mark.asyncio


def interaction():
    return [
        HumanMessage(id="m1", content="Read the mail"),
        AIMessage(
            id="m2",
            content="reading",
            tool_calls=[{"id": "c1", "name": "read", "args": {"path": "mail"}}],
        ),
        ToolMessage(id="m3", tool_call_id="c1", content="entire result"),
    ]


async def test_identity_headers_are_fresh_and_redirects_not_followed(env):
    server, transport, _, identity, _, _ = env
    for actor in ("p", "other"):
        await transport.context(replace(identity, actor_peer_id=actor), 100)
        headers = server.requests[-1][0].headers
        assert headers["X-OpenViking-Actor-Peer"] == actor
        assert headers["X-API-Key"] == "test-only"
        assert "X-OpenViking-Account" not in headers and "X-OpenViking-User" not in headers
    trusted = NativeHTTPTransport(
        url="http://ov.test",
        http_client=transport.http,
        caller=ResolvedCaller("ov", "a", "u", "v1", "trusted"),
    )
    await trusted.context(identity, 100)
    headers = server.requests[-1][0].headers
    assert headers["X-OpenViking-Account"] == "a" and headers["X-OpenViking-User"] == "u"
    assert "X-API-Key" not in headers
    with pytest.raises(ValueError, match="identity defaults"):
        NativeHTTPTransport(
            url="http://ov.test",
            caller=transport.caller,
            api_key="x",
            http_client=httpx.AsyncClient(headers={"X-OpenViking-User": "other"}),
        )
    server.error["context"] = 302
    with pytest.raises(OVHTTPError):
        await transport.context(identity, 100)
    await transport.http.aclose()


async def test_wire_contract_matches_native_request_models():
    # Import the real models with their dependencies, without starting a server.
    from pydantic import ValidationError

    from openviking.server.routers.sessions import (
        AddMessageRequest,
        BatchAddMessageRequest,
        CommitRequest,
        CreateSessionRequest,
    )

    payload = {
        "role": "user",
        "peer_id": "p",
        "parts": [{"type": "tool", "tool_id": "c", "tool_output": "full result"}],
    }
    AddMessageRequest(**payload)
    BatchAddMessageRequest(messages=[payload])
    assert CommitRequest(keep_recent_count=10).keep_recent_count == 10
    assert CreateSessionRequest(auto_commit_policy={"keep_recent_count": 0}).auto_commit_policy
    assert (
        not {"source_message_id", "payload_hash", "execution_scope_id", "run_id"}
        & AddMessageRequest.model_fields.keys()
    )
    with pytest.raises(ValidationError):
        AddMessageRequest(role="user")


async def test_latency_summary_uses_independent_events_and_nearest_rank():
    samples = LatencySamples(10)
    for n in range(1, 11):
        samples("search", duration_ms=n, status="ok")
    samples("search", duration_ms=100, status="error")
    rows = samples.summary()
    ok = next(row for row in rows if row["status"] == "ok")
    assert ok == {
        "event": "search",
        "status": "ok",
        "count": 10,
        "sum_ms": 55.0,
        "avg_ms": 5.5,
        "p90_ms": 9.0,
        "max_ms": 10.0,
    }


async def test_existing_session_identity_mapping_and_owner_check():
    from tests.ov_context_support.models import resolve_run_identity

    caller = ResolvedCaller("ov", "a", "u", "permission-v1")
    record = {"account_id": "a", "user_id": "u", "session_id": "agent-s", "ov_session_id": "ov-s"}
    identity = resolve_run_identity(caller, record, actor_peer_id="p", run_id="r")
    assert identity.session_id == "ov-s"
    with pytest.raises(ValueError, match="Session owner"):
        resolve_run_identity(
            caller, {**record, "account_id": "other"}, actor_peer_id="p", run_id="r"
        )
    with pytest.raises(ValueError):
        RunIdentity("a", "u", "p\tbad", "s", "r")
