"""Offline negative controls for the real-service acceptance collector."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.st.ov_context.components import (
    Suite,
    Tenant,
    execute,
    sha,
    signature,
    validate_model_input,
)

TENANT = Tenant("account", "alice", "mail", "s", "marker")


@pytest.mark.parametrize("changed_template,indexed", [(False, True), (True, True), (True, False)])
async def test_index_checks_new_and_changed_memories_without_type_assumptions(
    tmp_path: Path, monkeypatch, changed_template: bool, indexed: bool
) -> None:
    seed_uri = "viking://user/alice/memories/identity.md"
    new_uri = "viking://user/alice/memories/entities/order.md"
    (tmp_path / "memory-baseline.json").write_text(json.dumps({seed_uri: sha("seed")}))
    monkeypatch.setattr(
        "benchmark.office.suites.common.memory_inventory.memory_leaves",
        AsyncMock(return_value=[{"uri": seed_uri}, {"uri": new_uri}]),
    )
    sdk = SimpleNamespace()
    sdk.read = AsyncMock(
        side_effect=lambda uri, **_: (
            "changed"
            if uri == seed_uri and changed_template
            else "seed"
            if uri == seed_uri
            else "new"
        )
    )
    sdk.find = AsyncMock(
        side_effect=lambda *_, target_uri, **__: SimpleNamespace(
            memories=[SimpleNamespace(uri=target_uri, level=2)] if indexed else []
        )
    )
    suite = Suite.__new__(Suite)
    suite.prefix, suite.output, suite.sdk = "probe", tmp_path, sdk
    if indexed:
        result = await suite.index()
        assert len(result["files"]) == (2 if changed_template else 1)
        assert result["missing_index"] == 0
    else:
        with pytest.raises(AssertionError, match="missing from the searchable"):
            await suite.index()


def sample() -> dict[str, Any]:
    return {
        "system": {
            "content": "SYS_HEADER:order-audit\nTENANT:account/alice/mail\nBASE_RULE:use_only_this_tenant_data\n[OpenViking Context Guide]"
        },
        "input_tokens": 100,
        "input_limit": 200,
    }


def test_prefix_and_budget_control_accepts_valid_input() -> None:
    validate_model_input(sample(), TENANT)


@pytest.mark.parametrize(
    "tamper", ["missing", "duplicate", "foreign", "overflow", "actual-overflow", "no-system"]
)
def test_prefix_and_budget_control_rejects_invalid_input(tamper: str) -> None:
    value = sample()
    if tamper == "missing":
        value["system"]["content"] = value["system"]["content"].replace(
            "SYS_HEADER:order-audit", ""
        )
    elif tamper == "duplicate":
        value["system"]["content"] += "\nBASE_RULE:use_only_this_tenant_data"
    elif tamper == "foreign":
        value["system"]["content"] = value["system"]["content"].replace(
            "account/alice/mail", "account/bob/mail"
        )
    elif tamper == "overflow":
        value["input_tokens"] = 201
    elif tamper == "actual-overflow":
        value["actual_input_tokens"] = 201
    else:
        value["system"] = None
    with pytest.raises(AssertionError):
        validate_model_input(value, TENANT)


def test_projection_ignores_only_local_native_metadata() -> None:
    local = HumanMessage(id="lc-1", content="hello", additional_kwargs={"trace": "local"})
    stored = HumanMessage(id="ov-1", content=[{"type": "text", "text": "hello"}])
    assert signature([local]) == signature([stored])
    assert signature([local]) != signature([stored.model_copy(update={"content": "different"})])


@pytest.mark.parametrize(
    "field,value",
    [("tool_call_id", "other"), ("status", "error"), ("name", "other"), ("content", "different")],
)
def test_projection_detects_tool_semantic_changes(field: str, value: str) -> None:
    original = ToolMessage(id="local", tool_call_id="call", name="read", content="result")
    assert signature([original]) != signature([original.model_copy(update={field: value})])


def test_projection_detects_call_arguments_and_message_order() -> None:
    human = HumanMessage(content="question")
    ai = AIMessage(content="", tool_calls=[{"id": "c", "name": "read", "args": {"limit": 10}}])
    changed = AIMessage(content="", tool_calls=[{"id": "c", "name": "read", "args": {"limit": 11}}])
    assert signature([human, ai]) != signature([human, changed])
    assert signature([human, ai]) != signature([ai, human])


@pytest.mark.parametrize(
    "parallelism,timeout", [(0, 1), (33, 1), (1, 0), (1, float("nan")), (1, float("inf"))]
)
async def test_invalid_limits_fail_before_creating_network_clients(
    tmp_path: Path, parallelism: int, timeout: float
) -> None:
    args = argparse.Namespace(output=tmp_path, parallelism=parallelism, timeout=timeout)
    with pytest.raises(ValueError):
        await execute(args)


@pytest.mark.parametrize(
    "command,filename", [("writer", "conditions-writer.json"), ("worker", "worker-resume.json")]
)
async def test_existing_evidence_is_never_overwritten(
    tmp_path: Path, command: str, filename: str
) -> None:
    evidence = tmp_path / filename
    evidence.write_text("preserved")
    args = argparse.Namespace(
        output=tmp_path, parallelism=1, timeout=1, command=command, worker_phase="resume"
    )
    with pytest.raises(FileExistsError):
        await execute(args)
    assert evidence.read_text() == "preserved"


@pytest.mark.parametrize("outcome", ["pass", "error", "timeout"])
async def test_case_collector_records_failures_and_timeouts(tmp_path: Path, outcome: str) -> None:
    suite = Suite(
        argparse.Namespace(output=tmp_path, prefix="test", command="writer", timeout=0.01)
    )

    async def action() -> dict[str, Any]:
        if outcome == "error":
            raise ValueError("controlled failure")
        if outcome == "timeout":
            await asyncio.Event().wait()
        return {"verified": True}

    await suite.case("sample", action)
    row = json.loads((tmp_path / "results-writer.json").read_text())[0]
    assert row["status"] == ("passed" if outcome == "pass" else "failed")
    if outcome != "pass":
        failure = json.loads((tmp_path / "private/sample-error.json").read_text())
        assert failure["type"] == ("ValueError" if outcome == "error" else "TimeoutError")
        assert "Traceback" in failure["traceback"]


async def test_capture_keeps_provider_usage_with_the_correct_concurrent_owner() -> None:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import SystemMessage
    from langgraph.runtime import Runtime

    from examples.deepagents_ov_context.chat.agent import RequestContext
    from openviking.integrations.langchain.ov_context import OVContextConfig
    from tests.st.ov_context.components import Capture

    sink: list[dict[str, Any]] = []
    capture = Capture(
        sink, OVContextConfig(context_window=10000, output_reserve=100, safety_margin=0), []
    )
    ready = asyncio.Event()
    waiting = 0

    async def invoke(account: str, tokens: int) -> None:
        identity = Tenant(account, "user", "peer", "session", "marker").identity("same-run")
        request = ModelRequest(
            model=FakeListChatModel(responses=["unused"]),
            messages=[HumanMessage(content="question")],
            system_message=SystemMessage(content="rules"),
            state={"messages": []},
            runtime=Runtime(context=RequestContext(identity)),
        )

        async def handler(value: ModelRequest[Any]) -> ModelResponse[Any]:
            nonlocal waiting
            waiting += 1
            if waiting == 2:
                ready.set()
            await ready.wait()
            return ModelResponse(
                result=[
                    AIMessage(
                        content="done",
                        usage_metadata={
                            "input_tokens": tokens,
                            "output_tokens": 1,
                            "total_tokens": tokens + 1,
                        },
                    )
                ]
            )

        await capture.awrap_model_call(request, handler)

    await asyncio.gather(invoke("account-a", 150), invoke("account-b", 300))
    assert {row["owner"][0]: row["actual_input_tokens"] for row in sink} == {
        "account-a": 150,
        "account-b": 300,
    }


async def test_capture_rejects_invalid_model_tool_arguments() -> None:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.runtime import Runtime

    from examples.deepagents_ov_context.chat.agent import RequestContext
    from openviking.integrations.langchain.ov_context import OVContextConfig
    from tests.st.ov_context.components import Capture

    capture = Capture([], OVContextConfig(context_window=10000), [])
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="question")],
        state={"messages": []},
        runtime=Runtime(context=RequestContext(TENANT.identity("r"))),
    )

    async def handler(value: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(
            result=[
                AIMessage(
                    content="preparing",
                    invalid_tool_calls=[
                        {"name": "read", "args": "invalid JSON", "id": "c", "error": "parse"}
                    ],
                )
            ]
        )

    with pytest.raises(AssertionError, match="invalid tool arguments"):
        await capture.awrap_model_call(request, handler)


@pytest.mark.parametrize("bypass_owner_check", [False, True])
async def test_collision_validates_application_owner_before_ov(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_owner_check: bool
) -> None:
    from examples.deepagents_ov_context.chat.server import SessionStore

    suite = Suite.__new__(Suite)
    suite.prefix, suite.output, suite.requests = "application-owner", tmp_path, []
    sdk = SimpleNamespace(
        create_session=AsyncMock(), add_message=AsyncMock(), get_session_context=AsyncMock()
    )
    suite.sdk = sdk
    if bypass_owner_check:
        monkeypatch.setattr(SessionStore, "require_identity", lambda self, identity: None)
        with pytest.raises(AssertionError, match="must reject a different peer"):
            await suite.collision()
        sdk.get_session_context.assert_awaited_once()
    else:
        result = await suite.collision()
        assert result["different_peer_rejected"]
        assert result["rejection_layer"] == "application"
        assert result["ov_requests_after_rejection"] == 0
        sdk.get_session_context.assert_not_awaited()
