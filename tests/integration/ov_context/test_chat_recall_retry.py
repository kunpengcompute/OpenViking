"""Optional benchmark recall retries finish before any model/business execution."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.deepagents_ov_context.chat.ov_resilience import _RecallDeadlineReader
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile
from openviking.integrations.langchain.ov_context.reader import RecallResult

from .test_chat_api import parse_events


@pytest.mark.asyncio
async def test_deadline_timeout_retries_before_returning_injected_memory():
    calls = []
    emitted = []

    async def recall(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            await asyncio.sleep(10)
        return RecallResult(content="frozen method memory", status="success", count=1)

    reader = _RecallDeadlineReader(
        SimpleNamespace(recall=recall),
        retry_attempts=2,
        retry_delay=0.001,
        attempt_budget=0.03,
        observer=lambda name, **fields: emitted.append((name, fields)),
    )
    async with asyncio.timeout(0.03) as scope:
        token = reader.deadline.set(scope.when())
        outer = reader.timeout_scope.set(scope)
        try:
            result = await reader.recall("Q", identity="same-identity")
        finally:
            reader.deadline.reset(token)
            reader.timeout_scope.reset(outer)
    assert result.status == "success" and len(calls) == 3
    assert calls == [{"identity": "same-identity"}] * 3
    assert [f["status"] for n, f in emitted if n == "memory_recall_attempt"] == [
        "timeout",
        "timeout",
        "success",
    ]
    assert [f["next_attempt"] for n, f in emitted if n == "memory_recall_retry"] == [2, 3]


@pytest.mark.asyncio
async def test_exhausted_retries_remain_timeout_not_fake_memory():
    recall = AsyncMock(return_value=RecallResult(status="timeout"))
    reader = _RecallDeadlineReader(
        SimpleNamespace(recall=recall),
        retry_attempts=2,
        retry_delay=0,
        attempt_budget=0.03,
    )
    async with asyncio.timeout(0.03) as scope:
        reader.deadline.set(scope.when())
        reader.timeout_scope.set(scope)
        result = await reader.recall("Q")
    assert result.status == "timeout" and not result.content
    assert recall.await_count == 3


@pytest.mark.asyncio
async def test_empty_or_error_results_are_not_retried():
    for status in ["empty", "error", "success"]:
        recall = AsyncMock(return_value=RecallResult(status=status))
        reader = _RecallDeadlineReader(
            SimpleNamespace(recall=recall),
            retry_attempts=2,
            retry_delay=0,
            attempt_budget=0.03,
        )
        async with asyncio.timeout(0.03) as scope:
            reader.deadline.set(scope.when())
            reader.timeout_scope.set(scope)
            assert (await reader.recall("Q")).status == status
        assert recall.await_count == 1


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_start_another_attempt():
    entered = asyncio.Event()

    async def recall(*args, **kwargs):
        entered.set()
        await asyncio.sleep(10)

    reader = _RecallDeadlineReader(
        SimpleNamespace(recall=recall),
        retry_attempts=2,
        retry_delay=0,
        attempt_budget=0.1,
    )

    async def run():
        async with asyncio.timeout(0.1) as scope:
            reader.deadline.set(scope.when())
            reader.timeout_scope.set(scope)
            await reader.recall("Q")

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_retry_setting_is_explicit_bounded_and_persistable():
    assert ExecutionProfile().memory_recall_retry_attempts == 0
    p = ExecutionProfile(mode="ordinary", memory_recall_retry_attempts=2)
    assert ExecutionProfile.model_validate(p.model_dump()).memory_recall_retry_attempts == 2
    for value in [-1, 4, True, "2"]:
        with pytest.raises(ValueError):
            ExecutionProfile(memory_recall_retry_attempts=value)


@pytest.mark.parametrize("succeed", [True, False])
async def test_chat_retries_only_recall_before_single_model_and_single_write(
    resilient_chat, monkeypatch, succeed
):
    from examples.deepagents_ov_context.chat.config import ChatOVPolicy
    from openviking.integrations.langchain.ov_context.reader import OVContextReader

    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(io_timeout=0.05))
    calls = []

    async def recall(self, *args, **kwargs):
        calls.append(kwargs["identity"])
        if len(calls) < 3 or not succeed:
            return RecallResult(status="timeout")
        return RecallResult(content="method memory verified", status="success", count=1)

    monkeypatch.setattr(OVContextReader, "recall", recall)
    created = await client.post(
        "/api/sessions",
        json={"execution": {"mode": "ordinary", "memory_recall_retry_attempts": 2}},
    )
    assert created.status_code == 201
    saved = created.json()
    assert saved["config_snapshot"]["execution"]["profile"]["memory_recall_retry_attempts"] == 2
    response = await client.post(
        "/api/chat", json={"session_id": saved["session_id"], "message": "run once"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert len(calls) == 3 and calls == [calls[0]] * 3
    attempts = [d for e, d in events if e == "stage" and d.get("name") == "memory_recall_attempt"]
    assert [d["status"] for d in attempts] == [
        "timeout",
        "timeout",
        "success" if succeed else "timeout",
    ]
    inputs = [d for e, d in events if e == "model_input"]
    assert len(inputs) == events[-1][1]["statistics"]["model_calls"] == 1
    assert ("method memory verified" in str(inputs[0]["messages"])) is succeed
    assert len(server.messages) == 2  # Only the user message and final model response.
    assert events[-1][1]["statistics"]["seconds"] >= 2  # Backoff stays in wall-clock budget.
