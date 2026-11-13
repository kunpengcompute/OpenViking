"""Commit, compression and errors remain attributable across shared Run IDs."""

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from openviking.integrations.langchain.ov_context import Metrics, OVContext
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot

from .test_hook_design import Reader, make
from .test_session_hooks import runtime

pytestmark = pytest.mark.asyncio


def assert_owner(sample: dict[str, Any]) -> None:
    assert {
        name: sample[name]
        for name in ("account_id", "user_id", "actor_peer_id", "session_id", "run_id")
    } == {"account_id": "a", "user_id": "u", "actor_peer_id": "p", "session_id": "s", "run_id": "r"}


async def test_commit_receipt_and_failure_have_owner_task_and_stack(env) -> None:
    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env, Reader(), metrics=Metrics(lambda event, **fields: samples.append((event, fields)))
    )
    engine = mw._ov_session_engine
    context = OVContext(
        "s",
        owner=env[3].session_key,
        run_id="r",
        messages=[HumanMessage(id="old", content="old"), AIMessage(id="answer", content="answer")],
    )
    engine.config = replace(engine.config)
    engine.client.call = AsyncMock(
        return_value={
            "archived": True,
            "task_id": "t",
            "archive_uri": "viking://session/s/history/archive_001",
        }
    )
    assert await engine.commit(context, identity=context.identity)
    receipt = [data for event, data in samples if event == "context.commit"][-1]
    assert_owner(receipt)
    assert receipt["status"] == "accepted"
    assert receipt["task_id"] == "t"
    assert receipt["target_archive_id"] == "archive_001"
    context.pending_commit = None
    engine._commit_progress.clear()
    context.archive_id = "archive_001"
    engine.client.call.side_effect = RuntimeError("commit unavailable")
    with pytest.raises(RuntimeError, match="commit unavailable"):
        await engine.commit(context, identity=context.identity)
    failure = [data for event, data in samples if event == "context.commit"][-1]
    assert_owner(failure)
    assert failure["status"] == "error"
    assert failure["error_type"] == "RuntimeError"
    assert "RuntimeError: commit unavailable" in failure["error_stack"]


async def test_compression_logs_only_after_validated_cache_publication(env) -> None:
    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env, Reader(), metrics=Metrics(lambda event, **fields: samples.append((event, fields)))
    )
    context = OVContext(
        "s",
        owner=env[3].session_key,
        run_id="r",
        messages=[HumanMessage(id="old", content="x" * 1000)],
        pending_commit={"task_id": "t", "archive_id": None, "target_archive_id": "archive_001"},
    )
    summary = AIMessage(id="ov:summary", content="x" * 100)
    engine = mw._ov_session_engine
    engine._apply_snapshot(context, ContextSnapshot([summary], "archive_001"))
    compressed = [data for event, data in samples if event == "context.compression"]
    assert len(compressed) == 1
    assert_owner(compressed[0])
    assert compressed[0]["compression_ratio"] == 0.9
    assert compressed[0]["archive_id"] == "archive_001"
    assert compressed[0]["measurement"] == "cached_message_content_chars"
    assert [data["status"] for event, data in samples if event == "context.commit_result"] == [
        "archived"
    ]
    engine._apply_snapshot(context, ContextSnapshot([summary], "archive_001"))
    assert len([1 for event, _ in samples if event == "context.compression"]) == 1


async def test_estimate_has_complete_identity_and_provider_usage_is_callback_owned(env) -> None:
    samples: list[tuple[str, dict[str, Any]]] = []
    mw = make(
        env, Reader(), metrics=Metrics(lambda event, **fields: samples.append((event, fields)))
    )
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))

    async def handler(request):
        return AIMessage(
            content="done",
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    await mw.awrap_model_call(env[4].override(state=state, runtime=runtime(env)), handler)
    estimates = [data for event, data in samples if event == "context.estimate"]
    assert len(estimates) == 1
    assert_owner(estimates[0])
    assert estimates[0]["estimated_input_tokens"] >= 0
    assert not any(event == "model_usage" for event, _ in samples)
