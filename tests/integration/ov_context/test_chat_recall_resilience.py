"""Read-only recall failure does not disable later OV writes or future Runs."""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.deepagents_ov_context.chat.config import ChatOVPolicy
from examples.deepagents_ov_context.chat.ov_resilience import _RecallDeadlineReader
from openviking.integrations.langchain.ov_context.errors import IdentityMismatchError
from openviking.integrations.langchain.ov_context.reader import RecallResult

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events
from .test_chat_resilience import new


@pytest.mark.parametrize("search_mode", ["find", "context"])
async def test_recall_timeout_keeps_tool_loop_writes_commit_and_next_run(
    resilient_chat, search_mode
):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        ov_policy=ChatOVPolicy(io_timeout=0.3),
        reader_config=replace(
            runtime.settings.reader_config, search_mode=search_mode, recall_timeout=30
        ),
        commit_config=replace(
            runtime.settings.commit_config, completed_run_threshold=1, keep_recent_count=4
        ),
    )
    server.session_meta_hook = lambda sid: {"pending_tokens": 128}
    model_inputs = []

    class ToolModel(NoToolModel):
        def _generate(self, messages, *args, **kwargs):
            model_inputs.append(list(messages))
            index = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
            if any(isinstance(m, ToolMessage) for m in messages[index:]):
                return super()._generate(messages, *args, **kwargs)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": f"echo-{messages[index].id}",
                                    "name": "debug_echo",
                                    "args": {"text": "ok"},
                                }
                            ],
                        )
                    )
                ]
            )

    runtime.model = ToolModel()
    searches = []
    slow = True

    async def fault(request):
        if request.url.path == "/api/v1/search/find":
            searches.append(request.url.path)
            if slow:
                await asyncio.sleep(10)

    server.fault = fault
    sid = await new(client)
    for turn in range(2):
        response = await client.post(
            "/api/chat", json={"session_id": sid, "message": f"turn {turn}"}
        )
        events = parse_events(response.text)
        assert events[-1][0] == "done", response.text
        assert events[-1][1]["statistics"]["model_calls"] == 2
        view = events[-1][1]["context"]
        assert not view["ov_degraded"] and not view["ov_sync_pending"]
        assert view["write_index"] == view["raw_message_count"]
        samples = [d for e, d in events if e == "stage" and d.get("name") == "memory.recall"]
        assert len(samples) == 1  # No repeated waits during the model/tool loop.
        assert samples[0]["status"] == ("timeout" if slow else "success")
        assert not any(e == "stage" and d.get("name") == "ov.degraded" for e, d in events)
        if turn == 0:
            attempts = len(searches)
            assert attempts > 0
            slow = False
        else:
            assert len(searches) > attempts  # The same session retries on the next Run.
    assert len(server.messages) == 8  # user, AI tool call, tool result, final AI per Run.
    assert server.counts["commit"] > 0
    assert "short answers" not in str(model_inputs[:2])
    assert "short answers" in str(model_inputs[2:])


async def test_reader_failure_is_optional_but_later_write_failure_is_not(
    resilient_chat, monkeypatch
):
    from examples.deepagents_ov_context.chat.profiles import ObservedContextReader

    client, _, server = resilient_chat
    monkeypatch.setattr(
        ObservedContextReader, "recall", AsyncMock(side_effect=RuntimeError("reader"))
    )

    async def fault(request):
        if request.url.path.endswith("/messages/batch") and server.messages:
            raise TimeoutError("write receipt unknown")

    server.fault = fault
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert events[-1][1]["statistics"]["model_calls"] == 1
    stages = [d for e, d in events if e == "stage"]
    assert any(d.get("name") == "memory.recall" and d.get("status") == "error" for d in stages)
    failures = [d for d in stages if d.get("name") == "ov.degraded"]
    assert len(failures) == 1 and failures[0]["stage"] == "after_model"
    assert events[-1][1]["context"]["ov_degraded"]


async def test_recall_budget_uses_remaining_preparation_time(resilient_chat, monkeypatch):
    from openviking.integrations.langchain.ov_context.session_engine import SessionEngine

    client, runtime, server = resilient_chat
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy(io_timeout=0.3))
    original = SessionEngine.check_summary

    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.18)
        return await original(*args, **kwargs)

    monkeypatch.setattr(SessionEngine, "check_summary", delayed)

    async def search():
        await asyncio.sleep(10)

    server.search_hook = search
    sid = await new(client)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", response.text
    assert not events[-1][1]["context"]["ov_degraded"]
    assert len(server.messages) == 2


async def test_recall_deadline_is_task_local_and_discards_late_result():
    finished = asyncio.Event()

    async def recall(query, **kwargs):
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
        finished.set()
        return RecallResult("late memory", "success", 1)

    reader = _RecallDeadlineReader(SimpleNamespace(recall=recall))

    async def bounded():
        token = reader.deadline.set(asyncio.get_running_loop().time() + 0.02)
        try:
            started = time.monotonic()
            result = await reader.recall("bounded")
            assert time.monotonic() - started < 0.05
            return result
        finally:
            reader.deadline.reset(token)

    timed_out, normal = await asyncio.gather(bounded(), reader.recall("unbounded"))
    assert timed_out == RecallResult(status="timeout")
    assert normal.content == "late memory"
    await asyncio.wait_for(finished.wait(), 1)
    await asyncio.sleep(0.06)  # Allow the canceled client's late result callback to complete.


@pytest.mark.parametrize(
    "failure",
    [
        IdentityMismatchError("owner"),
        httpx.HTTPStatusError(
            "denied",
            request=httpx.Request("POST", "http://ov.test/search"),
            response=httpx.Response(403),
        ),
        asyncio.CancelledError(),
    ],
)
async def test_read_guard_does_not_swallow_identity_errors_or_cancellation(failure):
    reader = _RecallDeadlineReader(SimpleNamespace(recall=AsyncMock(side_effect=failure)))
    token = reader.deadline.set(asyncio.get_running_loop().time() + 1)
    try:
        with pytest.raises(type(failure)):
            await reader.recall("query")
    finally:
        reader.deadline.reset(token)
