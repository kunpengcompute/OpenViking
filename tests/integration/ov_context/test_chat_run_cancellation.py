"""Run termination must finish delegated work before publishing a terminal event."""

import asyncio
import os
import shlex
import signal
import sys
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from examples.deepagents_ov_context.chat.office_skills import SkillBackend
from examples.deepagents_ov_context.chat.server import ActiveRun

from .chat_model_support import NoToolModel


class DelegatingModel(NoToolModel):
    role: str
    calls: Any

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(self.role)
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="finished")
        else:
            name = "task" if self.role == "parent" else "hold"
            args = (
                {"subagent_type": "worker", "description": "Call hold, then finish."}
                if self.role == "parent"
                else {}
            )
            message = AIMessage(
                content="", tool_calls=[{"id": name + "-call", "name": name, "args": args}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, *args, **kwargs):
        return self._generate(messages, *args, **kwargs)


@pytest.mark.parametrize("termination", ["timeout", "cancel"])
@pytest.mark.parametrize("tool_kind", ["async", "shell"])
async def test_run_termination_stops_subagent_before_terminal(
    chat, monkeypatch, tmp_path, termination, tool_kind
):
    client, runtime = chat
    response = await client.post(
        "/api/sessions", json={"execution": {"mode": "baseline", "timeout_seconds": 1}}
    )
    assert response.status_code == 201
    session = runtime.store.get(response.json()["session_id"])
    entered, released, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, processes = [], []
    backend = SkillBackend(root_dir=tmp_path)
    create_process = asyncio.create_subprocess_shell

    async def track_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        processes.append(process)
        entered.set()
        return process

    if tool_kind == "shell":
        monkeypatch.setattr(asyncio, "create_subprocess_shell", track_process)

    @tool
    async def hold() -> str:
        """Wait for the test to release this tool."""
        try:
            if tool_kind == "shell":
                command = f"exec {shlex.quote(sys.executable)} -c " + shlex.quote(
                    "import time; time.sleep(60)"
                )
                await backend.aexecute(command, timeout=30)
            else:
                entered.set()
                await released.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "released"

    graph = create_deep_agent(
        model=DelegatingModel(role="parent", calls=calls),
        subagents=[
            {
                "name": "worker",
                "description": "Cancellation test worker",
                "system_prompt": "Call hold exactly once, then finish.",
                "model": DelegatingModel(role="worker", calls=calls),
                "tools": [hold],
            }
        ],
    )
    monkeypatch.setattr(runtime, "session_graph", lambda *args, **kwargs: (graph, None))
    run = ActiveRun(queue=asyncio.Queue())
    producer = asyncio.create_task(runtime.produce(session, "Delegate to worker.", run))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if termination == "cancel":
            run.cancel_requested = True
            producer.cancel()
        await asyncio.wait_for(producer, timeout=5)

        events = []
        while not run.queue.empty():
            events.append(run.queue.get_nowait())
        terminal = next(
            event for event in events if event and event.event in {"error", "cancelled"}
        )
        assert terminal.event == ("error" if termination == "timeout" else "cancelled")
        if termination == "timeout":
            assert terminal.data["error_type"] == "TimeoutError"
        assert cancelled.is_set(), "Run ended before cancellation reached its subagent tool"
        assert all(process.returncode is not None for process in processes)
        for process in processes:
            with pytest.raises(ProcessLookupError):
                os.killpg(process.pid, 0)

        calls_at_terminal = list(calls)
        released.set()
        await asyncio.sleep(0.1)
        assert calls == calls_at_terminal, "Subagent called the model after Run termination"
        assert calls == ["parent", "worker"]
        assert run.queue.empty(), "Callbacks published events after the stream ended"
    finally:
        released.set()
        for process in processes:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        # Allow leaked work in a failing regression to settle before fixture shutdown.
        await asyncio.sleep(0.1)
