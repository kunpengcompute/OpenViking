# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run event delivery, durable observations and model usage callbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pprint import pformat
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from examples.deepagents_ov_context.chat.event_log import RunEventLog
from examples.deepagents_ov_context.chat.history import EventKind, RunEvent, RunHistory
from examples.deepagents_ov_context.chat.session import SessionInfo, SessionStore

logger = logging.getLogger(__name__)


def error_stack(error: BaseException) -> str:
    """Keep call locations without source literals or provider error payloads."""
    frames = traceback.extract_tb(error.__traceback__)
    return "\n".join(
        [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames]
        + [type(error).__name__]
    )


@dataclass(frozen=True)
class StreamEvent:
    event: EventKind
    data: dict[str, Any]

    def encode(self, ordinal: int) -> str:
        return f"id: {ordinal}\nevent: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass
class ActiveRun:
    run_id: str = field(default_factory=lambda: "r-" + uuid.uuid4().hex)
    history: RunHistory | None = None
    session_id: str | None = None
    started: float = field(default_factory=time.monotonic)
    recorded: int = 0
    queue: asyncio.Queue[StreamEvent | None] = field(default_factory=lambda: asyncio.Queue(256))
    task: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    event_log: RunEventLog | None = None
    log_context: dict[str, str] = field(default_factory=dict)
    statistics: Any = None
    observations_complete: bool = True

    def bind_log(self, session: SessionInfo, log: RunEventLog) -> None:
        self.event_log = log
        self.log_context = {
            "account_id": session.account_id,
            "user_id": session.user_id,
            "actor_peer_id": session.agent_id,
            "session_id": session.session_id,
            "thread_id": session.thread_id,
        }

    def record(self, event: EventKind, data: dict[str, Any]) -> StreamEvent:
        if event == "token":
            return StreamEvent(event, data)
        self.recorded += 1
        elapsed = round((time.monotonic() - self.started) * 1000)
        logged = False
        if self.event_log is not None:
            event_name = (
                data.get("name", "stage")
                if event == "stage"
                else {
                    "start": "run.start",
                    "done": "run.end",
                    "error": "run.end",
                    "cancelled": "run.end",
                    "statistics": "run.statistics",
                    "message": "message.full",
                }.get(event, event)
            )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event_name,
                "event_kind": event,
                **self.log_context,
                "run_id": self.run_id,
                "sequence": self.recorded,
                "elapsed_ms": elapsed,
                "model_call_id": data.get("model_call_id"),
                "status": data.get("status", data.get("statistics", {}).get("status")),
                "duration_ms": data.get("duration_ms"),
                "data": data,
            }
            try:
                self.event_log.append(record)
                logged = True
                if event in {"done", "error", "cancelled"}:
                    self.event_log.flush()
            except Exception:
                self._incomplete(data)
                logger.warning("chat.event_log_failed", exc_info=True)
        # Token chunks are transport-only; complete model/tool messages are stored.
        if self.history is not None and self.session_id is not None:
            try:
                if logged and self.history.journal is self.event_log:
                    self.history.record_logged_event(final=event in {"done", "error", "cancelled"})
                else:
                    self.history.append(
                        self.session_id,
                        self.run_id,
                        RunEvent(ordinal=self.recorded, event=event, data=data, elapsed_ms=elapsed),
                    )
            except Exception as exc:
                # The journal cursor stays behind a failed projection. A later
                # read/flush or process restart retries the complete suffix.
                self._incomplete(data)
                logger.warning("chat.history_failed: %s", error_stack(exc))
        return StreamEvent(event, data)

    def _incomplete(self, data: dict[str, Any]) -> None:
        self.observations_complete = False
        if self.statistics is not None:
            self.statistics.observations_complete = False
        if isinstance(data.get("statistics"), dict):
            data["statistics"]["observations_complete"] = False

    def emit_metric(self, event: str, fields: dict[str, Any]) -> None:
        # Persist before the best-effort live notification, even if its queue is full.
        item = self.record("stage", {"name": event, **fields})
        with suppress(asyncio.QueueFull):
            self.queue.put_nowait(item)

    async def emit(self, event: EventKind, **data: Any) -> None:
        item = self.record(event, data)
        if self.cancel_requested:
            with suppress(asyncio.QueueFull):
                self.queue.put_nowait(item)
        else:
            await self.queue.put(item)


_active_run: ContextVar[ActiveRun | None] = ContextVar("chat_active_run", default=None)


class InputTrace(AsyncCallbackHandler):
    def __init__(self, run: ActiveRun, session: SessionInfo, store: SessionStore) -> None:
        self.run = run
        self.session, self.store = session, store
        self.started: set[uuid.UUID] = set()
        self.ended: set[uuid.UUID] = set()
        self.tool_started: dict[uuid.UUID, tuple[float, str, str | None]] = {}
        self.reasoning_parts: dict[uuid.UUID, list[str]] = {}
        self.reasoning_encrypted: set[uuid.UUID] = set()
        self.benchmark_audit = None
        self.benchmark_capture_lock = asyncio.Lock()

    async def capture_benchmark(self, event: str, name: str, call_id: str | None) -> None:
        if self.benchmark_audit is not None:
            async with self.benchmark_capture_lock:
                capture = asyncio.create_task(
                    asyncio.to_thread(
                        self.benchmark_audit.capture,
                        round(time.monotonic() - self.run.started, 3),
                        event,
                        tool=name,
                        tool_call_id=call_id,
                    )
                )
                try:
                    await asyncio.shield(capture)
                except asyncio.CancelledError:
                    await asyncio.shield(capture)
                    raise

    async def publish(self) -> None:
        assert self.session.last_run is not None
        self.session.last_run.observations_complete &= self.run.observations_complete
        self.store.save(self.session)
        await self.run.emit("statistics", statistics=self.session.last_run.model_dump())

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        assert self.session.last_run is not None
        if run_id in self.started:
            return
        self.started.add(run_id)
        self.session.last_run.model_calls += 1
        await self.publish()
        self.run.emit_metric("model.start", {"model_call_id": str(run_id)})
        budget = (kwargs.get("metadata") or {}).get("ov_input_budget", {})
        for batch in messages:
            await self.run.emit(
                "model_input",
                messages=[message.model_dump(mode="json") for message in batch],
                message_reprs=[pformat(message, width=100, sort_dicts=False) for message in batch],
                model_call_id=str(run_id),
                **budget,
            )

    @staticmethod
    def reasoning_fields(message: BaseMessage) -> tuple[str, bool]:
        metadata = message.response_metadata
        reasoning = message.additional_kwargs.get("reasoning_content") or metadata.get(
            "observed_reasoning_content"
        )
        encrypted = bool(
            message.additional_kwargs.get("encrypted_content")
            or metadata.get("observed_encrypted_content_present")
        )
        if not isinstance(reasoning, str):
            reasoning = ""
        return reasoning, encrypted

    async def on_llm_new_token(
        self, token: str, *, run_id: uuid.UUID, chunk: Any = None, **kwargs: Any
    ) -> None:
        if run_id in self.ended or not isinstance(chunk, ChatGeneration):
            return
        reasoning, encrypted = self.reasoning_fields(chunk.message)
        if reasoning:
            self.reasoning_parts.setdefault(run_id, []).append(reasoning)
        if encrypted:
            self.reasoning_encrypted.add(run_id)

    def record_reasoning(self, response: LLMResult | None, run_id: uuid.UUID, status: str) -> None:
        # ainvoke's error callback can omit partial generations. Retain streamed
        # observations until the call ends, including an idle timeout/cancellation.
        reasoning = "".join(self.reasoning_parts.pop(run_id, []))
        encrypted = run_id in self.reasoning_encrypted
        self.reasoning_encrypted.discard(run_id)
        metadata: dict[str, Any] = {}
        if response and response.generations and response.generations[0]:
            generation = response.generations[0][0]
            if isinstance(generation, ChatGeneration):
                text, present = self.reasoning_fields(generation.message)
                reasoning = text or reasoning
                encrypted |= present
                metadata = generation.message.response_metadata
        if reasoning or encrypted:
            self.run.emit_metric(
                "model.reasoning",
                {
                    "model_call_id": str(run_id),
                    "status": status,
                    "reasoning_content": reasoning,
                    "characters": len(reasoning),
                    "encrypted_content_present": encrypted,
                    "finish_reason": metadata.get("finish_reason"),
                },
            )

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        if run_id in self.ended:
            return
        self.ended.add(run_id)
        self.record_reasoning(response, run_id, "complete")
        assert self.session.last_run is not None
        stats = self.session.last_run
        if run_id not in self.started:
            self.started.add(run_id)
            stats.model_calls += 1
        # The standard callback returns the merged message. Never sum streamed
        # usage chunks and then count this complete message again.
        generation = (
            response.generations[0][0] if response.generations and response.generations[0] else None
        )
        usage = (
            generation.message.usage_metadata
            if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage)
            else None
        )
        if usage is not None:
            stats.input_tokens = (stats.input_tokens or 0) + usage["input_tokens"]
            stats.output_tokens = (stats.output_tokens or 0) + usage["output_tokens"]
            stats.total_tokens = (stats.total_tokens or 0) + usage["total_tokens"]
            stats.usage_calls += 1
        self.run.emit_metric(
            "model_usage",
            {
                "model_call_id": str(run_id),
                "status": "success" if usage is not None else "unknown",
                **(usage or {}),
            },
        )
        await self.publish()

    async def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        if run_id in self.ended:
            return
        self.ended.add(run_id)
        self.record_reasoning(kwargs.get("response"), run_id, "partial")
        self.run.emit_metric(
            "model.error",
            {
                "model_call_id": str(run_id),
                "status": "error",
                "error_type": type(error).__name__,
                "error_stack": error_stack(error),
            },
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or serialized.get("name", "unknown")
        call_id = kwargs.get("tool_call_id")
        self.tool_started[run_id] = (time.monotonic(), name, call_id)
        self.run.emit_metric(
            "tool.start",
            {
                "tool_run_id": str(run_id),
                "tool_call_id": call_id,
                "tool": name,
                "arguments": inputs if inputs is not None else input_str,
            },
        )

    async def on_tool_end(self, output: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        started, name, call_id = self.tool_started.pop(run_id, (time.monotonic(), "unknown", None))
        # Capture the tool's own output before later context middleware shortens its view.
        raw = output.model_dump(mode="json") if isinstance(output, BaseMessage) else output
        raw = json.loads(json.dumps(raw, ensure_ascii=False, default=str))
        self.run.emit_metric(
            "tool.end",
            {
                "tool_run_id": str(run_id),
                "tool_call_id": call_id,
                "tool": name,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "status": getattr(output, "status", "success"),
                "output": raw,
            },
        )
        # Snapshot only; business/semantic grading runs after the Agent is terminal.
        await self.capture_benchmark("tool.end", name, call_id)

    async def on_tool_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        started, name, call_id = self.tool_started.pop(run_id, (time.monotonic(), "unknown", None))
        self.run.emit_metric(
            "tool.error",
            {
                "tool_run_id": str(run_id),
                "tool_call_id": call_id,
                "tool": name,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "status": "error",
                "error_type": type(error).__name__,
                "error_stack": error_stack(error),
            },
        )
        await self.capture_benchmark("tool.error", name, call_id)
