"""Application boundary: optional OV I/O cannot stop the local Agent loop."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired

from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.errors import (
    ContextBudgetExceeded,
    ContextHistoryBudgetError,
    IdentityMismatchError,
)
from openviking.integrations.langchain.ov_context.reader import RecallResult
from openviking.integrations.langchain.ov_context.retry import retryable_read
from openviking.integrations.langchain.ov_context.state import OVPluginState
from openviking_cli._exception_utils import auth_error

logger = logging.getLogger(__name__)


class _RecallDeadlineReader:
    """Finish optional recall before the enclosing preparation deadline expires."""

    def __init__(
        self, reader, *, retry_attempts=0, retry_delay=1.0, attempt_budget=4.9, observer=None
    ):
        self.reader = reader
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.attempt_budget = attempt_budget
        self.observer = observer
        self.deadline: ContextVar[float | None] = ContextVar("ov_recall_deadline", default=None)
        self.timeout_scope: ContextVar[asyncio.Timeout | None] = ContextVar(
            "ov_recall_timeout_scope", default=None
        )

    def __getattr__(self, name):
        return getattr(self.reader, name)

    async def recall(self, *args, **kwargs):
        deadline = self.deadline.get()
        if deadline is None:
            return await self.reader.recall(*args, **kwargs)
        scope = self.timeout_scope.get()
        for attempt in range(1, self.retry_attempts + 2):
            result = await self._attempt(*args, **kwargs)
            if self.observer:
                self.observer(
                    "memory_recall_attempt",
                    attempt=attempt,
                    max_attempts=self.retry_attempts + 1,
                    status=result.status,
                )
            if result.status != "timeout" or attempt > self.retry_attempts or scope is None:
                return result
            # Extend only the optional read phase. The same Run, identity and query
            # continue; model generation and persistence are never replayed here.
            deadline = asyncio.get_running_loop().time() + self.retry_delay + self.attempt_budget
            scope.reschedule(deadline)
            self.deadline.set(deadline)
            if self.observer:
                self.observer(
                    "memory_recall_retry",
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    max_attempts=self.retry_attempts + 1,
                    reason="timeout",
                    delay_seconds=self.retry_delay,
                )
            await asyncio.sleep(self.retry_delay)

    async def _attempt(self, *args, **kwargs):
        deadline = self.deadline.get()
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        # Leave time for the caller to record a skipped recall and assemble locally.
        budget = remaining - min(0.1, remaining * 0.1)
        task = asyncio.create_task(self.reader.recall(*args, **kwargs))
        try:
            done, _ = await asyncio.wait({task}, timeout=budget)
            result = task.result() if done else RecallResult(status="timeout")
        except Exception as exc:
            if isinstance(exc, IdentityMismatchError) or auth_error(exc):
                raise
            result = RecallResult(status="timeout" if isinstance(exc, TimeoutError) else "error")
        finally:
            if not task.done():
                task.cancel()
            # Do not wait for a cancellation-resistant client or use its late result.
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        if result.status in {"timeout", "error"}:
            logger.warning("OV memory recall attempt returned: %s", result.status)
        return result


class ResilientOVState(OVPluginState):
    ov_failure: NotRequired[dict[str, str]]


class ResilientOVMiddleware(OpenVikingContextMiddleware):
    """Retain native graph history after a failed optional persistence operation.

    Optional recall failure skips only this Run. A persisted failure disables OV
    for this session. A timed-out write may have
    reached OV; automatic replay would risk duplicates. Recovery requires explicit
    reconciliation, while subsequent Chat turns continue from the local checkpoint.
    """

    state_schema = ResilientOVState

    @property
    def name(self):
        # Preserve existing checkpoint node names and restore/update callers.
        return "OpenVikingContextMiddleware"

    def __init__(self, *, io_timeout: float = 5.0, recall_retry_attempts: int = 0, **kwargs: Any):
        if not 0 < io_timeout <= 5:
            raise ValueError("OV I/O timeout must be in (0, 5] seconds")
        if type(recall_retry_attempts) is not int or not 0 <= recall_retry_attempts <= 3:
            raise ValueError("Recall retry attempts must be an integer in [0, 3]")
        # Lost write/commit receipts are not safe to replay automatically. The
        # fail-open boundary persists a marker on the first failure; strict mode
        # keeps the core engine's configurable reconciliation/retry behaviour.
        config = kwargs.get("context_config") or OVContextConfig()
        kwargs["context_config"] = replace(config, retry_attempts=0)
        super().__init__(**kwargs)
        self.io_timeout = io_timeout
        # Reserve cancellation/cleanup time within the external five-second cap.
        self.wait_budget = io_timeout * 0.98
        self.failure: dict[str, str] | None = None
        self._preparation_deadline: ContextVar[asyncio.Timeout | None] = ContextVar(
            "ov_preparation_deadline", default=None
        )
        reader = self._ov_session_engine.context_reader
        self._recall_reader = (
            _RecallDeadlineReader(
                reader,
                retry_attempts=recall_retry_attempts,
                attempt_budget=self.wait_budget,
                observer=self._ov_metrics.emit,
            )
            if reader is not None
            else None
        )
        self._ov_session_engine.context_reader = self._recall_reader

    def _disabled(self, state):
        return self.failure or state.get("ov_failure")

    async def prepare_session(self, operation, state):
        """Apply the optional-I/O boundary before entering the graph as well."""
        if self._disabled(state):
            self.failure = self._disabled(state)
            return
        try:
            async with asyncio.timeout(self.wait_budget):
                await operation()
        except Exception as exc:
            if not retryable_read(exc):
                raise
            self._degrade("session_prepare", exc, state)

    def _degrade(self, stage, exc, state):
        self.failure = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "persistence": "unconfirmed",
        }
        value = state.get("ov_context") or {}
        owner = tuple(value.get("owner", ()))
        self._ov_session_engine.active.pop(owner, None)
        logger.warning("OV degraded at %s: %s", stage, type(exc).__name__)
        try:
            self._ov_metrics.emit(
                "ov.degraded",
                **self.failure,
                timeout_seconds=self.io_timeout,
                wait_budget_seconds=self.wait_budget,
            )
        except Exception:
            pass
        return {"ov_failure": self.failure}

    async def _hook(self, stage, method, state, runtime):
        # Validate ownership before any optional-OV exception boundary.
        self._context(state, runtime)
        if self._disabled(state):
            return {"ov_failure": self._disabled(state)}
        try:
            async with asyncio.timeout(self.wait_budget):
                return await method(state, runtime)
        except Exception as exc:
            if isinstance(
                exc, (IdentityMismatchError, ContextBudgetExceeded, ContextHistoryBudgetError)
            ) or auth_error(exc):
                raise
            return self._degrade(stage, exc, state)

    async def abefore_agent(self, state, runtime):
        return await self._hook("before_agent", super().abefore_agent, state, runtime)

    async def abefore_model(self, state, runtime):
        return await self._hook("before_model", super().abefore_model, state, runtime)

    async def aafter_model(self, state, runtime):
        return await self._hook("after_model", super().aafter_model, state, runtime)

    async def aafter_agent(self, state, runtime):
        return await self._hook("after_agent", super().aafter_agent, state, runtime)

    def _with_failure(self, response):
        update = {"ov_failure": self.failure}
        if isinstance(response, ExtendedModelResponse):
            command = response.command or Command()
            return ExtendedModelResponse(
                response.model_response,
                replace(command, update={**(command.update or {}), **update}),
            )
        if isinstance(response, AIMessage):
            response = ModelResponse(result=[response])
        return ExtendedModelResponse(response, Command(update=update))

    async def _compress_context(self, *args, **kwargs):
        deadline = self._preparation_deadline.get()
        remaining = None
        if deadline is not None and deadline.when() is not None:
            loop = asyncio.get_running_loop()
            remaining = max(0.0, deadline.when() - loop.time())
            # Required archive waiting has its own bounded budget. Individual
            # SDK requests still use the application's short HTTP timeout.
            deadline.reschedule(
                loop.time() + remaining + self._ov_session_engine.config.commit_wait_timeout
            )
        try:
            return await super()._compress_context(*args, **kwargs)
        except asyncio.CancelledError as exc:
            # Only our deadline becomes a budget error; caller cancellation propagates.
            if deadline is not None and deadline.expired():
                raise ContextBudgetExceeded(
                    "Capacity compression exceeded its wait deadline"
                ) from exc
            raise
        finally:
            if deadline is not None and remaining is not None and not deadline.expired():
                deadline.reschedule(asyncio.get_running_loop().time() + remaining)

    async def _local_model_call(self, request, handler):
        from langchain_core.runnables.config import ensure_config, var_child_runnable_config

        assembler = self._ov_session_engine.assembler
        tokens = assembler.check_input_budget(request)
        budget = assembler.budget(request)
        fields = {
            "estimated_input_tokens": tokens,
            "input_limit": budget.input_limit,
            "counter": getattr(assembler.token_counter, "name", "custom"),
            "source": "local_fallback",
        }
        identity = self._identity(request.state, request.runtime)
        self._ov_metrics.emit(
            "context.estimate",
            session_id=identity.session_id,
            run_id=identity.run_id,
            account_id=identity.account_id,
            user_id=identity.user_id,
            actor_peer_id=identity.actor_peer_id,
            **fields,
        )
        config = ensure_config()
        token = var_child_runnable_config.set(
            {**config, "metadata": {**config.get("metadata", {}), "ov_input_budget": fields}}
        )
        try:
            return await handler(request)
        finally:
            var_child_runnable_config.reset(token)

    async def awrap_model_call(self, request, handler):
        self._context(request.state, request.runtime)
        if self._disabled(request.state):
            return await self._local_model_call(request, handler)
        called = False
        model_failed = False
        response = None
        reader = self._recall_reader
        token = None
        scope_token = None
        deadline_token = None
        try:
            async with asyncio.timeout(self.wait_budget) as deadline:
                deadline_token = self._preparation_deadline.set(deadline)
                if reader is not None:
                    token = reader.deadline.set(deadline.when())
                    scope_token = reader.timeout_scope.set(deadline)

                async def downstream(prepared):
                    nonlocal called, model_failed, response
                    called = True
                    # The 5-second deadline covers OV preparation, not model generation.
                    deadline.reschedule(None)
                    if reader is not None:
                        reader.deadline.set(None)
                    try:
                        response = await handler(prepared)
                        return response
                    except Exception:
                        model_failed = True
                        raise

                return await super().awrap_model_call(request, downstream)
        except Exception as exc:
            if (
                model_failed
                or isinstance(
                    exc, (IdentityMismatchError, ContextBudgetExceeded, ContextHistoryBudgetError)
                )
                or auth_error(exc)
            ):
                raise
            self._degrade(
                "prepare_context" if not called else "after_model_handler", exc, request.state
            )
        finally:
            if deadline_token is not None:
                self._preparation_deadline.reset(deadline_token)
            if reader is not None and token is not None:
                reader.deadline.reset(token)
            if reader is not None and scope_token is not None:
                reader.timeout_scope.reset(scope_token)
        if not called:
            response = await self._local_model_call(request, handler)
        return self._with_failure(response)

    async def awrap_tool_call(self, request, handler):
        if not request.tool_call["name"].startswith("openviking_tool_result_"):
            return await handler(request)
        if self._disabled(request.state):
            self.failure = self._disabled(request.state)
        else:
            try:
                async with asyncio.timeout(self.wait_budget):
                    return await handler(request)
            except Exception as exc:
                self._degrade("tool_result_read", exc, request.state)
        message = ToolMessage(
            content="OV暂不可用，本次读取未完成。请使用当前对话中已有的原始工具结果，不要重复读取OV。",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
        return Command(update={"messages": [message], "ov_failure": self.failure})

    async def aclose(self):
        try:
            async with asyncio.timeout(self.wait_budget):
                await super().aclose()
        except Exception as exc:
            logger.warning("OV close failed: %s", type(exc).__name__)
