# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session operations shared by middleware hooks and tool-result readers."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter, time
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import BaseMessage, ToolMessage

from openviking.integrations.langchain.client import OpenVikingConnection

from .assembler import ContextAssembler, ContextRefreshPlan, is_summary, validate_tool_pairs
from .budget import TokenCounter
from .commit import CommitTrigger
from .config import OVContextConfig
from .errors import (
    ContextBudgetExceeded,
    ContextHistoryBudgetError,
    ContextReadError,
    IdentityMismatchError,
)
from .identity import OVIdentity, SessionKey
from .metrics import Metrics
from .reader import ContextReaderBase, ContextReadResult, ContextSnapshot
from .retry import (
    retry_call,
    retryable_commit,
    retryable_read,
    retryable_write,
)
from .sdk import AsyncClientHandle, already_exists, auth_error, not_found
from .state import OVContext, PendingCommit
from .writer import MessageWriterBase, WriteResult

if TYPE_CHECKING:
    from .commit import OVCommitPolicy


@dataclass(frozen=True)
class _CommitProgress:
    pending: PendingCommit | None
    completed_runs: int
    commit_count: int | None


@dataclass
class _RefreshProgress:
    plan: ContextRefreshPlan
    write_index: int
    appended: list[BaseMessage] = field(default_factory=list)
    target_archive_id: str | None = None


@dataclass(frozen=True)
class _TokenReceipt:
    run_id: str
    write_index: int
    pending_tokens: int


class SessionEngine:
    """Shared execution services, not a second copy of session messages."""

    def __init__(
        self,
        connection: OpenVikingConnection,
        config: OVContextConfig,
        token_counter: TokenCounter | None,
        memory_policy: dict[str, Any] | None,
        *,
        message_writer: MessageWriterBase | None = None,
        context_reader: ContextReaderBase | None = None,
        session_commit_policy: OVCommitPolicy | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.client = AsyncClientHandle(connection)
        self.context_reader = context_reader
        self.config = config
        self.writer = message_writer
        self.assembler = ContextAssembler(config, token_counter)
        self.memory_policy = memory_policy
        self.session_commit_policy = session_commit_policy
        self.metrics = metrics or Metrics()
        self.active: dict[SessionKey, str] = {}
        # Side effects survive a failed model node until a Hook/cleanup saves them.
        # This holds only task metadata, never a second message cache.
        self._commit_progress: dict[SessionKey, _CommitProgress] = {}
        # Keep only the selected suffix and confirmed post-submission increments.
        self._refresh_progress: dict[SessionKey, _RefreshProgress] = {}
        # Invocation-local, never serialized into graph checkpoints. The host
        # still owns same-session Run serialization and external reconciliation.
        self._token_receipts: dict[SessionKey, _TokenReceipt] = {}

    def invalidate_tokens(self, context: OVContext) -> None:
        self._token_receipts.pop(context.session_key, None)

    async def _call_read(self, method: str, **kwargs: Any) -> Any:
        """Idempotent OV read carrying the configured bounded retry."""
        return await retry_call(
            lambda: self.client.call(method, **kwargs),
            retries=self.config.retry_attempts,
        )

    def recover_commit(self, context: OVContext, *, persisted: bool = False) -> None:
        """Recover task metadata; persisted=True marks a snapshot read from graph state."""
        refresh = self._refresh_progress.get(context.session_key)
        if (
            refresh is not None
            and persisted
            and (
                context.run_id != refresh.plan.run_id
                or context.archive_id != refresh.plan.source_archive_id
            )
        ):
            self._refresh_progress.pop(context.session_key, None)
        progress = self._commit_progress.get(context.session_key)
        if progress is None:
            return
        if persisted and (
            context.pending_commit == progress.pending
            and context.completed_runs == progress.completed_runs
            and context.commit_count == progress.commit_count
        ):
            del self._commit_progress[context.session_key]
            return
        if (
            context.pending_commit != progress.pending
            or context.commit_count != progress.commit_count
        ):
            context.summary_check_countdown = 0
        context.pending_commit = progress.pending.copy() if progress.pending is not None else None
        context.completed_runs = progress.completed_runs
        context.commit_count = progress.commit_count

    def _remember_commit(self, context: OVContext) -> None:
        self._commit_progress[context.session_key] = _CommitProgress(
            context.pending_commit.copy() if context.pending_commit is not None else None,
            context.completed_runs,
            context.commit_count,
        )

    def observe_commit_count(self, context: OVContext, count: int | None) -> None:
        """Account for a newly observed commit once, using an existing response."""
        if count is None:
            return
        previous = context.commit_count
        if previous is None:
            previous = self._archive_sequence(context.archive_id) or 0
        if count > previous:
            context.completed_runs = 0
            context.summary_check_countdown = 0
        context.commit_count = max(previous, count)

    @staticmethod
    def _archive_sequence(archive_id: str | None) -> int | None:
        if archive_id is None:
            return 0
        match = re.fullmatch(r"archive_(\d+)", archive_id)
        return int(match[1]) if match else None

    @staticmethod
    def _remember_auto_commit_policy(context: OVContext, meta: dict[str, Any]) -> None:
        if "auto_commit_policy" not in meta or meta.get("commit_count") is None:
            context.auto_commit_enabled = None
        else:
            policy = meta["auto_commit_policy"]
            # Unknown/malformed policies stay conservative; only explicit null
            # or an all-zero policy confirms that the server won't auto-commit.
            context.auto_commit_enabled = policy is not None and (
                not isinstance(policy, dict)
                or not policy
                or any(
                    policy.get(key, 0) != 0
                    for key in (
                        "pending_token_threshold",
                        "message_count_threshold",
                        "idle_timeout_seconds",
                    )
                )
            )

    def _probe_metadata_known(self, context: OVContext) -> bool:
        return (
            context.commit_count is not None
            and context.auto_commit_enabled is not None
            and self._archive_sequence(context.archive_id) is not None
        )

    def _unapplied_commits(self, context: OVContext) -> bool:
        applied = self._archive_sequence(context.archive_id)
        return (
            applied is not None
            and context.commit_count is not None
            and context.commit_count > applied
        )

    def _cached_tokens(self, context: OVContext) -> int | None:
        receipt = self._token_receipts.get(context.session_key)
        run_id = context.run_id or self.active.get(context.session_key)
        if (
            receipt is not None
            and run_id == receipt.run_id
            and context.write_index == receipt.write_index
            and context.pending_commit is None
        ):
            return receipt.pending_tokens
        return None

    async def maybe_check_summary(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
        model_request: ModelRequest[Any] | None = None,
    ) -> ContextReadResult:
        """Probe only outstanding/automatic commits, retaining legacy fallbacks."""
        self._validate_identity(context, identity)
        self.recover_commit(context)
        if context.context_status == "reload_required" or not self._probe_metadata_known(context):
            return await self.check_summary(
                context, identity=identity, token_budget=token_budget, model_request=model_request
            )
        outstanding = bool(context.pending_commit) or self._unapplied_commits(context)
        tokens = self._cached_tokens(context)
        # Without a write/read receipt, policy checks still need fresh counts.
        if not outstanding and tokens is None:
            return await self.check_summary(
                context, identity=identity, token_budget=token_budget, model_request=model_request
            )
        if not outstanding and not context.auto_commit_enabled:
            return ContextReadResult(context.archive_id, tokens, probe_performed=False)
        if context.summary_check_countdown > 0:
            context.summary_check_countdown -= 1
            return ContextReadResult(context.archive_id, tokens, probe_performed=False)
        return await self.check_summary(
            context, identity=identity, token_budget=token_budget, model_request=model_request
        )

    @staticmethod
    def _session_commit_count(meta: object, context: OVContext) -> int | None:
        if not isinstance(meta, dict) or meta.get("session_id") != context.session_id:
            raise ContextReadError("OV returned a different session in metadata")
        count = meta.get("commit_count")
        if count is not None and (type(count) is not int or count < 0):
            raise ContextReadError("Invalid session commit_count")
        if count is not None and context.commit_count is not None and count < context.commit_count:
            raise ContextReadError("Session commit_count regressed")
        return count

    async def _compression_failed(self, context: OVContext, *, identity: OVIdentity) -> bool:
        pending = context.pending_commit
        if pending is None or not pending.get("task_id"):
            return False
        task = await self._call_read("get_task", identity=identity, task_id=pending["task_id"])
        if task and task.get("status") in ("failed", "cancelled", "canceled"):
            context.pending_commit = None
            self._remember_commit(context)
            return True
        return False

    def require_io(self) -> None:
        missing = [
            name
            for name, value in (
                ("message_writer", self.writer),
                ("context_reader", self.context_reader),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Async OV context requires injected " + ", ".join(missing))
        for method in ("exchange_context", "recall"):
            if not callable(getattr(self.context_reader, method, None)):
                raise TypeError(f"context_reader must implement {method}()")

    @staticmethod
    def _validate_identity(context: OVContext, identity: OVIdentity) -> None:
        if context.owner != identity.session_key(context.session_id):
            raise IdentityMismatchError("Invocation identity does not match OVContext owner")

    @contextmanager
    def observe(
        self, event: str, context: OVContext | None = None, **fields: Any
    ) -> Iterator[dict[str, Any]]:
        """Emit a content-free sample on success, failure or cancellation."""
        start = perf_counter()
        sample = {
            "session_id": context.session_id if context else None,
            "run_id": context.run_id if context else None,
            "actor_peer_id": context.identity.actor_peer_id if context else None,
            "account_id": context.identity.account_id if context else None,
            "user_id": context.identity.user_id if context else None,
            "archive_id": context.archive_id if context else None,
            "task_id": context.pending_commit.get("task_id")
            if context and context.pending_commit
            else None,
            "status": "ok",
            **fields,
        }
        try:
            yield sample
        except BaseException as exc:
            sample["status"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            sample["error_type"] = type(exc).__name__
            sample["error_stack"] = traceback.format_exc()
            raise
        finally:
            if self.metrics:
                self.metrics.emit(event, duration_ms=(perf_counter() - start) * 1000, **sample)

    async def ensure_session(
        self, sid: str, *, identity: OVIdentity, run_id: str | None = None
    ) -> dict[str, Any]:
        with self.observe(
            "session.ensure",
            session_id=sid,
            run_id=run_id,
            account_id=identity.account_id,
            user_id=identity.user_id,
            actor_peer_id=identity.actor_peer_id,
        ):
            return await retry_call(
                lambda: self._ensure_session_once(sid, identity=identity),
                retries=self.config.retry_attempts,
            )

    async def _ensure_session_once(self, sid: str, *, identity: OVIdentity) -> dict[str, Any]:
        try:
            session = await self.client.call(
                "get_session", identity=identity, session_id=sid, auto_create=False
            )
        except Exception as exc:
            if not not_found(exc):
                raise
            try:
                session = await self.client.call(
                    "create_session",
                    identity=identity,
                    session_id=sid,
                    memory_policy=self.memory_policy,
                )
            except Exception as create_error:
                if not already_exists(create_error):
                    raise
                session = await self.client.call(
                    "get_session", identity=identity, session_id=sid, auto_create=False
                )
        if not isinstance(session, dict) or session.get("session_id") != sid:
            raise ValueError("OV returned a different session")
        return session

    async def refresh(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        initial: bool = False,
        run_id: str | None = None,
        token_budget: int | None = None,
        expected_archive_id: str | None = None,
        plan: ContextRefreshPlan | None = None,
    ) -> OVContext:
        """Read and build an independent candidate; admission precedes publication."""
        self._validate_identity(context, identity)
        self.invalidate_tokens(context)
        reader = self.context_reader
        if reader is None:
            raise ValueError("Async OV context requires injected context_reader")
        try:
            event = "context.initial_load" if initial else "context.refresh_read"
            with self.observe(event, context, force=initial, run_id=run_id or context.run_id):
                snapshot = await retry_call(
                    lambda: reader.exchange_context(
                        session_id=context.session_id,
                        identity=identity,
                        token_budget=token_budget,
                    ),
                    retries=self.config.retry_attempts,
                )
                if expected_archive_id is not None:
                    self._validate_boundary(expected_archive_id, snapshot.archive_id)
                    if snapshot.archive_id != expected_archive_id and (
                        self._archive_sequence(expected_archive_id) is None
                        or self._archive_sequence(snapshot.archive_id) is None
                    ):
                        raise ContextReadError(
                            "Context terminal boundary does not match expected_archive_id"
                        )
                return self._candidate_snapshot(context, snapshot, plan=plan)
        except Exception as exc:
            if (
                isinstance(exc, (IdentityMismatchError, ContextHistoryBudgetError))
                or auth_error(exc)
                or not initial
                or self.config.context_failure == "raise"
            ):
                raise
            candidate = OVContext.from_state(context)
            candidate.context_status = "reload_required"
            return candidate

    async def write(
        self,
        context: OVContext,
        raw_messages: Sequence[BaseMessage],
        peer_id: str | None = None,
        *,
        identity: OVIdentity,
    ) -> WriteResult | None:
        self._validate_identity(context, identity)
        self.recover_commit(context)
        writer = self.writer
        if writer is None:
            raise ValueError("Async OV context requires injected message_writer")
        if not 0 <= context.write_index <= len(raw_messages):
            raise ValueError("Raw message cursor regressed; restore the original graph history")
        if context.write_index == len(raw_messages):
            return None
        self.invalidate_tokens(context)
        with self.observe("message.write", context) as sample:
            try:
                result = await retry_call(
                    lambda: writer.write(
                        context.session_id,
                        list(raw_messages),
                        context.write_index,
                        context.run_id or "session",
                        identity=identity,
                        peer_id=peer_id,
                    ),
                    retries=self.config.retry_attempts,
                    retry_on=retryable_write,
                )
            except Exception as exc:
                if retryable_write(exc):
                    # Zero confirmed writes plus a pre-send failure proves that
                    # the saved cursor is still valid after retry exhaustion.
                    exc.ov_state_consistent = True  # type: ignore[attr-defined]
                raise
            if not isinstance(result, WriteResult):
                raise TypeError("MessageWriterBase.write must return WriteResult")
            if result.write_index != len(raw_messages):
                raise ValueError("Writer must return the end cursor of the full message batch")
            sample["message_count"] = result.written_count
        progress = self._refresh_progress.get(context.session_key)
        if progress is not None and progress.plan.run_id == context.run_id:
            self.assembler._apply_delta(progress.appended, result.model_messages)
            progress.write_index = result.write_index
        if not result.model_messages:
            context.write_index = result.write_index
            context.last_activity = time()
            return result
        by_id = {message.id: i for i, message in enumerate(context.messages)}
        by_call = {
            message.tool_call_id: i
            for i, message in enumerate(context.messages)
            if isinstance(message, ToolMessage)
        }
        for message in result.model_messages:
            index = by_id.get(message.id)
            if index is None and isinstance(message, ToolMessage):
                # Match a changed native ID without duplicating the tool result.
                # Writer owns the entire replacement, including its latest view.
                index = by_call.get(message.tool_call_id)
            if index is not None:
                context.messages[index] = message
            else:
                index = len(context.messages)
                context.messages.append(message)
            by_id[message.id] = index
            if isinstance(message, ToolMessage):
                by_call[message.tool_call_id] = index
        context.write_index = result.write_index
        context.last_activity = time()
        if context.run_id and type(result.pending_tokens) is int and result.pending_tokens >= 0:
            self._token_receipts[context.session_key] = _TokenReceipt(
                context.run_id, result.write_index, result.pending_tokens
            )
        return result

    async def check_summary(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
        model_request: ModelRequest[Any] | None = None,
    ) -> ContextReadResult:
        self._validate_identity(context, identity)
        self.invalidate_tokens(context)
        self.recover_commit(context)
        old_archive = context.archive_id
        old_pending = context.pending_commit
        old_count = context.commit_count
        with self.observe(
            "context.refresh", context, force=context.context_status == "reload_required"
        ) as sample:
            result = await self._check_summary(
                context, identity=identity, token_budget=token_budget, model_request=model_request
            )
            context.summary_check_countdown = self.config.summary_check_interval - 1
            if context.run_id and type(result.pending_tokens) is int:
                self._token_receipts[context.session_key] = _TokenReceipt(
                    context.run_id, context.write_index, result.pending_tokens
                )
            if old_pending != context.pending_commit or (
                old_count != context.commit_count and context.session_key in self._commit_progress
            ):
                self._remember_commit(context)
            sample.update(
                version_changed=old_archive != context.archive_id,
                pending_tokens=result.pending_tokens,
                archive_id=context.archive_id,
            )
            return result

    async def _check_summary(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
        model_request: ModelRequest[Any] | None = None,
    ) -> ContextReadResult:
        reader = self.context_reader
        if reader is None:
            raise ValueError("Async OV context requires injected context_reader")
        meta = await self._call_read(
            "get_session",
            identity=identity,
            session_id=context.session_id,
            auto_create=False,
        )
        if not isinstance(meta, dict):
            raise ContextReadError("Session metadata must be a dictionary")
        if "commit_count" in meta:
            count = self._session_commit_count(meta, context)
            if count is not None:
                self.observe_commit_count(context, count)
        self._remember_auto_commit_policy(context, meta)
        pending_tokens = meta.get("pending_tokens")
        if pending_tokens is not None and (type(pending_tokens) is not int or pending_tokens < 0):
            raise ContextReadError("Session pending_tokens must be a non-negative integer or None")
        advertised = "latest_finished_archive_id" in meta
        boundary = meta.get("latest_finished_archive_id")
        if advertised:
            self._validate_boundary(context.archive_id, boundary)
        ready = advertised and boundary != context.archive_id
        expected = boundary if advertised else None
        pending = context.pending_commit
        finished = False
        failed = False
        if pending and pending["task_id"]:
            task = await self._call_read("get_task", identity=identity, task_id=pending["task_id"])
            status = task.get("status") if isinstance(task, dict) else None
            if status in ("cancelled", "canceled"):
                context.pending_commit = None
                self._remember_commit(context)
                raise ContextBudgetExceeded("OV compression task did not complete")
            finished = status in ("completed", "done", "failed")
            failed = status == "failed"
            if finished:
                result = task.get("result") or {}
                target = self._target_archive(result) or pending.get("target_archive_id")
                if target is not None:
                    target_seq = self._archive_sequence(target)
                    expected_seq = self._archive_sequence(expected)
                    if (
                        expected is None
                        or target_seq is None
                        or expected_seq is None
                        or target_seq > expected_seq
                    ):
                        expected = target
                    progress = self._refresh_progress.get(context.session_key)
                    if progress is not None and progress.target_archive_id is None:
                        progress.target_archive_id = target
        if ready or finished or context.context_status == "reload_required":
            if failed:
                context.pending_commit = None
                self._refresh_progress.pop(context.session_key, None)
                self._remember_commit(context)
                raise ContextBudgetExceeded("OV compression task did not complete")
            try:
                candidate = await self.refresh(
                    context,
                    identity=identity,
                    expected_archive_id=expected,
                    token_budget=token_budget,
                )
                self.apply_context(
                    context,
                    candidate,
                    finished_commit=finished and context.pending_commit == pending,
                    model_request=model_request,
                )
            except ContextBudgetExceeded:
                raise
            except Exception as exc:
                if isinstance(exc, IdentityMismatchError) or auth_error(exc):
                    raise
                if context.session_key in self._refresh_progress or finished:
                    raise ContextBudgetExceeded(
                        f"Capacity compression failed ({type(exc).__name__})"
                    ) from exc
                raise
        return ContextReadResult(context.archive_id, pending_tokens)

    @staticmethod
    def _validate_boundary(current: str | None, incoming: str | None) -> None:
        if incoming is not None and (not isinstance(incoming, str) or not incoming.strip()):
            raise ContextReadError("Invalid context terminal boundary")
        if current is not None and incoming is None:
            raise ContextReadError("OV archive version regressed")
        # Native archive_N IDs have numeric ordering; opaque IDs are only checked
        # against an explicitly advertised/finished target, never lexicographically.
        old = re.fullmatch(r"archive_(\d+)", current or "")
        new = re.fullmatch(r"archive_(\d+)", incoming or "")
        if old and new and int(new[1]) < int(old[1]):
            raise ContextReadError("OV archive version regressed")

    @staticmethod
    def _target_archive(result: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        target = result.get("archive_id")
        if target is None and isinstance(result.get("archive_uri"), str):
            target = result["archive_uri"].rstrip("/").rsplit("/", 1)[-1]
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ContextReadError("Invalid commit target archive ID")
        return target

    def _candidate_snapshot(
        self,
        context: OVContext,
        snapshot: ContextSnapshot,
        *,
        plan: ContextRefreshPlan | None = None,
    ) -> OVContext:
        self._validate_boundary(context.archive_id, snapshot.archive_id)
        progress = self._refresh_progress.get(context.session_key)
        if progress is not None and (
            progress.plan.run_id != context.run_id
            or progress.plan.source_archive_id != context.archive_id
        ):
            self._refresh_progress.pop(context.session_key, None)
            progress = None
        if progress is not None and snapshot.archive_id != context.archive_id:
            target = progress.target_archive_id or (context.pending_commit or {}).get(
                "target_archive_id"
            )
            if target is None or snapshot.archive_id != target:
                # A local tail belongs only to its confirmed target. An unknown
                # or different target must use the complete server view, while
                # preserving upstream's requirement for a usable compression.
                if snapshot.archive_id == progress.plan.source_archive_id or not any(
                    is_summary(message) and message.content for message in snapshot.messages
                ):
                    raise ContextReadError("Compression did not produce a usable summary")
                progress = None
        if plan is None and progress is not None and snapshot.archive_id != context.archive_id:
            plan = progress.plan
        appended: Sequence[BaseMessage] = ()
        if plan is not None and progress is not None:
            if progress.write_index != context.write_index:
                raise ContextReadError("Refresh increment does not cover the written cursor")
            appended = progress.appended
        return self.assembler.refresh(context, snapshot, plan=plan, appended_messages=appended)

    def _apply_snapshot(
        self,
        context: OVContext,
        snapshot: ContextSnapshot,
        *,
        force: bool = False,
        finished_commit: bool = False,
        model_request: ModelRequest[Any] | None = None,
    ) -> None:
        self._validate_boundary(context.archive_id, snapshot.archive_id)
        if not force and snapshot.archive_id == context.archive_id:
            self._acknowledge_snapshot(context, snapshot.archive_id, finished_commit)
            return
        candidate = self._candidate_snapshot(context, snapshot)
        self.apply_context(
            context, candidate, finished_commit=finished_commit, model_request=model_request
        )

    def apply_context(
        self,
        context: OVContext,
        candidate: OVContext,
        *,
        finished_commit: bool = False,
        model_request: ModelRequest[Any] | None = None,
    ) -> None:
        """Publish only a structurally valid, admitted candidate cache."""
        self._validate_identity(context, candidate.identity)
        self._validate_boundary(context.archive_id, candidate.archive_id)
        self.assembler._validate_messages(candidate.messages)
        if model_request is not None:
            self.assembler.prepare(model_request, candidate)
        self.invalidate_tokens(context)
        old_archive = context.archive_id
        before_chars = self._content_chars(context.messages)
        before_count = len(context.messages)
        context.replace(candidate.messages, candidate.archive_id)
        if context.commit_count is None:
            context.commit_count = self._archive_sequence(old_archive)
        context.context_status = candidate.context_status
        if self.metrics and candidate.archive_id != old_archive and before_chars:
            after_chars = self._content_chars(context.messages)
            self.metrics.emit(
                "context.compression",
                account_id=context.identity.account_id,
                user_id=context.identity.user_id,
                actor_peer_id=context.identity.actor_peer_id,
                session_id=context.session_id,
                run_id=context.run_id,
                task_id=context.pending_commit.get("task_id") if context.pending_commit else None,
                previous_archive_id=old_archive,
                archive_id=context.archive_id,
                before_message_count=before_count,
                after_message_count=len(context.messages),
                before_chars=before_chars,
                after_chars=after_chars,
                compression_ratio=1 - after_chars / before_chars,
                measurement="cached_message_content_chars",
                status="ok",
            )
        self._acknowledge_snapshot(context, candidate.archive_id, finished_commit)

    def _acknowledge_snapshot(
        self, context: OVContext, archive_id: str | None, finished_commit: bool
    ) -> None:
        # Called only after replacement succeeds, or for an already applied version.
        sequence = self._archive_sequence(archive_id)
        self.observe_commit_count(context, sequence)
        pending = context.pending_commit
        target = pending.get("target_archive_id") if pending else None
        target_sequence = self._archive_sequence(target)
        covers_target = target is not None and (
            archive_id == target
            or (
                sequence is not None and target_sequence is not None and sequence >= target_sequence
            )
        )
        if (
            pending
            and archive_id != pending["archive_id"]
            and (
                covers_target
                or (
                    pending.get("target_archive_id") is None
                    and (finished_commit or pending["task_id"] is None)
                )
            )
        ):
            self.metrics.emit(
                "context.commit_result",
                account_id=context.identity.account_id,
                user_id=context.identity.user_id,
                actor_peer_id=context.identity.actor_peer_id,
                session_id=context.session_id,
                run_id=context.run_id,
                task_id=pending.get("task_id"),
                archive_id=context.archive_id,
                status="archived",
            )
            context.pending_commit = None

    @staticmethod
    def _content_chars(messages: Sequence[BaseMessage]) -> int:
        return sum(
            len(message.content)
            if isinstance(message.content, str)
            else len(json.dumps(message.content, ensure_ascii=False))
            for message in messages
        )

    _is_summary = staticmethod(is_summary)

    def retained_tail(self, context: OVContext, keep: int | None = None) -> tuple[int, int]:
        plan = self.assembler.plan_refresh(context, keep_recent_count=keep or 0)
        return plan.keep_recent_count, plan.archive_message_count

    async def pending_tokens(self, context: OVContext, *, identity: OVIdentity) -> int | None:
        """Reuse this Run's acknowledged snapshot, otherwise read current counts."""

        self._validate_identity(context, identity)
        if self._probe_metadata_known(context) and (
            context.pending_commit or self._unapplied_commits(context)
        ):
            return None
        cached = self._cached_tokens(context)
        if cached is not None:
            return cached
        self.invalidate_tokens(context)
        session = await self._call_read(
            "get_session",
            identity=identity,
            session_id=context.session_id,
            auto_create=False,
        )
        if not isinstance(session, Mapping) or session.get("session_id") != context.session_id:
            raise ContextReadError("OV returned a different session while reading pending tokens")
        if "commit_count" in session:
            self.observe_commit_count(context, self._session_commit_count(session, context))
            if self._unapplied_commits(context):
                return None
        pending_tokens = session.get("pending_tokens")
        if pending_tokens is not None and (
            isinstance(pending_tokens, bool)
            or not isinstance(pending_tokens, int)
            or pending_tokens < 0
        ):
            raise ContextReadError("Invalid session pending_tokens")
        return pending_tokens

    async def commit(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        pending_tokens: int | None = None,
        trigger: CommitTrigger = "policy",
        keep_recent_count: int | None = None,
    ) -> bool:
        """Commit with configured compression retention unless explicitly overridden."""
        self._validate_identity(context, identity)
        with self.observe("context.commit", context, trigger=trigger) as sample:
            submitted = await self._commit(
                context,
                identity=identity,
                pending_tokens=pending_tokens,
                trigger=trigger,
                keep_recent_count=keep_recent_count,
            )
            pending = context.pending_commit
            sample.update(
                status="accepted" if submitted else "skipped",
                task_id=pending.get("task_id") if pending else None,
                target_archive_id=pending.get("target_archive_id") if pending else None,
                archive_id=context.archive_id,
                pending_tokens=pending_tokens,
            )
            progress = self._refresh_progress.get(context.session_key)
            if submitted and progress is not None:
                sample.update(
                    ov_keep_recent_count=progress.plan.keep_recent_count,
                    archive_message_count=progress.plan.archive_message_count,
                    retained_native_message_count=len(progress.plan.recent_messages),
                )
            return submitted

    async def _commit(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        pending_tokens: int | None,
        trigger: CommitTrigger,
        keep_recent_count: int | None,
    ) -> bool:
        self.recover_commit(context)
        if self._probe_metadata_known(context) and (
            context.pending_commit or self._unapplied_commits(context)
        ):
            return True
        if context.pending_commit and not await self._compression_failed(
            context, identity=identity
        ):
            return True
        try:
            validate_tool_pairs([m for m in context.messages if not self._is_summary(m)])
        except ValueError:
            return False
        policy = self.session_commit_policy
        if keep_recent_count is None:
            keep_recent_count = policy.config.keep_recent_count if policy else 0
        plan = self.assembler.plan_refresh(context, keep_recent_count=keep_recent_count)
        if plan.archive_message_count == 0:
            return False
        if policy and pending_tokens is None:
            return False
        if policy and context.session_key in self._token_receipts:
            # A receipt may rule out a commit; actual submission rechecks the
            # server because an independent auto-commit may have changed it.
            decision = vars(type(policy)).get("commit_reason")
            if inspect.isfunction(decision) and not inspect.iscoroutinefunction(decision):
                if (
                    decision(policy, pending_tokens, context.completed_runs, trigger=trigger)
                    is None
                ):
                    return False
            self.invalidate_tokens(context)
            pending_tokens = await self.pending_tokens(context, identity=identity)
            if pending_tokens is None:
                return False
        self._refresh_progress[context.session_key] = _RefreshProgress(plan, context.write_index)
        result = await retry_call(
            lambda: self._send_commit(
                context,
                identity=identity,
                pending_tokens=pending_tokens,
                trigger=trigger,
                policy=policy,
                keep=plan.keep_recent_count,
            ),
            retries=self.config.retry_attempts,
            retry_on=retryable_commit,
        )
        if result.get("archived") is False:
            self._refresh_progress.pop(context.session_key, None)
            return False
        if result.get("task_id") or result.get("archived"):
            target = self._target_archive(result)
            context.pending_commit = {
                "task_id": result.get("task_id"),
                "archive_id": context.archive_id,
            }
            if target is not None:
                context.pending_commit["target_archive_id"] = target
        if result.get("archived") is True:
            context.completed_runs = 0
            confirmed = self._archive_sequence(self._target_archive(result))
            if confirmed:
                context.commit_count = max(context.commit_count or 0, confirmed)
            elif context.commit_count is not None:
                context.commit_count += 1
        if context.pending_commit:
            context.summary_check_countdown = 0
            progress = self._refresh_progress.get(context.session_key)
            if progress is not None:
                progress.target_archive_id = context.pending_commit.get("target_archive_id")
            self._remember_commit(context)
        return bool(context.pending_commit)

    async def _send_commit(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        pending_tokens: int | None,
        trigger: CommitTrigger,
        policy: OVCommitPolicy | None,
        keep: int,
    ) -> dict[str, Any]:
        if policy:
            assert pending_tokens is not None
            try:
                committed = await policy.commit(
                    context.session_id,
                    pending_tokens,
                    context.completed_runs,
                    identity=identity,
                    trigger=trigger,
                    keep_recent_count=keep,
                )
            except BaseException:
                self.invalidate_tokens(context)
                raise
            if committed.status != "skipped":
                self.invalidate_tokens(context)
            if committed.status != "archived":
                return {"archived": False}
            return {
                "archived": committed.archived,
                "task_id": committed.task_id,
                "archive_uri": committed.archive_uri,
            }
        self.invalidate_tokens(context)
        return await self.client.call(
            "commit_session",
            identity=identity,
            session_id=context.session_id,
            keep_recent_count=keep,
        )

    async def wait_summary(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
        model_request: ModelRequest[Any] | None = None,
    ) -> None:
        async def wait() -> None:
            failures = 0
            delay = self.config.poll_interval
            max_delay = max(delay, self.config.poll_max_interval)
            while True:
                try:
                    await self.check_summary(
                        context,
                        identity=identity,
                        token_budget=token_budget,
                        model_request=model_request,
                    )
                except Exception as exc:
                    if self.config.retry_attempts <= 0 or not retryable_read(exc):
                        raise
                    failures += 1
                    if failures > self.config.retry_attempts:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue
                failures = 0
                if context.pending_commit is None and not self._unapplied_commits(context):
                    return
                # check_summary already reads this task and handles failure and
                # cancellation. Reuse that outcome rather than polling it twice.
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

        with self.observe("context.compress_wait", context):
            await asyncio.wait_for(wait(), self.config.commit_wait_timeout)
