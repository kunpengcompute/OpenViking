# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Session Service for OpenViking.

Provides session management operations: session, sessions, add_message, commit, delete.
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from openviking.core.namespace import canonical_session_uri
from openviking.server.config import ToolOutputExternalizationConfig
from openviking.server.identity import RequestContext
from openviking.service.session_auto_commit import has_idle_uncommitted_content, is_idle_commit_due
from openviking.service.task_tracker import get_task_tracker
from openviking.session import Session
from openviking.session.auto_commit_policy import (
    AutoCommitPolicy,
    resolve_session_auto_commit_policy,
    validate_idle_timeout_input,
)
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory_policy import MemoryPolicy
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import VikingFS
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.exceptions import (
    AlreadyExistsError,
    NotFoundError,
    NotInitializedError,
)
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.memory_config import SessionAutoCommitConfig

logger = get_logger(__name__)
COMMIT_RECOVERY_RETRY_DELAY = 1.0
COMMIT_RECOVERY_CACHE_LIMIT = 1024

if TYPE_CHECKING:
    from openviking.session.compressor_v2 import SessionCompressorV2


class SessionService:
    """Session management service."""

    def __init__(
        self,
        vikingdb: Optional[VikingDBManager] = None,
        viking_fs: Optional[VikingFS] = None,
        session_compressor: Optional["SessionCompressorV2"] = None,
    ):
        self._vikingdb = vikingdb
        self._viking_fs = viking_fs
        self._session_compressor = session_compressor
        self._tool_output_externalization_config = ToolOutputExternalizationConfig()
        self._auto_commit_inflight: set[tuple[str, str, str]] = set()
        self._auto_commit_inflight_lock = asyncio.Lock()
        self._auto_commit_tasks: set[asyncio.Task] = set()
        self._commit_recovery_inflight: set[tuple[str, str, str]] = set()
        self._commit_recovery_invalidated: set[tuple[str, str, str]] = set()
        self._commit_recovery_checks: OrderedDict[tuple[str, str, str], tuple[int, bool, float]] = (
            OrderedDict()
        )
        self._session_auto_commit_config = SessionAutoCommitConfig()

    def _schedule_commit_recovery(self, session: Session) -> None:
        """Resume durable pending work without blocking session reads.

        One recovery runs per account/user/session. Completed archive versions
        are cached; unfinished work can be checked again after a short delay.
        The archive lease still coordinates workers across service processes.
        """
        count = session.meta.commit_count
        # An interrupted Phase 1 can leave an intent before commit_count is saved.
        key = (session.ctx.account_id, session.ctx.user.user_id, session.session_id)
        if key in self._commit_recovery_inflight:
            return
        checked = self._commit_recovery_checks.get(key)
        if checked is not None:
            version, complete, retry_after = checked
            if version == count and (complete or monotonic() < retry_after):
                self._commit_recovery_checks.move_to_end(key)
                return
        self._commit_recovery_inflight.add(key)

        async def recover() -> None:
            from openviking.session.commit_recovery import recover_pending

            complete = False
            try:
                complete = await recover_pending(session)
            except Exception:
                logger.warning("Session commit recovery awaits retry: %s", key, exc_info=True)
            finally:
                self._commit_recovery_inflight.discard(key)
                if key in self._commit_recovery_invalidated:
                    # A commit changed durable intents during this scan; the
                    # next access must check again, even if its count is equal.
                    self._commit_recovery_invalidated.discard(key)
                    self._commit_recovery_checks.pop(key, None)
                else:
                    self._commit_recovery_checks[key] = (
                        count,
                        complete,
                        monotonic() + COMMIT_RECOVERY_RETRY_DELAY,
                    )
                    self._commit_recovery_checks.move_to_end(key)
                while len(self._commit_recovery_checks) > COMMIT_RECOVERY_CACHE_LIMIT:
                    self._commit_recovery_checks.popitem(last=False)

        task = asyncio.create_task(recover())
        self._auto_commit_tasks.add(task)
        task.add_done_callback(self._auto_commit_tasks.discard)

    def set_dependencies(
        self,
        vikingdb: VikingDBManager,
        viking_fs: VikingFS,
        session_compressor: "SessionCompressorV2",
    ) -> None:
        """Set dependencies (for deferred initialization)."""
        self._vikingdb = vikingdb
        self._viking_fs = viking_fs
        self._session_compressor = session_compressor

    def set_tool_output_externalization_config(
        self, config: ToolOutputExternalizationConfig
    ) -> None:
        """Set tool output externalization controls for newly created sessions."""
        self._tool_output_externalization_config = config.model_copy(deep=True)

    def set_session_auto_commit_config(self, config: SessionAutoCommitConfig) -> None:
        """Set server-wide defaults used by sessions without an explicit policy."""
        self._session_auto_commit_config = config.model_copy(deep=True)

    def _ensure_initialized(self) -> None:
        """Ensure all dependencies are initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")

    @staticmethod
    def _record_lifecycle_metric(action: str, status: str) -> None:
        """Best-effort session lifecycle metrics should never break the main flow."""
        try:
            from openviking.metrics.datasources.session import SessionLifecycleDataSource

            SessionLifecycleDataSource.record_lifecycle(action=action, status=status)
        except Exception:
            logger.debug(
                "Failed to record session lifecycle metric action=%s status=%s",
                action,
                status,
                exc_info=True,
            )

    @staticmethod
    def _record_archive_metric(status: str) -> None:
        """Best-effort archive metrics should never break the main flow."""
        try:
            from openviking.metrics.datasources.session import SessionLifecycleDataSource

            SessionLifecycleDataSource.record_archive(status=status)
        except Exception:
            logger.debug(
                "Failed to record session archive metric status=%s",
                status,
                exc_info=True,
            )

    def session(
        self,
        ctx: RequestContext,
        session_id: Optional[str] = None,
        *,
        session_uri: Optional[str] = None,
    ) -> Session:
        """Construct a Session handle without reading persisted state.

        Args:
            session_id: Session ID, creates a new session (auto-generated ID) if None

        Returns:
            Session instance
        """
        self._ensure_initialized()
        session = Session(
            viking_fs=self._viking_fs,
            vikingdb_manager=self._vikingdb,
            session_compressor=self._session_compressor,
            user=ctx.user,
            ctx=ctx,
            session_id=session_id,
            session_uri=session_uri,
            tool_output_externalization_config=self._tool_output_externalization_config,
        )
        session._background_tasks = self._auto_commit_tasks
        return session

    async def create(
        self,
        ctx: RequestContext,
        session_id: Optional[str] = None,
        memory_policy: Optional[Dict[str, Any]] = None,
        auto_commit_policy: Optional[Dict[str, Any]] = None,
        update_auto_commit_policy: bool = False,
    ) -> Session:
        """Create a session and persist its root path.

        Args:
            ctx: Request context
            session_id: Optional session ID. If provided, creates a session with the given ID.
                       If None, creates a new session with auto-generated ID.
            memory_policy: Optional default extraction policy for future commits.

        Raises:
            AlreadyExistsError: If a session with the given ID already exists
        """
        self._record_lifecycle_metric("create", "attempt")
        try:
            if session_id:
                existing = self.session(ctx, session_id)
                if await existing.exists():
                    raise AlreadyExistsError(f"Session '{session_id}' already exists")
            session = self.session(ctx, session_id)
            if memory_policy is not None:
                policy = MemoryPolicy.from_dict(memory_policy)
                policy.validate_memory_types(
                    set(MemoryTypeRegistry().list_names(include_disabled=False))
                )
                session.meta.memory_policy = policy.to_dict()
            if update_auto_commit_policy:
                session.meta.auto_commit_policy_configured = True
                session.meta.auto_commit_policy = (
                    None
                    if auto_commit_policy is None
                    else self._validate_auto_commit_policy(auto_commit_policy)
                )
            await session.ensure_exists()
            self._record_lifecycle_metric("create", "ok")
            return session
        except Exception:
            self._record_lifecycle_metric("create", "error")
            raise

    async def open(
        self, session_id: str, ctx: RequestContext, *, auto_create: bool = False
    ) -> Session:
        """Resolve a Session URI; mutations load state under their own lock."""
        session = self.session(ctx, session_id)
        if not await session.exists():
            if not auto_create:
                raise NotFoundError(session_id, "session")
            await session.ensure_exists()
        return session

    async def get(
        self, session_id: str, ctx: RequestContext, *, auto_create: bool = False
    ) -> Session:
        """Get an existing session.

        Args:
            session_id: Session ID
            ctx: Request context
            auto_create: If True, create the session when it does not exist.
                         Default is False (raise NotFoundError).
        """
        try:
            session = await self.open(session_id, ctx, auto_create=auto_create)
            await session.load()
            self._schedule_commit_recovery(session)
            self._record_lifecycle_metric("get", "ok")
            return session
        except Exception:
            self._record_lifecycle_metric("get", "error")
            raise

    async def get_context(
        self, session_id: str, ctx: RequestContext, token_budget: int = 128_000
    ) -> Dict[str, Any]:
        """Load the live tail and archive view under the Phase 1 session lock."""
        from openviking.session.commit_recovery import SESSION_LOCK_TIMEOUT
        from openviking.storage.transaction import LockContext, get_lock_manager

        session = await self.open(session_id, ctx)
        path = self._viking_fs._uri_to_path(session.uri, ctx=ctx)
        async with LockContext(
            get_lock_manager(), [path], lock_mode="exact", timeout=SESSION_LOCK_TIMEOUT
        ):
            await session.load()
            result = await session.get_session_context(token_budget=token_budget)
        self._schedule_commit_recovery(session)
        return result

    async def sessions(self, ctx: RequestContext) -> List[Dict[str, Any]]:
        """Get all sessions for the current user.

        Returns:
            List of session info dicts
        """
        self._ensure_initialized()
        session_base_uri = canonical_session_uri(ctx)
        sessions_by_id: Dict[str, Dict[str, Any]] = {}

        try:
            entries = await self._viking_fs.ls(session_base_uri, ctx=ctx)
            for entry in entries:
                name = entry.get("name", "")
                if name in [".", ".."]:
                    continue
                sessions_by_id[name] = {
                    "session_id": name,
                    "uri": f"{session_base_uri}/{name}",
                    "is_dir": entry.get("isDir", False),
                }
        except Exception:
            logger.debug("Failed to list sessions", exc_info=True)

        try:
            entries = await self._viking_fs.ls("viking://session", ctx=ctx)
            for entry in entries:
                name = entry.get("name", "")
                if name in [".", ".."] or name in sessions_by_id:
                    continue
                sessions_by_id[name] = {
                    "session_id": name,
                    "uri": entry.get("uri", f"viking://session/{name}"),
                    "is_dir": entry.get("isDir", False),
                }
        except Exception:
            logger.debug("Failed to list legacy sessions", exc_info=True)
        return list(sessions_by_id.values())

    async def delete(self, session_id: str, ctx: RequestContext) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()

        session_uri = canonical_session_uri(ctx, session_id)
        session = await self.open(session_id, ctx)
        if not await session.exists():
            self._record_lifecycle_metric("delete", "error")
            raise NotFoundError(session_id, "session")

        await self._viking_fs.rm(session_uri, recursive=True, ctx=ctx)
        logger.info(f"Deleted session: {session_id}")
        self._record_lifecycle_metric("delete", "ok")
        return True

    async def commit(
        self,
        session_id: str,
        ctx: RequestContext,
        keep_recent_count: int = 0,
    ) -> Dict[str, Any]:
        """Commit a session (archive messages and extract memories).

        Delegates to commit_async() for true non-blocking behavior.

        Args:
            session_id: Session ID to commit
            keep_recent_count: See :meth:`commit_async`.

        Returns:
            Commit result
        """
        return await self.commit_async(
            session_id,
            ctx,
            keep_recent_count=keep_recent_count,
        )

    async def commit_async(
        self,
        session_id: str,
        ctx: RequestContext,
        keep_recent_count: int = 0,
        *,
        persist_keep_recent_count: bool = True,
        record_auto_commit_success: bool = False,
        pre_commit_check: Optional[Callable[[Session], bool]] = None,
    ) -> Dict[str, Any]:
        """Async commit a session.

        Phase 1 (archive) always runs inline.  Phase 2 (memory extraction)
        runs in a background task, returning a task_id for polling.

        Args:
            session_id: Session ID to commit
            keep_recent_count: Number of most-recent messages to keep in the
                live session after commit. ``0`` archives everything.

        Returns:
            Commit result with keys: session_id, status, task_id,
            archive_uri, archived
        """
        self._ensure_initialized()
        session = await self.open(session_id, ctx)
        try:
            result = await session.commit_async(
                keep_recent_count=keep_recent_count,
                persist_keep_recent_count=persist_keep_recent_count,
                record_auto_commit_success=record_auto_commit_success,
                pre_commit_check=pre_commit_check,
            )
        finally:
            key = (session.ctx.account_id, session.ctx.user.user_id, session.session_id)
            # Even a failed commit can have persisted an intent while leaving
            # the metadata version unchanged. Invalidate its terminal scan.
            self._commit_recovery_checks.pop(key, None)
            if key in self._commit_recovery_inflight:
                self._commit_recovery_invalidated.add(key)
            self._schedule_commit_recovery(session)
        self._record_lifecycle_metric("commit", "ok" if result.get("status") else "error")
        self._record_archive_metric("ok" if result.get("archived") else "skip")
        return result

    def effective_auto_commit_policy(self, session: Session) -> Optional[Dict[str, Any]]:
        """Return the effective policy, including global defaults, if enabled."""
        policy = resolve_session_auto_commit_policy(
            session.meta.to_dict(), self._session_auto_commit_config
        )
        return policy.to_dict() if policy is not None else None

    async def update_auto_commit_policy(
        self,
        session_id: str,
        ctx: RequestContext,
        *,
        policy: Optional[Dict[str, Any]],
        update: bool = True,
    ) -> Session:
        """Set, partially update, or disable one session's auto-commit policy."""
        session = await self.get(session_id, ctx, auto_create=False)
        if not update:
            return session
        if policy is None:
            await session.update_auto_commit_policy(None)
            return await self.get(session_id, ctx, auto_create=False)

        # Merge the patch with the latest persisted policy while holding the
        # session metadata lock, so concurrent partial updates cannot overwrite
        # one another.
        await session.update_auto_commit_policy(policy)
        return await self.get(session_id, ctx, auto_create=False)

    @staticmethod
    def _validate_auto_commit_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
        if "idle_timeout_seconds" in policy:
            validate_idle_timeout_input(policy["idle_timeout_seconds"])
        return AutoCommitPolicy.from_dict(policy).to_dict()

    async def maybe_schedule_idle_auto_commit(
        self,
        session_id: str,
        ctx: RequestContext,
        *,
        idle_timeout_seconds: Optional[int] = None,
    ) -> bool:
        """Backward-compatible entrypoint for the periodic idle scheduler."""
        return await self.maybe_schedule_auto_commit(session_id, ctx, reason_hint="idle_timeout")

    async def maybe_schedule_auto_commit(
        self,
        session_id: str,
        ctx: RequestContext,
        *,
        reason_hint: str,
        session: Optional[Session] = None,
    ) -> bool:
        """Schedule a policy-triggered commit without blocking message writes."""
        claim = (ctx.account_id, ctx.user.user_id, session_id)
        try:
            if session is None:
                session = await self.get(session_id, ctx, auto_create=False)
            if not self._should_run_auto_commit(session, reason_hint):
                return False

            async with self._auto_commit_inflight_lock:
                if claim in self._auto_commit_inflight:
                    return False
                if await get_task_tracker().has_running(
                    "session_commit",
                    session_id,
                    account_id=ctx.account_id,
                    user_id=ctx.user.user_id,
                ):
                    return False
                self._auto_commit_inflight.add(claim)
        except Exception:
            logger.debug(
                "Skipped idle auto-commit scheduling for %s",
                session_id,
                exc_info=True,
            )
            return False

        task = asyncio.create_task(self._run_auto_commit(session_id, ctx, reason=reason_hint))
        self._auto_commit_tasks.add(task)
        task.add_done_callback(self._auto_commit_tasks.discard)
        return True

    async def _run_auto_commit(
        self,
        session_id: str,
        ctx: RequestContext,
        *,
        reason: str,
    ) -> None:
        """Run one automatic commit and release its per-process scheduling claim."""
        claim = (ctx.account_id, ctx.user.user_id, session_id)
        try:
            tracker = get_task_tracker()
            if await tracker.has_running(
                "session_commit",
                session_id,
                account_id=ctx.account_id,
                user_id=ctx.user.user_id,
            ):
                return

            session = await self.get(session_id, ctx, auto_create=False)
            if not self._should_run_auto_commit(session, reason):
                return

            policy = resolve_session_auto_commit_policy(
                session.meta.to_dict(), self._session_auto_commit_config
            )
            if policy is None:
                return
            idle_commit = reason == "idle_timeout"
            await self.commit_async(
                session_id,
                ctx,
                keep_recent_count=0 if idle_commit else policy.keep_recent_count,
                persist_keep_recent_count=not idle_commit,
                record_auto_commit_success=True,
                pre_commit_check=lambda current: self._should_run_auto_commit(current, reason),
            )
        except Exception as exc:
            logger.warning("Automatic session commit failed for %s: %s", session_id, exc)
        finally:
            async with self._auto_commit_inflight_lock:
                self._auto_commit_inflight.discard(claim)

    async def close_background_tasks(self) -> None:
        """Wait for scheduled commits and recovery before dependencies close."""
        while self._auto_commit_tasks:
            await asyncio.gather(*list(self._auto_commit_tasks), return_exceptions=True)

    def _should_run_auto_commit(self, session: Session, reason: str) -> bool:
        policy = resolve_session_auto_commit_policy(
            session.meta.to_dict(), self._session_auto_commit_config
        )
        if policy is None:
            return False

        meta = session.meta.to_dict()
        if reason == "message_write":
            if not (
                int(meta.get("pending_tokens", 0) or 0) > 0
                or int(meta.get("message_count", 0) or 0) > 0
            ):
                return False
            if self._within_min_commit_interval(session, policy):
                return False
            return bool(
                (
                    policy.pending_token_threshold > 0
                    and int(meta.get("pending_tokens", 0) or 0) > policy.pending_token_threshold
                )
                or (
                    policy.message_count_threshold > 0
                    and int(meta.get("message_count", 0) or 0) > policy.message_count_threshold
                )
            )

        if reason == "idle_timeout":
            if not self._session_auto_commit_config.enabled:
                return False
            if not has_idle_uncommitted_content(meta) or policy.idle_timeout_seconds <= 0:
                return False
            if self._within_min_commit_interval(session, policy):
                return False
            return is_idle_commit_due(meta, datetime.now(timezone.utc), policy.idle_timeout_seconds)
        return False

    @staticmethod
    def _within_min_commit_interval(session: Session, policy: AutoCommitPolicy) -> bool:
        interval = policy.min_commit_interval_seconds
        last_commit_at = session.meta.last_auto_commit_at
        if interval <= 0 or not last_commit_at:
            return False
        try:
            last = parse_iso_datetime(last_commit_at)
        except (TypeError, ValueError, OverflowError):
            return False
        now = datetime.now(last.tzinfo) if last.tzinfo else datetime.now()
        return (now - last).total_seconds() < interval

    async def get_commit_task(self, task_id: str, ctx: RequestContext) -> Optional[Dict[str, Any]]:
        """Query background commit task status by task_id for the calling owner."""
        task = await get_task_tracker().get(
            task_id,
            account_id=ctx.account_id,
            user_id=ctx.user.user_id,
        )
        return task.to_dict() if task else None

    async def extract(self, session_id: str, ctx: RequestContext) -> List[Any]:
        """Extract memories from a session.

        Args:
            session_id: Session ID to extract from

        Returns:
            List of extracted memories
        """
        self._ensure_initialized()
        if not self._session_compressor:
            raise NotInitializedError("SessionCompressorV2")

        session = await self.get(session_id, ctx)
        archive_uri = f"{session.uri}/manual_extract"

        memories = await self._session_compressor.extract_long_term_memories(
            messages=session.messages,
            user=ctx.user,
            session_id=session_id,
            ctx=ctx,
            archive_uri=archive_uri,
        )
        self._record_lifecycle_metric("extract", "ok")
        return memories
