# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Runtime helpers for server-side idle session commits."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from openviking.pyagfs import AsyncAGFSClient
from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext, Role
from openviking.session.auto_commit_policy import resolve_session_auto_commit_policy
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

SESSION_META_SUFFIX = "/.meta.json"
AGFS_SESSION_SCAN_ROOT = "/local"


class SessionAutoCommitScheduler:
    """Periodically find idle sessions and schedule full session commits."""

    def __init__(
        self,
        session_service: Any,
        viking_fs: Any,
        config: Any,
        *,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._session_service = session_service
        self._viking_fs = viking_fs
        self._config = config
        self._check_interval = float(config.check_interval_seconds)
        self._scan_rate_limit = max(0.0, float(config.scan_rate_limit_files_per_second))
        self._min_read_interval = 1.0 / self._scan_rate_limit if self._scan_rate_limit > 0 else 0.0
        self._last_scan_read_at = 0.0
        self._sleep = sleep
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._agfs_client: Optional[AsyncAGFSClient] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info(
            "SessionAutoCommitScheduler started with check interval %.3fs",
            self._check_interval,
        )
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            if not self._config.enabled:
                await self._sleep(self._check_interval)
                continue
            round_start = time.monotonic()
            try:
                await self._scan_once()
            except Exception as exc:
                logger.error(
                    "Session auto-commit scheduler loop failed: %s",
                    exc,
                    exc_info=True,
                )

            wait = self._check_interval - (time.monotonic() - round_start)
            if wait > 0:
                try:
                    await self._sleep(wait)
                except asyncio.CancelledError:
                    break

    async def _scan_once(self) -> None:
        if not self._config.enabled:
            return
        now = datetime.now(timezone.utc)
        scanned = 0
        candidates = 0
        scheduled = 0
        agfs = self._get_agfs_client()

        async for meta_path in self._iter_session_meta_paths(agfs):
            await self._pace_scan_read()
            item = await self._read_idle_candidate(agfs, meta_path, now)
            scanned += 1
            if item is None:
                continue

            session_id, account_id, user_id = item
            candidates += 1
            ctx = RequestContext(
                user=UserIdentifier(account_id=account_id, user_id=user_id),
                role=Role.USER,
            )
            if await self._session_service.maybe_schedule_auto_commit(
                session_id,
                ctx,
                reason_hint="idle_timeout",
            ):
                scheduled += 1

        if candidates > 0:
            logger.info(
                "SessionAutoCommitScheduler scanned=%d candidates=%d scheduled=%d",
                scanned,
                candidates,
                scheduled,
            )

    async def _pace_scan_read(self) -> None:
        if self._min_read_interval <= 0:
            return
        now = time.monotonic()
        wait = self._min_read_interval - (now - self._last_scan_read_at)
        if wait > 0:
            await self._sleep(wait)
        self._last_scan_read_at = time.monotonic()

    def _get_agfs_client(self) -> AsyncAGFSClient:
        if self._agfs_client is None:
            self._agfs_client = AsyncAGFSClient(self._viking_fs.agfs)
        return self._agfs_client

    async def _read_idle_candidate(
        self,
        agfs: AsyncAGFSClient,
        meta_path: str,
        now: datetime,
    ) -> Optional[tuple[str, str, str]]:
        try:
            content = await agfs.read(meta_path)
            raw = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            meta = json.loads(raw)
            if not isinstance(meta, dict):
                logger.warning(
                    "Invalid session meta object for idle auto-commit: %s",
                    meta_path,
                )
                return None
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid session meta JSON for idle auto-commit: %s (%s)",
                meta_path,
                exc,
            )
            return None
        except Exception as exc:
            if is_not_found_error(exc):
                logger.debug(
                    "Session meta disappeared during idle auto-commit scan: %s",
                    meta_path,
                )
                return None
            logger.warning(
                "Failed to read session meta for idle auto-commit: %s",
                meta_path,
                exc_info=True,
            )
            return None

        policy = resolve_session_auto_commit_policy(meta, self._config)
        if policy is None or not is_idle_commit_due(meta, now, policy.idle_timeout_seconds):
            return None

        path_account_id = _account_id_from_meta_path(meta_path)
        path_user_id = _user_id_from_meta_path(meta_path)
        session_id = str(meta.get("session_id") or _session_id_from_meta_path(meta_path)).strip()
        account_id = str(meta.get("created_by_account_id") or path_account_id).strip()
        user_id = str(meta.get("created_by_user_id") or path_user_id).strip()
        if not session_id or not account_id or not user_id:
            return None
        return session_id, account_id, user_id

    async def _iter_session_meta_paths(
        self,
        agfs: AsyncAGFSClient,
    ) -> AsyncIterator[str]:
        try:
            account_entries = await agfs.ls(AGFS_SESSION_SCAN_ROOT)
        except Exception:
            logger.warning(
                "Failed to scan AGFS tree for idle auto-commit",
                exc_info=True,
            )
            return

        seen: set[str] = set()
        for account_entry in account_entries:
            account_id = str(account_entry.get("name") or "").strip()
            if not account_id or account_id == "_system" or not account_entry.get("isDir", False):
                continue

            users_root = f"/local/{account_id}/user"
            try:
                user_entries = await agfs.ls(users_root)
            except Exception as exc:
                if not is_not_found_error(exc):
                    logger.warning(
                        "Failed to scan users for idle auto-commit: %s",
                        users_root,
                        exc_info=True,
                    )
                continue

            for user_entry in user_entries:
                user_id = str(user_entry.get("name") or "").strip()
                if not user_id or not user_entry.get("isDir", False):
                    continue

                sessions_root = f"{users_root}/{user_id}/sessions"
                try:
                    session_entries = await agfs.ls(sessions_root)
                except Exception as exc:
                    if not is_not_found_error(exc):
                        logger.warning(
                            "Failed to scan sessions for idle auto-commit: %s",
                            sessions_root,
                            exc_info=True,
                        )
                    continue

                for session_entry in session_entries:
                    session_id = str(session_entry.get("name") or "").strip()
                    if not session_id or not session_entry.get("isDir", False):
                        continue
                    meta_path = f"{sessions_root}/{session_id}{SESSION_META_SUFFIX}"
                    if meta_path not in seen:
                        seen.add(meta_path)
                        yield meta_path


def has_idle_uncommitted_content(meta: dict[str, Any]) -> bool:
    """Return whether a session still has live messages to archive."""
    return _coerce_non_negative_int(meta.get("pending_tokens", 0)) > 0 or (
        _coerce_non_negative_int(meta.get("message_count", 0)) > 0
    )


def is_idle_commit_due(
    meta: dict[str, Any],
    now: datetime,
    idle_timeout_seconds: int,
) -> bool:
    """Check the global idle timeout against a session's last activity."""
    if idle_timeout_seconds <= 0 or not has_idle_uncommitted_content(meta):
        return False

    # Older metadata has no last_message_at. updated_at is the safest available
    # fallback and prevents existing sessions from being excluded indefinitely.
    last_message_at = meta.get("last_message_at") or meta.get("updated_at")
    if not last_message_at:
        return False
    try:
        last_activity = parse_iso_datetime(str(last_message_at))
    except (TypeError, ValueError, OverflowError):
        return False

    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=timezone.utc)
    else:
        last_activity = last_activity.astimezone(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    return last_activity + timedelta(seconds=idle_timeout_seconds) <= now


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _session_id_from_meta_path(meta_path: str) -> str:
    if not meta_path.endswith(SESSION_META_SUFFIX):
        return ""
    parts = [part for part in meta_path.split("/") if part]
    if len(parts) >= 6 and parts[0] == "local" and parts[2] == "user" and parts[4] == "sessions":
        return parts[5]
    return ""


def _account_id_from_meta_path(meta_path: str) -> str:
    parts = [part for part in meta_path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "local":
        return parts[1]
    return ""


def _user_id_from_meta_path(meta_path: str) -> str:
    parts = [part for part in meta_path.split("/") if part]
    if len(parts) >= 6 and parts[0] == "local" and parts[2] == "user" and parts[4] == "sessions":
        return parts[3]
    return ""
