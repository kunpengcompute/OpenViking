# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Async memory recall and context exchange from one terminal-version response."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage

from openviking_cli._exception_utils import auth_error
from openviking_cli.client.identity import OVIdentity, _identity_arguments
from openviking_cli.retrieve.context import RecallPolicy, parse_context_result
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext

from .context_messages import restore_messages
from .ranking import is_experience, rank_candidates, render_experience, should_recall_experience
from .types import (
    AsyncMemoryClient,
    ContextReadError,
    ContextSnapshot,
    MemoryReaderConfig,
    RecallResult,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _discard_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def _within_deadline(operation: Coroutine[Any, Any, _T], timeout: float) -> _T:
    """Discard late results even if a client suppresses cancellation."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    task = asyncio.create_task(operation)
    try:
        completed, _ = await asyncio.wait({task}, timeout=timeout)
        if completed and loop.time() <= deadline:
            return task.result()
        raise asyncio.TimeoutError
    finally:
        if not task.done():
            task.cancel()
        task.add_done_callback(_discard_result)


class OVContextReader:
    """Session-stateless execution module shared by one middleware across sessions.

    Reuse the application-injected OV async client. Each invocation can supply
    its own immutable identity; omitted identity preserves the SDK defaults.
    Session IDs arrive with each context exchange; progress, recall results and
    Run flags belong to state["ov_context"], never to the reader. The caller
    coordinates session operations; OV records preserve native message IDs.
    """

    def __init__(
        self,
        client: AsyncMemoryClient,
        config: MemoryReaderConfig | None = None,
    ):
        # Connection resources are shared; request identity is never stored here.
        self.client = client
        self.config = config or MemoryReaderConfig()

    async def recall(
        self, query: str, *, identity: OVIdentity | None = None, session_id: str | None = None
    ) -> RecallResult:
        """Recall from original user text under one deadline, including all body reads."""
        if not isinstance(query, str):
            raise TypeError("query must be the original user's text")
        if not query.strip() or self.config.max_injected_chars == 0:
            return RecallResult()
        try:
            operation = (
                self._recall_context(query, **_identity_arguments(identity))
                if self.config.search_mode == "context"
                else self._recall(query, **_identity_arguments(identity))
            )
            return await _within_deadline(operation, self.config.recall_timeout)
        except asyncio.TimeoutError:
            return RecallResult(status="timeout")
        except Exception as exc:
            if auth_error(exc):
                raise
            logger.debug("OV recall failed", exc_info=True)
            return RecallResult(status="error")

    async def _recall_context(
        self, query: str, *, identity: OVIdentity | None = None
    ) -> RecallResult:
        cfg = self.config
        ordinary = cfg.enabled and cfg.recall_limit > 0 and cfg.candidate_limit > 0
        experience = (
            cfg.experience_enabled
            and cfg.experience_limit > 0
            and cfg.experience_candidate_limit > 0
            and cfg.experience_max_chars > 0
        )
        if cfg.recall_quotas is not None:
            if not any(cfg.recall_quotas.values()):
                return RecallResult()
        elif not ordinary and not experience:
            return RecallResult()
        policy = RecallPolicy(**{name: getattr(cfg, name) for name in RecallPolicy.model_fields})
        # Session IDs belong to history exchange; recall never loads a session
        # or asks a model to classify, expand or rewrite the original query.
        result = parse_context_result(
            await self.client.search(
                query,
                mode="context",
                recall_policy=policy.model_dump(mode="json"),
                quotas=cfg.recall_quotas,
                detail=cfg.recall_detail,
                max_tokens=cfg.recall_max_tokens,
                query_expansion="off",
                **_identity_arguments(identity),
            )
        )
        entries, content = result["entries"], result["rendered"]
        if entries and content.strip():
            return RecallResult(content, "success", len(entries))
        errors = result["stats"].get("retrieval_errors") or result["stats"].get("read_errors")
        return RecallResult(status="error" if errors else "empty")

    async def _recall(self, query: str, *, identity: OVIdentity | None = None) -> RecallResult:
        cfg = self.config
        ordinary = cfg.enabled and cfg.recall_limit > 0 and cfg.candidate_limit > 0
        experience = (
            cfg.experience_enabled
            and cfg.experience_limit > 0
            and cfg.experience_candidate_limit > 0
            and cfg.experience_max_chars > 0
            and should_recall_experience(query, cfg.experience_min_query_chars)
        )
        searches: list[tuple[bool, str | list[str]]] = []
        if ordinary:
            roots = list(
                dict.fromkeys(
                    cfg.target_uris if cfg.target_uris is not None else ("viking://user/memories",)
                )
            )
            if roots:
                searches.append((False, roots[0] if len(roots) == 1 else roots))
        if experience:
            roots = list(dict.fromkeys(cfg.experience_target_uris))
            if roots:
                searches.append((True, roots[0] if len(roots) == 1 else roots))
        if not searches:
            return RecallResult()
        tasks = [
            asyncio.create_task(
                self._search(query, uri, experience=exp, **_identity_arguments(identity))
            )
            for exp, uri in searches
        ]
        try:
            # Auth errors fail immediately, even if another search is still waiting.
            # Optional errors are returned as outcomes to preserve partial success.
            outcomes = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                task.add_done_callback(_discard_result)
        ordinary_items: list[MatchedContext] = []
        experience_items: list[MatchedContext] = []
        had_error = False
        for (exp, _), outcome in zip(searches, outcomes, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, Exception):
                if auth_error(outcome):
                    raise outcome
                had_error = True
                continue
            for item in outcome.memories:
                if item.context_type != ContextType.MEMORY:
                    continue
                if exp:
                    if is_experience(item):
                        experience_items.append(item)
                elif not is_experience(item):
                    ordinary_items.append(item)
        sections: list[str] = []
        count = 0
        # Task guidance precedes user facts. Both sections share the final character limit.
        for exp, items, header, limit in (
            (True, experience_items, "[任务经验]\n以下内容作为执行参考。", cfg.experience_limit),
            (False, ordinary_items, "[长期记忆]", cfg.recall_limit),
        ):
            candidates = rank_candidates(items, query, cfg, experience=exp)
            entries: list[str] = []
            for item in candidates:
                if len(entries) >= limit:
                    break
                try:
                    content = await self._content(
                        item, experience=exp, **_identity_arguments(identity)
                    )
                except Exception as exc:
                    if auth_error(exc):
                        raise
                    had_error = True
                    continue
                if not content:
                    continue
                entry = (
                    render_experience(item, content)
                    if exp
                    else (f"- [{item.category or 'memory'}] {content}\n  来源：{item.uri}")
                )
                section = header + "\n" + "\n\n".join([*entries, entry])
                full_content = "\n\n".join([*sections, section])
                if len(full_content) > cfg.max_injected_chars:
                    continue
                if exp and len(section) > cfg.experience_max_chars:
                    continue
                entries.append(entry)
            if entries:
                sections.append(header + "\n" + "\n\n".join(entries))
                count += len(entries)
        if count:
            return RecallResult("\n\n".join(sections), "success", count)
        return RecallResult(status="error" if had_error else "empty")

    async def _search(
        self,
        query: str,
        uri: str | list[str],
        *,
        experience: bool,
        identity: OVIdentity | None = None,
    ) -> FindResult | Exception:
        cfg = self.config
        try:
            return await self.client.find(
                query,
                target_uri=uri,
                limit=cfg.experience_candidate_limit if experience else cfg.candidate_limit,
                score_threshold=cfg.experience_score_threshold
                if experience
                else cfg.score_threshold,
                **_identity_arguments(identity),
            )
        except Exception as exc:
            if auth_error(exc):
                raise
            return exc

    async def _content(
        self, item: MatchedContext, *, experience: bool, identity: OVIdentity | None = None
    ) -> str:
        fallback = (item.abstract or item.overview or "").strip()
        if self.config.prefer_abstract and item.abstract.strip() and not experience:
            return item.abstract.strip()
        if item.level == 2:
            try:
                content = await self.client.read(item.uri, **_identity_arguments(identity))
                return content.strip() or fallback
            except Exception as exc:
                if auth_error(exc):
                    raise
                logger.debug("OV memory body read failed; using available summary", exc_info=True)
                if not fallback:
                    raise
        return fallback

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity | None = None,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        """Read this session's model view and verify an optional terminal boundary.

        The caller passes OVContext.session_id. Messages and the done/failed
        boundary come from one response, without separate archive lookups.
        Session locking and current Run user validation belong to the caller.
        """
        try:
            return await _within_deadline(
                self._exchange_context(
                    session_id,
                    expected_archive_id,
                    token_budget=token_budget,
                    **_identity_arguments(identity),
                ),
                self.config.context_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ContextReadError("Working context read timed out") from exc
        except ContextReadError:
            raise
        except Exception as exc:
            if auth_error(exc):
                raise
            if getattr(exc, "code", None) == "RESOURCE_EXHAUSTED":
                from .types import ContextHistoryBudgetError

                raise ContextHistoryBudgetError(
                    "Complete session summary exceeds the history budget"
                ) from exc
            raise ContextReadError("Working context could not be verified") from exc

    async def _exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None,
        *,
        identity: OVIdentity | None = None,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        if token_budget is not None and (type(token_budget) is not int or token_budget < 0):
            raise ContextReadError("token_budget must be a non-negative integer")
        context = await self.client.get_session_context(
            session_id,
            token_budget=self.config.context_token_budget if token_budget is None else token_budget,
            **_identity_arguments(identity),
        )
        if not isinstance(context, dict) or "latest_finished_archive_id" not in context:
            raise ContextReadError("Context response is missing latest_finished_archive_id")
        archive_id = context["latest_finished_archive_id"]
        if archive_id is not None and (not isinstance(archive_id, str) or not archive_id.strip()):
            raise ContextReadError("latest_finished_archive_id must be a non-empty string or None")
        if expected_archive_id is not None and archive_id != expected_archive_id:
            raise ContextReadError("Context terminal boundary does not match expected_archive_id")
        records = context.get("messages")
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ContextReadError("Context response messages must be a list of OV message records")
        summary = context.get("latest_archive_overview", "")
        if not isinstance(summary, str):
            raise ContextReadError("Context overview must be a string")
        summary = summary.strip()
        # The server assembles this view at the returned terminal boundary. A failed
        # boundary can legitimately retain an older overview, or have no overview.
        messages = restore_messages(records)
        if summary:
            messages.insert(
                0,
                HumanMessage(
                    # Synthetic view identity, not the source archive of the overview.
                    id=f"ov_summary:{archive_id}",
                    content=f"[Session History Summary]\n{summary}",
                ),
            )
        return ContextSnapshot(messages, archive_id)
