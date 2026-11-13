# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory recall and native context reading contracts and implementation."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Coroutine
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from openviking.integrations._message_content import (
    comparable_content as comparable_content,
)
from openviking.integrations._message_content import (
    decode_tool_content,
)
from openviking.integrations.langchain.ov_context.recall_ranking import (
    is_experience,
    rank_candidates,
    render_experience,
    should_recall_experience,
)
from openviking_cli._exception_utils import auth_error
from openviking_cli.client.identity import _identity_arguments
from openviking_cli.retrieve.context import (
    ContextSearchOptions,
    RecallPolicy,
    parse_context_result,
)
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext

from .errors import ContextHistoryBudgetError, ContextReadError
from .identity import OVIdentity
from .recall_assembly import assemble_recall
from .recall_dedup import RecallDeduplicator
from .recall_selection import is_event_memory, query_purpose

RecallStatus = Literal["success", "empty", "timeout", "error"]


class AsyncMemoryClient(Protocol):
    """Application-injected OV async client supporting invocation-local identity."""

    async def find(
        self,
        query: str,
        *,
        target_uri: str | list[str] = "",
        limit: int = 10,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
        identity: OVIdentity | None = None,
    ) -> FindResult: ...

    async def read(self, uri: str, *, identity: OVIdentity | None = None) -> str: ...

    async def get_session_context(
        self, session_id: str, token_budget: int = 128_000, *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        """Return messages and their latest_finished_archive_id in the same response."""
        ...


@dataclass(frozen=True)
class MemoryReaderConfig:
    """Recall obeys character limits and, in context mode, recall_max_tokens.

    In find mode, candidate limits apply once across all roots in each channel.
    context_token_budget separately controls the session history view.
    """

    enabled: bool = True
    target_uris: tuple[str, ...] | None = None
    candidate_limit: int = 25
    recall_limit: int = 5
    score_threshold: float = 0.1
    prefer_abstract: bool = True
    ranking: Literal["openclaw", "score"] = "openclaw"
    prefer_leaf: bool = True
    max_injected_chars: int = 4_000
    recall_timeout: float = 5.0
    experience_enabled: bool = False
    experience_target_uris: tuple[str, ...] = ("viking://user/memories/experiences",)
    experience_candidate_limit: int = 12
    experience_limit: int = 3
    experience_score_threshold: float = 0.35
    experience_max_chars: int = 2_000
    experience_min_query_chars: int = 12
    context_timeout: float = 10.0
    context_token_budget: int = 128_000
    search_mode: Literal["find", "context"] = "context"
    recall_max_tokens: int = 1600
    recall_quotas: dict[str, int] | None = None
    recall_detail: str | dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.search_mode not in {"find", "context"}:
            raise ValueError("search_mode must be 'find' or 'context'")
        ContextSearchOptions(
            mode="context",
            max_tokens=self.recall_max_tokens,
            quotas=self.recall_quotas,
            detail=self.recall_detail,
        )
        for name in ("target_uris", "experience_target_uris"):
            uris = getattr(self, name)
            if name == "target_uris" and uris is None:
                continue
            if not isinstance(uris, tuple) or any(
                not isinstance(uri, str) or not uri.startswith("viking://") for uri in uris
            ):
                raise ValueError(f"{name} must be a tuple of viking:// URIs")
        for name in (
            "candidate_limit",
            "recall_limit",
            "max_injected_chars",
            "experience_candidate_limit",
            "experience_limit",
            "experience_max_chars",
            "experience_min_query_chars",
            "context_token_budget",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("score_threshold", "experience_score_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in ("recall_timeout", "context_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ranking not in {"openclaw", "score"}:
            raise ValueError("ranking must be 'openclaw' or 'score'")


@dataclass(frozen=True)
class RecallResult:
    """One recall attempt; the caller records every status in OVContext."""

    content: str = ""
    status: RecallStatus = "empty"
    count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSnapshot:
    """Model view and its applied terminal archive boundary, including done/failed."""

    messages: list[BaseMessage] = field(default_factory=list)
    # This boundary can differ from the archive that produced the summary text.
    archive_id: str | None = None


@dataclass(frozen=True)
class ContextReadResult:
    """SessionEngine summary-check metadata for model budgeting and observation."""

    archive_id: str | None
    pending_tokens: int | None = None
    probe_performed: bool = True


class ContextReaderBase(Protocol):
    """Return recall results and snapshots using the explicit invocation identity.

    Borrow the same DeepAgent-provided async client as Writer and Middleware.
    Never retain a mutable current caller or Session on a shared reader.
    """

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        """Return native messages and their done/failed boundary from one response.

        SessionEngine owns readiness checks, session serialization, snapshot
        validation, OVContext.replace() and pending commit bookkeeping. The
        reader never mutates the caller's cache or merges graph history.
        expected_archive_id verifies a confirmed terminal boundary, not a
        commit counter or the source archive of the summary text.
        """
        ...

    async def recall(self, query: str, *, identity: OVIdentity, session_id: str) -> RecallResult:
        """Retrieve, filter, read and render authorized memory for this query.

        Own retrieval timeouts and filters. Authentication failures propagate;
        optional failures return empty content with timeout/error status.
        Middleware owns per-Run reuse and injection into the current user copy.
        Pass session_id to SDK search, but not to SDK find.
        """
        ...


def restore_messages(records: list[dict[str, Any]]) -> list[BaseMessage]:
    """Handle both user-role tool results and legacy assistant-role tool results."""
    messages: list[BaseMessage] = []
    for record in records:
        role = record.get("role")
        if role not in {"user", "assistant"}:
            raise ContextReadError(f"Unsupported OV message role: {role!r}")
        ov_id = record.get("id")
        # The caller preserves native message IDs when writing records to OV.
        message_id = ov_id
        origin = {"openviking_message_id": ov_id}
        message_kind = record.get("message_kind")
        source_ids = record.get("source_message_ids")
        if message_kind is not None:
            origin["openviking_message_kind"] = message_kind
        if source_ids is not None:
            if not isinstance(source_ids, list) or any(
                not isinstance(key, str) or not key.strip() for key in source_ids
            ):
                raise ContextReadError("Invalid source_message_ids in OV message")
            origin["openviking_source_message_ids"] = list(source_ids)
        blocks: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        parts = record.get("parts")
        if parts is None:
            parts = [{"type": "text", "text": record.get("content", "")}]
        for part in parts:
            kind = part.get("type", "text")
            if kind == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_url":
                blocks.append({"type": "image_url", "image_url": deepcopy(part["image_url"])})
            elif kind == "context":
                blocks.append({"type": "text", "text": part.get("abstract", "")})
            elif kind == "tool":
                if not part.get("tool_id"):
                    raise ContextReadError("Tool Part is missing its tool_id")
                status = part.get("tool_status", "pending")
                if status in {"completed", "error"}:
                    results.append(part)
                elif role == "assistant" and status in {"pending", "running"}:
                    if not part.get("tool_name"):
                        raise ContextReadError("Assistant tool call is missing its tool_name")
                    calls.append(
                        {
                            "id": part["tool_id"],
                            "name": part["tool_name"],
                            "args": deepcopy(part.get("tool_input") or {}),
                        }
                    )
                else:
                    raise ContextReadError(f"Unsupported tool status {status!r} for {role}")
            else:
                raise ContextReadError(f"Unsupported OV Part type: {kind!r}")
        content: str | list[str | dict[str, Any]] = list(blocks)
        if all(block["type"] == "text" for block in blocks):
            content = "".join(block["text"] for block in blocks)
        if role == "assistant" and (blocks or calls or not results):
            messages.append(
                AIMessage(
                    id=message_id, content=content, tool_calls=calls, additional_kwargs=origin
                )
            )
        elif role == "user" and (blocks or not results):
            user_id = message_id
            if message_kind == "user_query" and source_ids and len(source_ids) == 1:
                user_id = source_ids[0]
            messages.append(HumanMessage(id=user_id, content=content, additional_kwargs=origin))
        for part in results:
            try:
                tool_content = decode_tool_content(part)
            except ValueError as exc:
                raise ContextReadError("Invalid structured tool content") from exc
            result_id = message_id
            if blocks or calls or len(results) > 1:
                result_id = f"{message_id or 'ov_tool'}:{part['tool_id']}"
            messages.append(
                ToolMessage(
                    id=result_id,
                    content=tool_content,
                    tool_call_id=part["tool_id"],
                    name=part.get("tool_name") or None,
                    status="error" if part.get("tool_status") == "error" else "success",
                    additional_kwargs=origin,
                )
            )
    return messages


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
    its own immutable identity. Structured recall requires an explicit identity
    to resolve user and current-peer scopes before candidate retrieval.
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

    def _ordinary_targets(self, identity: OVIdentity | None) -> tuple[str, ...]:
        if self.config.target_uris is not None:
            return self.config.target_uris
        roots = ("viking://user/memories",)
        if identity is not None:
            roots += (f"viking://user/peers/{identity.actor_peer_id}/memories",)
        return roots

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
        if identity is None:
            raise ValueError("Structured memory recall requires an explicit OVIdentity")
        policy_values = {name: getattr(cfg, name) for name in RecallPolicy.model_fields}
        if cfg.recall_quotas is None and cfg.target_uris is None and identity is not None:
            policy_values["target_uris"] = self._ordinary_targets(identity)
        policy = RecallPolicy(**policy_values)
        # Session IDs belong to history exchange; recall never loads a session
        # or asks a model to classify, expand or rewrite the original query.
        options = ContextSearchOptions(
            mode="context",
            recall_policy=policy,
            quotas=cfg.recall_quotas,
            detail=cfg.recall_detail,
            max_tokens=cfg.recall_max_tokens,
            query_expansion="off",
        )
        result = parse_context_result(
            await assemble_recall(self.client, query, identity, cfg, options)
        )
        entries, content = result["entries"], result["rendered"]
        if entries and content.strip():
            return RecallResult(content, "success", len(entries), result["stats"])
        errors = result["stats"].get("retrieval_errors") or result["stats"].get("read_errors")
        return RecallResult(status="error" if errors else "empty", stats=result["stats"])

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
            roots = list(dict.fromkeys(self._ordinary_targets(identity)))
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
        deduplicator = RecallDeduplicator(
            query, enabled=query_purpose(query).kind not in {"history", "mixed"}
        )
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
                dedup_scope = (
                    item.uri
                    if is_event_memory(item) and query_purpose(query).kind != "execute"
                    else ""
                )
                content = deduplicator.filter(content, scope=dedup_scope).text
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
                deduplicator.remember(content, scope=dedup_scope)
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


__all__ = [
    "AsyncMemoryClient",
    "MemoryReaderConfig",
    "RecallResult",
    "RecallStatus",
    "ContextReadResult",
    "ContextSnapshot",
    "ContextReaderBase",
    "OVContextReader",
    "ContextReadError",
    "ContextHistoryBudgetError",
    "restore_messages",
    "comparable_content",
]
