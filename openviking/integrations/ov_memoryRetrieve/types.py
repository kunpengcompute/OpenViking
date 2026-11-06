# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Contracts for the OV memory retrieval module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from openviking.integrations._context_contracts import (
    ContextHistoryBudgetError as ContextHistoryBudgetError,
)
from openviking.integrations._context_contracts import ContextReadError as ContextReadError
from openviking.integrations._context_contracts import ContextSnapshot as ContextSnapshot
from openviking_cli.client.identity import OVIdentity
from openviking_cli.retrieve.context import ContextSearchOptions, ContextSearchResult
from openviking_cli.retrieve.types import FindResult

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
        identity: OVIdentity | None = None,
    ) -> FindResult: ...

    async def read(self, uri: str, *, identity: OVIdentity | None = None) -> str: ...

    async def search(
        self,
        query: str,
        *,
        mode: str = "list",
        target_uri: str = "",
        limit: int = 10,
        score_threshold: float | None = None,
        identity: OVIdentity | None = None,
        **context_options: Any,
    ) -> FindResult | ContextSearchResult: ...

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
    candidate_limit: int = 20
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
