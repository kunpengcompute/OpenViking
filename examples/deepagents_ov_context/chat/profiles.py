# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session-owned execution settings shared by the Chat page and batch clients."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from examples.deepagents_ov_context.chat.config import ChatOVPolicy
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import (
    MemoryReaderConfig,
    OVContextReader,
    RecallResult,
)
from openviking_cli.client.identity import OVIdentity

MAX_OUTPUT_TOKENS = 65_536


class CommitOverrides(BaseModel):
    """Optional per-session values; omitted fields inherit the service environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    token_budget: int | None = Field(default=None, gt=0, strict=True)
    commit_token_threshold_ratio: float | None = Field(
        default=None, gt=0, le=0.7, allow_inf_nan=False, strict=True
    )
    completed_run_threshold: int | None = Field(default=None, gt=0, strict=True)
    keep_recent_count: int | None = Field(default=None, ge=0, le=10_000, strict=True)

    def resolve(self, defaults: CommitConfig) -> CommitConfig:
        return replace(defaults, **self.model_dump(exclude_none=True))


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["current", "baseline", "context_only", "ordinary", "experience"] = "current"
    enable_todos: bool | None = Field(default=None, strict=True)
    browser_only: bool = Field(default=False, strict=True)
    native_tool_compression: bool = Field(default=True, strict=True)
    target_uris: tuple[str, ...] | None = None
    experience_target_uris: tuple[str, ...] | None = None
    memory_budget_chars: int | None = Field(default=None, ge=1, le=64000)
    memory_budget_tokens: int | None = Field(default=None, ge=1, le=128000)
    memory_detail: Literal["abstract", "overview", "full"] | None = None
    recall_timeout: float | None = Field(default=None, gt=0, le=600)
    memory_recall_retry_attempts: int = Field(default=0, ge=0, le=3, strict=True)
    context_window: int | None = Field(default=None, ge=1024, le=1_000_000)
    base_context_ratio: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    capacity_ratio: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    recursion_limit: int | None = Field(default=None, ge=1, le=1000)
    thinking: bool | None = None
    memory_extraction: bool = Field(default=True, strict=True)
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)
    model_call_timeout_seconds: float | None = Field(
        default=None, gt=0, le=3600, allow_inf_nan=False
    )
    system_prompt: str | None = Field(default=None, min_length=1, max_length=32000)
    commit: CommitOverrides = Field(default_factory=CommitOverrides)

    @field_validator("target_uris", "experience_target_uris")
    @classmethod
    def valid_uris(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if values is not None and any(
            not uri.startswith("viking://") or "\n" in uri or "\r" in uri for uri in values
        ):
            raise ValueError("Memory targets must be viking:// URIs")
        return values

    @model_validator(mode="after")
    def valid_budget(self) -> ExecutionProfile:
        if self.base_context_ratio is not None and self.capacity_ratio is not None:
            if self.base_context_ratio >= self.capacity_ratio:
                raise ValueError("base_context_ratio must be less than capacity_ratio")
        if self.context_window is not None and self.max_tokens is not None:
            if self.context_window <= self.max_tokens + 512:
                raise ValueError("Context window must allow input, output and safety margin")
        return self

    @property
    def uses_ov(self) -> bool:
        return self.mode != "baseline"

    @property
    def native_context(self) -> bool:
        return self.mode in {"current", "context_only", "ordinary", "experience"}

    def memory_config(self, defaults: MemoryReaderConfig) -> MemoryReaderConfig:
        """Apply saved page overrides to the application defaults on every graph rebuild."""
        overrides: dict[str, Any] = {}
        if self.mode != "current":
            overrides.update(
                enabled=self.mode == "ordinary",
                experience_enabled=self.mode == "experience",
            )
            if defaults.search_mode == "context" and defaults.recall_quotas is not None:
                # Explicit quotas override reader switches, so isolate page modes here.
                overrides["recall_quotas"] = {
                    category: count
                    for category, count in defaults.recall_quotas.items()
                    if (self.mode == "ordinary" and category != "experiences")
                    or (self.mode == "experience" and category == "experiences")
                }
        for name in ("target_uris", "experience_target_uris", "recall_timeout"):
            value = getattr(self, name)
            if value is not None:
                overrides[name] = value
        if self.memory_budget_tokens is not None:
            overrides["recall_max_tokens"] = self.memory_budget_tokens
        if self.memory_detail is not None:
            overrides["recall_detail"] = self.memory_detail
        if self.memory_budget_chars is not None:
            overrides.update(
                max_injected_chars=self.memory_budget_chars,
                experience_max_chars=self.memory_budget_chars,
            )
        return replace(defaults, **overrides)

    def commit_settings(self, defaults: CommitConfig) -> CommitConfig:
        return self.commit.resolve(defaults)


class ObservedContextReader(OVContextReader):
    """Persist recall observations through the same Chat Run event stream."""

    observer: Callable[..., None]

    def __init__(
        self,
        client: Any,
        config: MemoryReaderConfig | None = None,
        *,
        scope_policy: ChatOVPolicy | None = None,
    ) -> None:
        super().__init__(client, config)
        self.scope_policy = scope_policy

    async def recall(
        self, query: str, *, identity: OVIdentity | None = None, session_id: str | None = None
    ) -> RecallResult:
        config = self.config
        if self.scope_policy is not None:
            if identity is None:
                raise ValueError("Identity-scoped recall requires an explicit identity")
            config = replace(
                config,
                target_uris=self.scope_policy.recall_targets(
                    config.target_uris,
                    identity.actor_peer_id,
                    context_quotas=config.search_mode == "context"
                    and config.recall_quotas is not None,
                ),
            )
        # A per-call Reader avoids retaining a mutable current-user/Agent scope.
        result = await OVContextReader(self.client, config).recall(
            query, identity=identity, session_id=session_id
        )
        self.observer(
            "memory_recall_result",
            query=query,
            status=result.status,
            count=result.count,
            content=result.content,
            stats=result.stats,
            target_uris=list(config.target_uris) if config.target_uris is not None else None,
        )
        return result
