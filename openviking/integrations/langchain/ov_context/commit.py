# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session commit configuration, policy contract and implementation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

from .identity import OVIdentity

CommitTrigger = Literal["policy", "capacity"]


class OVCommitClient(Protocol):
    async def commit_session(
        self,
        session_id: str,
        *,
        keep_recent_count: int,
        identity: OVIdentity | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CommitConfig:
    """Thresholds for capacity and completed-Run commits."""

    token_budget: int
    commit_token_threshold_ratio: Decimal | float | int
    completed_run_threshold: int
    keep_recent_count: int  # In-Run compression only; after_agent commits retain zero.

    def __post_init__(self) -> None:
        for name in ("token_budget", "completed_run_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.keep_recent_count, bool)
            or not isinstance(self.keep_recent_count, int)
            or self.keep_recent_count < 0
            or self.keep_recent_count > 10_000
        ):
            raise ValueError("keep_recent_count must be an integer from 0 to 10000")
        ratio_value = self.commit_token_threshold_ratio
        if isinstance(ratio_value, bool) or not isinstance(ratio_value, (Decimal, float, int)):
            raise ValueError("commit_token_threshold_ratio must be in (0, 0.7]")
        ratio = Decimal(str(ratio_value))
        if not ratio.is_finite() or not (Decimal(0) < ratio <= Decimal("0.7")):
            raise ValueError("commit_token_threshold_ratio must be in (0, 0.7]")
        object.__setattr__(self, "commit_token_threshold_ratio", ratio)


def load_commit_config(path: str | Path) -> CommitConfig:
    """Load the supported [commit] settings from a TOML file."""

    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib
        except ImportError as exc:  # pragma: no cover - depends on Python 3.10 extras
            raise RuntimeError("Loading TOML on Python 3.10 requires tomli") from exc

    with Path(path).open("rb") as config_file:
        document = tomllib.load(config_file)
    raw = document.get("commit")
    if not isinstance(raw, dict):
        raise ValueError("config must contain a [commit] table")
    common = {"completed_run_threshold", "keep_recent_count"}
    snake_keys = {"token_budget", "commit_token_threshold_ratio"}
    legacy_keys = {"tokenBudget", "commitTokenThresholdRatio"}
    if set(raw) == common | snake_keys:
        budget_key, ratio_key = "token_budget", "commit_token_threshold_ratio"
    elif set(raw) == common | legacy_keys:
        budget_key, ratio_key = "tokenBudget", "commitTokenThresholdRatio"
    else:
        raise ValueError(
            "invalid [commit] keys: use either token_budget and "
            "commit_token_threshold_ratio, or the legacy camelCase pair, "
            "along with completed_run_threshold and keep_recent_count"
        )
    ratio = raw[ratio_key]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError(f"{ratio_key} must be numeric")
    return CommitConfig(
        token_budget=raw[budget_key],
        commit_token_threshold_ratio=Decimal(str(ratio)),
        completed_run_threshold=raw["completed_run_threshold"],
        keep_recent_count=raw["keep_recent_count"],
    )


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of evaluating a capacity/Run commit policy."""

    status: str  # skipped | archived | not_archived | failed
    reason: str | None = None  # pending_capacity | completed_runs | capacity
    archived: bool | None = None
    task_id: str | None = None
    archive_uri: str | None = None
    detail: str | None = None


class OVCommitPolicy(Protocol):
    @property
    def config(self) -> CommitConfig: ...

    async def commit(
        self,
        session_id: str,
        pending_tokens: int,
        runs: int,
        *,
        identity: OVIdentity,
        trigger: CommitTrigger = "policy",
        keep_recent_count: int | None = None,
    ) -> CommitResult: ...


class OpenVikingCommitPolicy:
    """OpenViking commit behavior for LangChain and LangGraph sessions.

    ``mode`` and ``pending_token_threshold`` preserve the original synchronous
    policy fields. ``config`` enables capacity and completed-Run decisions for
    the async context middleware.
    """

    def __init__(
        self,
        config: CommitConfig | str | None = None,
        ov_client: Any = None,
        *,
        mode: Literal["never", "always", "pending_tokens"] = "never",
        pending_token_threshold: int = 8_000,
    ) -> None:
        if isinstance(config, str):
            # Preserve the original positional (mode, pending_token_threshold) form.
            old_mode = config
            if isinstance(ov_client, int) and not isinstance(ov_client, bool):
                pending_token_threshold = ov_client
                ov_client = None
            elif ov_client is not None:
                raise TypeError("A mode policy cannot take an OV client")
            config = None
            mode = old_mode  # type: ignore[assignment]
        if config is not None and not isinstance(config, CommitConfig):
            raise TypeError("config must be a CommitConfig")
        if mode not in ("never", "always", "pending_tokens"):
            raise ValueError(f"Unsupported OpenViking commit mode: {mode}")
        if (
            isinstance(pending_token_threshold, bool)
            or not isinstance(pending_token_threshold, int)
            or pending_token_threshold < 0
        ):
            raise ValueError("pending_token_threshold must be a non-negative integer")
        if config is None and ov_client is not None:
            raise ValueError("config is required when an OV client is supplied")
        self._config = config
        self.ov_client = ov_client
        self.mode = mode
        self.pending_token_threshold = pending_token_threshold

    @property
    def config(self) -> CommitConfig:
        if self._config is None:
            raise ValueError("Capacity/Run commits require a CommitConfig")
        return self._config

    @property
    def has_session_policy(self) -> bool:
        return self._config is not None

    def bind_client(self, ov_client: Any) -> None:
        """Use the middleware's async client when none was supplied."""

        if not self.has_session_policy:
            raise ValueError("Capacity/Run commits require a CommitConfig")
        if self.ov_client is None:
            self.ov_client = ov_client

    def commit_reason(
        self,
        pending_tokens: int,
        runs: int,
        *,
        trigger: CommitTrigger = "policy",
    ) -> str | None:
        """Evaluate thresholds without I/O; receipts may only skip this check."""
        for name, value in (("pending_tokens", pending_tokens), ("runs", runs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if trigger not in ("policy", "capacity"):
            raise ValueError(f"Unsupported commit trigger: {trigger}")

        config = self.config
        capacity_hit = Decimal(pending_tokens) >= (
            config.token_budget * config.commit_token_threshold_ratio
        )
        runs_hit = runs >= config.completed_run_threshold and pending_tokens > 0
        if pending_tokens == 0 or not (capacity_hit or runs_hit or trigger != "policy"):
            return None
        reason = "pending_capacity" if capacity_hit else "completed_runs"
        if trigger != "policy":
            reason = trigger
        return reason

    async def commit(
        self,
        session_id: str,
        pending_tokens: int,
        runs: int,
        *,
        identity: OVIdentity | None = None,
        trigger: CommitTrigger = "policy",
        keep_recent_count: int | None = None,
    ) -> CommitResult:
        if not session_id:
            raise ValueError("session_id must not be empty")
        keep = self.config.keep_recent_count if keep_recent_count is None else keep_recent_count
        if type(keep) is not int or not 0 <= keep <= 10000:
            raise ValueError("keep_recent_count must be an integer from 0 to 10000")
        reason = self.commit_reason(pending_tokens, runs, trigger=trigger)
        if reason is None:
            return CommitResult(status="skipped")
        if self.ov_client is None:
            raise ValueError("An OV client is required for capacity/Run commits")

        call = getattr(self.ov_client, "call", None)
        if callable(call) and not callable(getattr(self.ov_client, "commit_session", None)):
            response = await call(
                "commit_session",
                session_id=session_id,
                keep_recent_count=keep,
                identity=identity,
            )
        else:
            method = self.ov_client.commit_session
            kwargs: dict[str, Any] = {"keep_recent_count": keep}
            try:
                parameters = inspect.signature(method).parameters.values()
            except (TypeError, ValueError):
                accepts_identity = True
            else:
                accepts_identity = any(
                    parameter.name == "identity" or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            if identity is not None and accepts_identity:
                kwargs["identity"] = identity
            response = await method(session_id, **kwargs)

        if not isinstance(response, Mapping) or not isinstance(response.get("archived"), bool):
            return CommitResult(
                status="failed",
                reason=reason,
                detail="OV commit response must contain a boolean 'archived' field",
            )
        archived = response["archived"]
        return CommitResult(
            status="archived" if archived else "not_archived",
            reason=reason,
            archived=archived,
            task_id=response.get("task_id"),
            archive_uri=response.get("archive_uri"),
        )
