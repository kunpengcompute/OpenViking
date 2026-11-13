# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-session automatic commit policy and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openviking_cli.exceptions import InvalidArgumentError

DEFAULT_PENDING_TOKEN_THRESHOLD = 150_000
DEFAULT_MESSAGE_COUNT_THRESHOLD = 100
DEFAULT_IDLE_TIMEOUT_SECONDS = 86_400
DEFAULT_KEEP_RECENT_COUNT = 0
DEFAULT_MIN_COMMIT_INTERVAL_SECONDS = 0

MIN_IDLE_TIMEOUT_SECONDS = 300
MAX_PENDING_TOKEN_THRESHOLD = 1_000_000
MAX_MESSAGE_COUNT_THRESHOLD = 1_000
MAX_IDLE_TIMEOUT_SECONDS = 604_800

_POLICY_KEYS = {
    "pending_token_threshold",
    "message_count_threshold",
    "idle_timeout_seconds",
    "keep_recent_count",
    "min_commit_interval_seconds",
}


def _coerce_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidArgumentError(f"auto_commit_policy.{field} must be an integer") from exc
    return min(max(parsed, minimum), maximum)


def validate_idle_timeout_input(value: Any) -> int:
    """Validate a newly supplied idle timeout without changing stored policies."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("auto_commit_policy.idle_timeout_seconds must be an integer") from exc
    if parsed != 0 and parsed < MIN_IDLE_TIMEOUT_SECONDS:
        raise ValueError(
            "auto_commit_policy.idle_timeout_seconds must be 0 or at least "
            f"{MIN_IDLE_TIMEOUT_SECONDS}"
        )
    return parsed


@dataclass
class AutoCommitPolicy:
    """Effective automatic-commit policy for one session."""

    pending_token_threshold: int = DEFAULT_PENDING_TOKEN_THRESHOLD
    message_count_threshold: int = DEFAULT_MESSAGE_COUNT_THRESHOLD
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    keep_recent_count: int = DEFAULT_KEEP_RECENT_COUNT
    min_commit_interval_seconds: int = DEFAULT_MIN_COMMIT_INTERVAL_SECONDS

    @classmethod
    def default(cls) -> "AutoCommitPolicy":
        return cls()

    @classmethod
    def global_idle_default(cls, idle_timeout_seconds: int) -> "AutoCommitPolicy":
        """Keep global-only sessions on idle commits; clients retain token triggers."""
        return cls(
            pending_token_threshold=0,
            message_count_threshold=0,
            idle_timeout_seconds=max(0, int(idle_timeout_seconds)),
        )

    @classmethod
    def disabled(cls) -> "AutoCommitPolicy":
        return cls(
            pending_token_threshold=0,
            message_count_threshold=0,
            idle_timeout_seconds=0,
            keep_recent_count=0,
            min_commit_interval_seconds=0,
        )

    @classmethod
    def from_dict(cls, data: Any) -> "AutoCommitPolicy":
        """Fill defaults and clamp a policy object to supported bounds."""
        if isinstance(data, AutoCommitPolicy):
            return data
        if not isinstance(data, dict):
            raise InvalidArgumentError("auto_commit_policy must be an object")

        extra_keys = set(data) - _POLICY_KEYS
        if extra_keys:
            raise InvalidArgumentError(
                "auto_commit_policy supports only: " + ", ".join(sorted(_POLICY_KEYS))
            )

        default = cls.default()
        return cls(
            pending_token_threshold=_coerce_int(
                data.get("pending_token_threshold", default.pending_token_threshold),
                field="pending_token_threshold",
                minimum=0,
                maximum=MAX_PENDING_TOKEN_THRESHOLD,
            ),
            message_count_threshold=_coerce_int(
                data.get("message_count_threshold", default.message_count_threshold),
                field="message_count_threshold",
                minimum=0,
                maximum=MAX_MESSAGE_COUNT_THRESHOLD,
            ),
            idle_timeout_seconds=_coerce_int(
                data.get("idle_timeout_seconds", default.idle_timeout_seconds),
                field="idle_timeout_seconds",
                minimum=0,
                maximum=MAX_IDLE_TIMEOUT_SECONDS,
            ),
            keep_recent_count=_coerce_int(
                data.get("keep_recent_count", default.keep_recent_count),
                field="keep_recent_count",
                minimum=0,
                maximum=MAX_MESSAGE_COUNT_THRESHOLD,
            ),
            min_commit_interval_seconds=_coerce_int(
                data.get(
                    "min_commit_interval_seconds",
                    default.min_commit_interval_seconds,
                ),
                field="min_commit_interval_seconds",
                minimum=0,
                maximum=MAX_IDLE_TIMEOUT_SECONDS,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "pending_token_threshold": self.pending_token_threshold,
            "message_count_threshold": self.message_count_threshold,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "keep_recent_count": self.keep_recent_count,
            "min_commit_interval_seconds": self.min_commit_interval_seconds,
        }


def resolve_session_auto_commit_policy(
    meta: dict[str, Any], config: Any
) -> AutoCommitPolicy | None:
    """Resolve a stored override or inherit the server-wide idle policy."""
    configured = bool(
        meta.get(
            "auto_commit_policy_configured",
            "auto_commit_policy" in meta,
        )
    )
    if configured:
        policy = meta.get("auto_commit_policy")
        if policy is None:
            return None
        return AutoCommitPolicy.from_dict(policy)

    if not getattr(config, "enabled", False):
        return None
    timeout = int(getattr(config, "idle_timeout_seconds", 0) or 0)
    if timeout <= 0:
        return None
    return AutoCommitPolicy.global_idle_default(timeout)
