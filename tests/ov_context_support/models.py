# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Identity and retrieval settings for offline transport fixtures only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypedDict

from openviking.integrations.langchain.ov_context.identity import RunIdentity, valid_id


@dataclass(frozen=True)
class ResolvedCaller:
    """Provided by the host's credential adapter, never inferred from message text."""

    service_id: str
    account_id: str
    user_id: str
    authorization_fingerprint: str
    auth_mode: Literal["api_key", "trusted", "dev"] = "api_key"

    def __post_init__(self) -> None:
        for value in (
            self.service_id,
            self.account_id,
            self.user_id,
            self.authorization_fingerprint,
        ):
            valid_id(value)
        if self.auth_mode not in ("api_key", "trusted", "dev"):
            raise ValueError("Unsupported authentication mode")

    def headers(self, identity: RunIdentity) -> dict[str, str]:
        if (self.account_id, self.user_id) != (identity.account_id, identity.user_id):
            raise ValueError("Run owner does not match the authenticated caller")
        result = {"X-OpenViking-Actor-Peer": identity.actor_peer_id}
        if self.auth_mode in ("trusted", "dev"):
            result.update(
                {"X-OpenViking-Account": self.account_id, "X-OpenViking-User": self.user_id}
            )
        return result

    def partition(self, identity: RunIdentity) -> tuple[str, ...]:
        self.headers(identity)
        return (
            self.service_id,
            identity.account_id,
            identity.user_id,
            identity.session_id,
            identity.execution_scope_id,
            identity.actor_peer_id,
            identity.run_id,
            self.authorization_fingerprint,
        )


def resolve_run_identity(
    caller: ResolvedCaller,
    session: Mapping[str, str],
    *,
    actor_peer_id: str,
    run_id: str,
    execution_scope_id: str = "main",
) -> RunIdentity:
    """Use the existing host-authorized session record; no binding table.

    Host authorizes session access and the peer before calling this method.
    An optional ov_session_id comes from that record, never from user text.
    """
    if (session["account_id"], session["user_id"]) != (caller.account_id, caller.user_id):
        raise ValueError("Session owner does not match the authenticated caller")
    return RunIdentity(
        caller.account_id,
        caller.user_id,
        actor_peer_id,
        session.get("ov_session_id") or session["session_id"],
        run_id,
        execution_scope_id,
    )


@dataclass(frozen=True)
class RecallPolicy:
    target_uri: tuple[str, ...] = ()
    context_type: str = "memory"
    memory_types: tuple[str, ...] = ("preferences", "profile")
    search_mode: Literal["search", "find"] = "search"
    limit: int = 5
    score_threshold: float | None = None
    content_mode: Literal["overview", "full"] = "overview"
    max_recall_tokens: int = 2048

    def __post_init__(self) -> None:
        if isinstance(self.target_uri, str) or isinstance(self.memory_types, str):
            raise ValueError("Recall targets and memory types must be sequences, not a string")
        if any(
            not isinstance(value, str) or not value
            for value in (*self.target_uri, *self.memory_types)
        ):
            raise ValueError("Recall targets and memory types must be nonempty strings")
        if type(self.limit) is not int or type(self.max_recall_tokens) is not int:
            raise ValueError("Recall limits must be integers")
        object.__setattr__(self, "target_uri", tuple(sorted(set(self.target_uri))))
        object.__setattr__(self, "memory_types", tuple(sorted(set(self.memory_types))))
        if self.search_mode not in ("search", "find") or self.content_mode not in (
            "overview",
            "full",
        ):
            raise ValueError("Invalid recall mode")
        if self.limit < 1 or self.max_recall_tokens < 0:
            raise ValueError("Invalid recall limits")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")
        if self.context_type not in ("memory", "resource", "skill"):
            raise ValueError("Invalid context_type")
        if any(not uri.startswith("viking://") for uri in self.target_uri):
            raise ValueError("Recall targets must be Viking URIs")


class SessionContextView(TypedDict, total=False):
    """Native session material; not a model message list or a coverage manifest."""

    session_id: str
    latest_archive_overview: str
    pre_archive_abstracts: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    estimatedTokens: int
    stats: dict[str, Any]
    status: str
    host_messages_authoritative: bool
