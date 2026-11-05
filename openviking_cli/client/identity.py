# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Immutable per-request identity, independent of agent frameworks and credentials."""

from dataclasses import dataclass
from typing import Literal, TypedDict

SessionKey = tuple[str, str, str, str]
ClientAuthMode = Literal["api_key", "trusted", "dev"]


def valid_identity_field(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(c in "/\\" or ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Identity fields must be nonempty, without path/header separators")


@dataclass(frozen=True, slots=True)
class OVIdentity:
    """Caller identity for one request; credentials remain configured on the client."""

    account_id: str
    user_id: str
    actor_peer_id: str

    def __post_init__(self) -> None:
        for value in (self.account_id, self.user_id, self.actor_peer_id):
            valid_identity_field(value)

    def session_key(self, session_id: str) -> SessionKey:
        valid_identity_field(session_id)
        return self.account_id, self.user_id, self.actor_peer_id, session_id

    def headers(self) -> dict[str, str]:
        """Return fresh headers; never retain or mutate a shared client's headers."""
        self.__post_init__()
        return {
            "X-OpenViking-Account": self.account_id,
            "X-OpenViking-User": self.user_id,
            "X-OpenViking-Actor-Peer": self.actor_peer_id,
        }


class _IdentityArguments(TypedDict, total=False):
    identity: OVIdentity


def _identity_arguments(identity: OVIdentity | None) -> _IdentityArguments:
    """Preserve legacy delegated call signatures when identity is omitted."""
    return {} if identity is None else {"identity": identity}
