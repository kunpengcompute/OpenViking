# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Invocation identity and session identifiers."""

from __future__ import annotations

from dataclasses import dataclass

from openviking_cli.client.identity import OVIdentity, SessionKey, valid_identity_field


def valid_id(value: str) -> None:
    valid_identity_field(value)


@dataclass(frozen=True)
class RunIdentity:
    account_id: str
    user_id: str
    actor_peer_id: str
    session_id: str
    run_id: str
    execution_scope_id: str = "main"

    def __post_init__(self) -> None:
        for value in vars(self).values():
            valid_id(value)

    @property
    def sdk_identity(self) -> OVIdentity:
        return OVIdentity(self.account_id, self.user_id, self.actor_peer_id)

    @property
    def session_key(self) -> SessionKey:
        return self.sdk_identity.session_key(self.session_id)


__all__ = ["OVIdentity", "SessionKey", "RunIdentity", "valid_id"]
