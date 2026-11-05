# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Lazy async SDK ownership; no sync network calls in graph hooks."""

from __future__ import annotations

import asyncio
import inspect
from math import isfinite
from typing import Any, Mapping, Protocol, TypedDict

from openviking.integrations.langchain.client import OpenVikingConnection
from openviking_cli._exception_utils import auth_error as auth_error

from .identity import OVIdentity


def not_found(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 404 or getattr(exc, "code", None) == "NOT_FOUND"


def already_exists(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == "ALREADY_EXISTS"


class SessionInfo(TypedDict, total=False):
    session_id: str
    pending_tokens: int
    archive_id: str | None
    commit_count: int
    latest_finished_archive_id: str | None


class CommitReceipt(TypedDict, total=False):
    archived: bool
    task_id: str | None
    archive_uri: str | None


class TaskInfo(TypedDict, total=False):
    task_id: str
    status: str


class OVSessionClient(Protocol):
    """Session SDK contract. Identity is per invocation; DeepAgent owns the SDK."""

    async def get_session(
        self, session_id: str, *, identity: OVIdentity, auto_create: bool = False
    ) -> Mapping[str, Any]: ...

    async def create_session(
        self, session_id: str, *, identity: OVIdentity, memory_policy: dict[str, Any] | None = None
    ) -> Mapping[str, Any]: ...

    async def commit_session(
        self, session_id: str, *, identity: OVIdentity, keep_recent_count: int
    ) -> Mapping[str, Any]: ...

    async def get_task(self, task_id: str, *, identity: OVIdentity) -> Mapping[str, Any] | None: ...

    async def list_tasks(
        self, *, task_type: str, resource_id: str, status: str, limit: int, identity: OVIdentity
    ) -> list[dict[str, Any]]: ...


class AsyncClientHandle:
    """Initialize owned SDKs; borrow application-initialized SDKs without mutation."""

    def __init__(self, connection: OpenVikingConnection) -> None:
        self.connection = connection
        self.client = connection.client
        self.owned = self.client is None
        self.ready = False
        self.lock = asyncio.Lock()

    async def get(self) -> Any:
        async with self.lock:
            if not self.owned:
                self.ready = False
                http = getattr(self.client, "_http", None)
                if http is not None and getattr(http, "is_closed", False):
                    raise RuntimeError("The application-owned HTTP client is closed")
                if http is None and not getattr(self.client, "_initialized", False):
                    raise RuntimeError("Initialize the borrowed async OV client in the application")
                self.ready = True
                return self.client
            if self.ready:
                return self.client
            if self.client is None:
                if self.connection.path:
                    raise ValueError("Async OV context requires an HTTP URL or async client")
                from openviking_cli.client.http import AsyncHTTPClient

                if not self.connection.url or not self.connection.api_key:
                    raise ValueError("Self-created async SDK requires url and a service api_key")
                if not isfinite(self.connection.timeout) or self.connection.timeout <= 0:
                    raise ValueError("SDK timeout must be a positive finite number")
                if self.connection.auth_mode != "trusted":
                    raise ValueError("Async middleware requires trusted SDK authentication")
                self.client = AsyncHTTPClient(
                    **{
                        name: getattr(self.connection, name)
                        for name in (
                            "url",
                            "api_key",
                            "account",
                            "user",
                            "user_id",
                            "actor_peer_id",
                            "timeout",
                            "extra_headers",
                            "auth_mode",
                            "http_client",
                        )
                    }
                )
            initialize = getattr(self.client, "initialize", None)
            if (
                initialize
                and getattr(self.client, "_http", None) is None
                and not getattr(self.client, "_initialized", False)
            ):
                if not inspect.iscoroutinefunction(initialize):
                    raise TypeError("Use an asynchronous OV client with ainvoke/astream")
                await initialize()
            self.ready = True
            return self.client

    async def call(self, method: str, *, identity: OVIdentity, **kwargs: Any) -> Any:
        if not isinstance(identity, OVIdentity):
            raise ValueError("SDK operations require an explicit OVIdentity")
        client = await self.get()
        function = getattr(client, method)
        if not inspect.iscoroutinefunction(function):
            raise TypeError(f"OV client.{method} must be asynchronous")
        return await function(identity=identity, **kwargs)

    async def close(self) -> None:
        if self.owned and self.client is not None:
            await self.client.close()
        self.ready = False
