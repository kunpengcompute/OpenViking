# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""SDK-shaped adapter for the offline HTTP fixture."""

from __future__ import annotations

from typing import Any

from openviking.integrations.langchain.ov_context.identity import OVIdentity, RunIdentity
from tests.ov_context_support.transport import NativeHTTPTransport


class TransportSDKAdapter:
    """Compatibility for the previous low-level transport constructor."""

    def __init__(self, transport: NativeHTTPTransport, actor_peer_id: str = "default") -> None:
        self.transport, self.actor_peer_id = transport, actor_peer_id
        self._initialized = True  # The fixture supplies an already-open HTTP transport.

    def identity(
        self, session_id: str = "retrieval", *, identity: OVIdentity | None = None
    ) -> RunIdentity:
        caller = self.transport.caller
        selected = identity or OVIdentity(caller.account_id, caller.user_id, self.actor_peer_id)
        result = RunIdentity(
            selected.account_id, selected.user_id, selected.actor_peer_id, session_id, "sdk"
        )
        caller.headers(result)
        return result

    async def get_session(
        self, session_id: str, auto_create: bool = False, *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        return await self.transport.request(
            self.identity(session_id, identity=identity),
            "GET",
            self.transport.session_path(self.identity(session_id, identity=identity)),
            event="session.get",
        )

    async def create_session(
        self,
        session_id: str,
        memory_policy: dict[str, Any] | None = None,
        *,
        identity: OVIdentity | None = None,
    ) -> dict[str, Any]:
        await self.transport.ensure_session(
            self.identity(session_id, identity=identity), memory_policy
        )
        return {"session_id": session_id}

    async def get_session_context(
        self, session_id: str, token_budget: int, *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        return await self.transport.context(
            self.identity(session_id, identity=identity), token_budget
        )

    async def batch_add_messages(
        self, session_id: str, messages: list[dict[str, Any]], *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        return await self.transport.batch(self.identity(session_id, identity=identity), messages)

    async def find(
        self,
        query: str,
        target_uri: str | list[str],
        limit: int,
        score_threshold: float | None = None,
        context_type: str = "memory",
        *,
        identity: OVIdentity | None = None,
    ) -> dict[str, Any]:
        from tests.ov_context_support.models import RecallPolicy

        return await self.transport.recall(
            self.identity(identity=identity),
            query,
            RecallPolicy(
                target_uri=tuple(target_uri)
                if isinstance(target_uri, list)
                else (target_uri,)
                if target_uri
                else (),
                limit=limit,
                score_threshold=score_threshold,
                search_mode="find",
                memory_types=(),
                context_type=context_type,
            ),
        )

    async def read(self, uri: str, *, identity: OVIdentity | None = None) -> str:
        return await self.transport.read(self.identity(identity=identity), uri)

    async def commit_session(
        self, session_id: str, keep_recent_count: int, *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        return await self.transport.commit(
            self.identity(session_id, identity=identity), keep_recent_count
        )

    async def get_task(self, task_id: str, *, identity: OVIdentity | None = None) -> dict[str, Any]:
        return await self.transport.task_status(self.identity(identity=identity), task_id)

    async def list_tasks(
        self,
        *,
        task_type: str,
        resource_id: str,
        status: str,
        limit: int = 50,
        identity: OVIdentity | None = None,
    ) -> list[dict[str, Any]]:
        return await self.transport.request(
            self.identity(resource_id, identity=identity),
            "GET",
            "/api/v1/tasks",
            params={
                "task_type": task_type,
                "resource_id": resource_id,
                "status": status,
                "limit": limit,
            },
            event="task.list",
        )

    async def get_session_archive(
        self, session_id: str, archive_id: str, *, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        from urllib.parse import quote

        return await self.transport.request(
            self.identity(session_id, identity=identity),
            "GET",
            self.transport.session_path(self.identity(session_id, identity=identity))
            + "/archives/"
            + quote(archive_id, safe=""),
            event="archive",
        )

    async def _tool_result(
        self,
        session_id: str,
        tool_result_id: str = "",
        suffix: str = "",
        *,
        identity: OVIdentity | None = None,
        **params: str | int,
    ) -> dict[str, Any]:
        from urllib.parse import quote

        run_identity = self.identity(session_id, identity=identity)
        path = self.transport.session_path(run_identity) + "/tool-results"
        if tool_result_id:
            path += "/" + quote(tool_result_id, safe="")
        return await self.transport.request(
            run_identity, "GET", path + suffix, params=params, event="tool_result"
        )

    async def read_tool_result(
        self,
        session_id: str,
        tool_result_id: str,
        *,
        offset: int = 0,
        limit: int = 20_000,
        identity: OVIdentity | None = None,
    ) -> dict[str, Any]:
        return await self._tool_result(
            session_id, tool_result_id, identity=identity, offset=offset, limit=limit
        )

    async def search_tool_result(
        self,
        session_id: str,
        tool_result_id: str,
        *,
        query: str,
        limit: int = 20,
        identity: OVIdentity | None = None,
    ) -> dict[str, Any]:
        return await self._tool_result(
            session_id, tool_result_id, identity=identity, suffix="/search", q=query, limit=limit
        )

    async def list_tool_results(
        self, session_id: str, *, limit: int = 50, identity: OVIdentity | None = None
    ) -> dict[str, Any]:
        return await self._tool_result(session_id, identity=identity, limit=limit)
