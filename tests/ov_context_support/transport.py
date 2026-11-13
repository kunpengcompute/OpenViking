# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native OV 0.4.1 HTTP routes with a fresh identity header on every request."""

from __future__ import annotations

import json
from copy import deepcopy
from time import perf_counter
from typing import Any
from urllib.parse import quote

import httpx

from openviking.integrations.langchain.ov_context.identity import RunIdentity, valid_id
from openviking.integrations.langchain.ov_context.metrics import Metrics
from tests.ov_context_support.models import ResolvedCaller


class OVHTTPError(RuntimeError):
    def __init__(self, status_code: int, code: str = "") -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(f"OpenViking request failed: HTTP {status_code}")


class NativeHTTPTransport:
    """One authenticated principal per adapter; underlying HTTP connections may be shared."""

    def __init__(
        self,
        *,
        url: str,
        caller: ResolvedCaller,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 60,
        http_client: httpx.AsyncClient | None = None,
        metrics: Metrics | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if api_key and bearer_token:
            raise ValueError("Choose API key or bearer credentials")
        if caller.auth_mode == "api_key" and not (api_key or bearer_token):
            raise ValueError("API-key mode requires explicit credentials")
        if timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("Invalid HTTP limits")
        if http_client is not None:
            protected = {
                "authorization",
                "x-api-key",
                "x-openviking-account",
                "x-openviking-user",
                "x-openviking-actor-peer",
            }
            if (
                protected.intersection(http_client.headers)
                or http_client.params
                or http_client.auth is not None
            ):
                raise ValueError(
                    "Shared HTTP clients must not have identity defaults or query parameters"
                )
        parsed = httpx.URL(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.host
            or parsed.userinfo
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use an HTTP base URL without credentials or query")
        self.caller = caller
        self.url = str(parsed).rstrip("/")
        self._credentials = (
            {"X-API-Key": api_key}
            if api_key
            else {"Authorization": f"Bearer {bearer_token}"}
            if bearer_token
            else {}
        )
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient(timeout=timeout)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.metrics = metrics or Metrics()

    async def request(
        self, identity: RunIdentity, method: str, path: str, *, event: str, **kwargs: Any
    ) -> Any:
        headers = {**self._credentials, **self.caller.headers(identity)}
        start = perf_counter()
        status = "error"
        try:
            async with self.http.stream(
                method,
                self.url + path,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=False,
                auth=None,
                **kwargs,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise OVHTTPError(response.status_code)
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise ValueError("OV response exceeds the configured byte limit")
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise OVHTTPError(response.status_code, "INVALID_RESPONSE")
            status = "ok"
            return payload.get("result")
        finally:
            self.metrics.emit(
                event,
                run_id=identity.run_id,
                scope_id=identity.execution_scope_id,
                duration_ms=(perf_counter() - start) * 1000,
                status=status,
            )

    @staticmethod
    def session_path(identity: RunIdentity) -> str:
        return "/api/v1/sessions/" + quote(identity.session_id, safe="")

    async def ensure_session(
        self, identity: RunIdentity, memory_policy: dict[str, Any] | None
    ) -> None:
        path = self.session_path(identity)
        try:
            await self.request(identity, "GET", path, event="session.get")
            return
        except OVHTTPError as exc:
            if exc.status_code != 404:
                raise
        body: dict[str, Any] = {"session_id": identity.session_id}
        if memory_policy is not None:
            body["memory_policy"] = deepcopy(memory_policy)
        try:
            await self.request(
                identity, "POST", "/api/v1/sessions", event="session.create", json=body
            )
        except OVHTTPError as exc:
            if exc.status_code != 409:
                raise
            await self.request(identity, "GET", path, event="session.get")

    async def context(self, identity: RunIdentity, token_budget: int) -> dict[str, Any]:
        result = await self.request(
            identity,
            "GET",
            self.session_path(identity) + "/context",
            event="context",
            params={"token_budget": token_budget},
        )
        if not isinstance(result, dict):
            raise ValueError("Invalid session context response")
        return result

    async def recall(self, identity: RunIdentity, query: str, policy: Any) -> dict[str, Any]:
        # No session_id or enable_intent field: preserve the native no-intent path.
        payload = {
            "query": query,
            "target_uri": list(policy.target_uri),
            "context_type": policy.context_type,
            "limit": policy.limit,
            "score_threshold": policy.score_threshold,
        }
        if policy.content_mode == "full":
            payload["level"] = 2  # Native file-level filter: content/read cannot read directories.
        return await self.request(
            identity, "POST", "/api/v1/search/" + policy.search_mode, event="search", json=payload
        )

    async def read(self, identity: RunIdentity, uri: str) -> str:
        result = await self.request(
            identity,
            "GET",
            "/api/v1/content/read",
            event="read",
            params={"uri": uri, "offset": 0, "limit": -1},
        )
        if not isinstance(result, str):
            raise ValueError("Invalid content/read response")
        return result

    async def batch(self, identity: RunIdentity, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.request(
            identity,
            "POST",
            self.session_path(identity) + "/messages/batch",
            event="batch",
            json={"messages": messages},
        )

    async def commit(self, identity: RunIdentity, keep_recent_count: int) -> dict[str, Any]:
        result = await self.request(
            identity,
            "POST",
            self.session_path(identity) + "/commit",
            event="commit",
            json={"keep_recent_count": keep_recent_count},
        )
        if not isinstance(result, dict):
            raise ValueError("Invalid commit response")
        self.metrics.emit(
            "commit.result",
            run_id=identity.run_id,
            scope_id=identity.execution_scope_id,
            task_id=result.get("task_id"),
            commit_status=result.get("status"),
            archived=result.get("archived"),
        )
        return result

    async def task_status(self, identity: RunIdentity, task_id: str) -> dict[str, Any]:
        """Diagnostic read, outside the model/commit critical path. No automatic polling."""
        valid_id(task_id)
        result = await self.request(
            identity, "GET", "/api/v1/tasks/" + quote(task_id, safe=""), event="task"
        )
        if not isinstance(result, dict):
            raise ValueError("Invalid task response")
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()
