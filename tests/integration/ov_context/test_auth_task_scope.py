"""Real server auth and task routes with the production SDK."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from openviking.server.auth import get_request_context
from openviking.server.identity import AuthMode, RequestContext, ResolvedIdentity, Role
from openviking.server.routers import tasks as tasks_router
from openviking.service.task_tracker import TaskTracker
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import OpenVikingError, PermissionDeniedError, UnauthenticatedError


class Keys:
    def resolve(self, api_key: str) -> ResolvedIdentity:
        if api_key != "test-key":
            raise UnauthenticatedError("Invalid API Key")
        return ResolvedIdentity(Role.USER, "account-a", "alice")

    def get_user_role(self, account: str, user: str) -> Role:
        return Role.USER


def app_for(mode: str, monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, TaskTracker]:
    store = AsyncMock()
    store.list.return_value = []
    store.get.return_value = None
    tracker = TaskTracker(store)
    monkeypatch.setattr(tasks_router, "get_task_tracker", lambda: tracker)
    app = FastAPI()
    app.state.config = SimpleNamespace(
        get_effective_auth_mode=lambda: AuthMode(mode), root_api_key=None
    )
    app.state.api_key_manager = Keys()
    app.include_router(tasks_router.router)

    @app.exception_handler(OpenVikingError)
    async def error_handler(request, exc: OpenVikingError):
        code = getattr(exc, "code", "INTERNAL")
        return JSONResponse(
            status_code={
                "PERMISSION_DENIED": 403,
                "UNAUTHENTICATED": 401,
                "NOT_FOUND": 404,
                "INVALID_ARGUMENT": 400,
            }.get(code, 500),
            content={"status": "error", "error": {"code": code, "message": str(exc)}},
        )

    @app.get("/api/v1/sessions/{session_id}")
    async def session_meta(session_id: str, ctx: RequestContext = Depends(get_request_context)):
        return {
            "status": "ok",
            "result": {
                "session_id": session_id,
                "commit_count": 0,
                "pending_tokens": 0,
                "account": ctx.account_id,
                "user": ctx.user.user_id,
                "actor": ctx.actor_peer_id,
            },
        }

    return app, tracker


@asynccontextmanager
async def sdk_for(
    app: FastAPI, **options
) -> AsyncIterator[tuple[AsyncHTTPClient, httpx.AsyncClient]]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={
            "x-openviking-account": "borrowed-account",
            "x-openviking-user": "borrowed-user",
            "X-Shared": "keep",
        },
    ) as pool:
        sdk = AsyncHTTPClient(
            url="http://ov.test",
            api_key="test-key",
            http_client=pool,
            profile_enabled=False,
            **options,
        )
        await sdk.initialize()
        try:
            yield sdk, pool
        finally:
            await sdk.close()


async def test_default_api_key_identity_works_without_assertion_headers(monkeypatch):
    app, _ = app_for("api_key", monkeypatch)
    async with sdk_for(
        app,
        account="configured-account",
        user="configured-user",
        extra_headers={"x-openviking-user": "extra-user"},
    ) as (sdk, pool):
        defaults = dict(pool.headers)
        plain = await sdk.get_session("s")
        explicit = await sdk.get_session("s", identity=OVIdentity("account-a", "alice", "mail"))
        assert (plain["account"], plain["user"]) == ("account-a", "alice")
        assert (explicit["account"], explicit["user"], explicit["actor"]) == (
            "account-a",
            "alice",
            "mail",
        )
        assert dict(pool.headers) == defaults
        assert not pool.is_closed


@pytest.mark.parametrize("mode", ["trusted", "dev"])
async def test_assertion_modes_keep_per_request_identity(monkeypatch, mode):
    app, _ = app_for(mode, monkeypatch)
    async with sdk_for(app, auth_mode=mode) as (sdk, pool):
        defaults = dict(pool.headers)
        for identity in (OVIdentity("a", "alice", "mail"), OVIdentity("b", "bob", "doc")):
            result = await sdk.get_session("s", identity=identity)
            assert (result["account"], result["user"], result["actor"]) == (
                identity.account_id,
                identity.user_id,
                identity.actor_peer_id,
            )
        assert dict(pool.headers) == defaults


async def test_nonroot_cannot_query_another_task_owner(monkeypatch):
    app, tracker = app_for("api_key", monkeypatch)
    own = await tracker.create("session_commit", "s", account_id="account-a", user_id="alice")
    foreign = await tracker.create("session_commit", "s", account_id="account-a", user_id="bob")
    async with sdk_for(app) as (sdk, _):
        alice = OVIdentity("account-a", "alice", "mail")
        tasks = await sdk.list_tasks(resource_id="s", identity=alice)
        assert [task["task_id"] for task in tasks] == [own.task_id]
        bob = OVIdentity("account-a", "bob", "mail")
        with pytest.raises(PermissionDeniedError):
            await sdk.list_tasks(resource_id="s", identity=bob)
        with pytest.raises(PermissionDeniedError):
            await sdk.get_task(foreign.task_id, identity=bob)


async def test_task_scope_is_applied_before_limit(monkeypatch):
    app, tracker = app_for("dev", monkeypatch)
    own = await tracker.create("session_commit", "s", account_id="a", user_id="alice")
    for _ in range(5):
        await tracker.create("session_commit", "s", account_id="a", user_id="bob")
    async with sdk_for(app, auth_mode="dev") as (sdk, _):
        tasks = await sdk.list_tasks(
            resource_id="s", limit=1, identity=OVIdentity("a", "alice", "mail")
        )
        assert [task["task_id"] for task in tasks] == [own.task_id]


async def test_api_key_server_still_rejects_raw_identity_assertions(monkeypatch):
    app, _ = app_for("api_key", monkeypatch)
    async with httpx.AsyncClient(
        base_url="http://ov.test", transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.get(
            "/api/v1/sessions/s",
            headers={
                "X-API-Key": "test-key",
                "X-OpenViking-Account": "account-a",
                "X-OpenViking-User": "alice",
            },
        )
        assert response.status_code == 403


async def test_task_scope_requires_both_account_and_user(monkeypatch):
    app, _ = app_for("dev", monkeypatch)
    async with httpx.AsyncClient(
        base_url="http://ov.test", transport=httpx.ASGITransport(app=app)
    ) as client:
        for path in ("/api/v1/tasks", "/api/v1/tasks/unknown"):
            response = await client.get(path, params={"account_id": "a"})
            assert response.status_code == 400
