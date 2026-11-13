# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Offline contracts through the real reader, SDK, router and search service."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openviking.core.namespace import canonicalize_uri
from openviking.models.embedder.base import EmbedResult, embed_compat
from openviking.retrieve.intent_analyzer import IntentAnalyzer
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.server.routers import search as search_router
from openviking.service.search_service import SearchService
from openviking.storage.expr import And, Eq, Or, PathScope, RawDSL
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.exceptions import OpenVikingError, PermissionDeniedError
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext
from openviking_cli.session.user_id import UserIdentifier


class Embedder:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.release = asyncio.Event()
        self.release.set()
        self.started = asyncio.Event()
        self.cancelled = 0

    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, *, is_query):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
            await asyncio.sleep(0)
            if self.fail:
                raise RuntimeError("fixture model unavailable")
            return EmbedResult(dense_vector=[1.0, 0.0])
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def context(user="alice"):
    return RequestContext(UserIdentifier("tenant", user), Role.USER, actor_peer_id="assistant")


def matches(expr, item):
    if expr is None:
        return True
    if isinstance(expr, And):
        return all(matches(e, item) for e in expr.conds)
    if isinstance(expr, Or):
        return any(matches(e, item) for e in expr.conds)
    if isinstance(expr, PathScope):
        return item.uri == expr.path or item.uri.startswith(expr.path.rstrip("/") + "/")
    if isinstance(expr, Eq):
        return getattr(item, expr.field) == expr.value
    if isinstance(expr, RawDSL):
        expr = expr.payload
    if expr["op"] == "contains":
        return expr["substring"] in getattr(item, expr["field"])
    if expr["op"] == "must":
        if expr.get("field") == "uri" and expr.get("para", "") != "-d=0":
            return any(
                item.uri == root or item.uri.startswith(root.rstrip("/") + "/")
                for root in expr["conds"]
            )
        return getattr(item, expr["field"]) in expr["conds"]
    if expr["op"] == "and":
        return all(matches(e, item) for e in expr["conds"])
    if expr["op"] == "or":
        return any(matches(e, item) for e in expr["conds"])
    raise AssertionError(expr)


class FixtureFS:
    def __init__(self):
        self.embedder = Embedder()
        self.calls = []
        self.reads = []
        self.error = None
        self.read_error = None
        self.search = AsyncMock(side_effect=AssertionError("Context must not enter session search"))
        self.categories = {
            "events": 6,
            "entities": 2,
            "preferences": 2,
            "experiences": 2,
            "cases": 1,
        }
        self.extra_items = []
        self.body = "## Situation\nAn API timed out.\n\n## Approach\nRetry once using the recorded request ID.\n\n## Reflect\nCheck the outcome."

    async def find(self, query, *, ctx, target_uri, limit, filter=None, **kwargs):
        self.calls.append((query, ctx, target_uri, limit, filter, kwargs))
        await embed_compat(self.embedder, query, is_query=True)
        if self.error:
            raise self.error
        roots = [target_uri] if isinstance(target_uri, str) else target_uri
        roots = [canonicalize_uri(r, ctx) for r in roots if r]
        items = []
        for cat, count in self.categories.items():
            for n in range(count):
                uri = f"viking://user/{ctx.user.user_id}/memories/{cat}/{n}.md"
                item = MatchedContext(
                    uri=uri,
                    context_type=ContextType.MEMORY,
                    category=cat,
                    score=(0.95 if cat == "events" else 0.6) - n / 100,
                    abstract=f"{cat} fact {n}",
                )
                items.append(item)
        items.extend(self.extra_items)
        items = [
            item
            for item in items
            if (
                not roots
                or any(
                    item.uri == root or item.uri.startswith(root.rstrip("/") + "/")
                    for root in roots
                )
            )
            and matches(filter, item)
        ]
        items.sort(key=lambda i: -i.score)
        selected = items[:limit]
        return FindResult(
            memories=[i for i in selected if i.context_type == ContextType.MEMORY],
            resources=[i for i in selected if i.context_type == ContextType.RESOURCE],
            skills=[i for i in selected if i.context_type == ContextType.SKILL],
        )

    async def read(self, uri, *, ctx):
        self.reads.append((uri, ctx))
        if self.read_error:
            raise self.read_error
        return self.body.encode()


@pytest_asyncio.fixture
async def api(monkeypatch):
    monkeypatch.setattr(
        IntentAnalyzer,
        "analyze",
        AsyncMock(side_effect=AssertionError("Intent analysis forbidden")),
    )
    fs = FixtureFS()
    service = SimpleNamespace(
        search=SearchService(fs),
        sessions=SimpleNamespace(
            session=AsyncMock(side_effect=AssertionError("Session access forbidden"))
        ),
    )
    monkeypatch.setattr(search_router, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(search_router.router)

    async def identity(request: Request):
        return context(request.headers.get("X-OpenViking-User", "alice"))

    app.dependency_overrides[get_request_context] = identity

    @app.get("/api/v1/content/read")
    async def read_content(request: Request, uri: str, offset: int = 0, limit: int = -1):
        value = await fs.read(uri, ctx=await identity(request))
        value = value.decode() if isinstance(value, bytes) else value
        return {
            "status": "ok",
            "result": value[offset:] if limit < 0 else value[offset : offset + limit],
        }

    @app.exception_handler(OpenVikingError)
    async def error_handler(request, exc):
        return JSONResponse(
            status_code=403 if isinstance(exc, PermissionDeniedError) else 400,
            content={"status": "error", "error": {"code": exc.code, "message": str(exc)}},
        )

    transport = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture")
    sdk = AsyncHTTPClient(
        url="http://fixture",
        user="alice",
        account="tenant",
        auth_mode="trusted",
        profile_enabled=False,
        http_client=transport,
    )
    await sdk.initialize()
    try:
        yield fs, service, sdk
    finally:
        await sdk.close()
        await transport.aclose()


def hit(uri, context_type=ContextType.MEMORY, score=0.6, *, level=2, category=""):
    return MatchedContext(
        uri=uri,
        context_type=context_type,
        category=category,
        score=score,
        abstract=f"fact {uri}",
        level=level,
    )
