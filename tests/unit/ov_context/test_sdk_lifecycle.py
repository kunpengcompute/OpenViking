from unittest.mock import AsyncMock

import httpx
import pytest

from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import RunIdentity
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking_cli.client.http import AsyncHTTPClient


async def test_owned_sdk_uses_trusted_identity_without_closing_the_borrowed_pool():
    import asyncio

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "ok", "result": {"session_id": "s"}})

    pool = httpx.AsyncClient(transport=httpx.MockTransport(respond), headers={"X-Pool": "shared"})
    before = dict(pool.headers)
    handle = AsyncClientHandle(
        OpenVikingConnection(
            url="http://ov.test",
            api_key="test-service-credential",
            http_client=pool,
            timeout=3,
        )
    )
    try:
        one, two = await asyncio.gather(handle.get(), handle.get())
        assert one is two
        identities = [
            RunIdentity("a", user, "p", "s", "r").sdk_identity for user in ("alice", "bob")
        ]
        await asyncio.gather(
            *(handle.call("get_session", session_id="s", identity=i) for i in identities)
        )
        assert {r.headers["X-OpenViking-User"] for r in requests} == {"alice", "bob"}
        assert all(r.headers["X-API-Key"] == "test-service-credential" for r in requests)
        assert dict(pool.headers) == before
        await handle.close()
        assert not pool.is_closed and one._http is None
    finally:
        await pool.aclose()


@pytest.mark.parametrize("auto_initialize", [True, False])
async def test_borrowed_sdk_requires_application_initialization(auto_initialize):
    async with httpx.AsyncClient() as pool:
        sdk = AsyncHTTPClient(url="http://ov.test", http_client=pool)
        sdk.initialize = AsyncMock(wraps=sdk.initialize)
        handle = AsyncClientHandle(
            OpenVikingConnection(client=sdk, auto_initialize=auto_initialize)
        )

        with pytest.raises(RuntimeError, match="Initialize the borrowed"):
            await handle.get()
        sdk.initialize.assert_not_awaited()
        assert not handle.ready
        assert sdk._http is None and not handle.ready

        await sdk.initialize()
        try:
            assert await handle.get() is sdk
            assert sdk.initialize.await_count == 1
        finally:
            await sdk.close()


async def test_borrowed_sdk_and_pool_remain_available_after_handle_close():
    def respond(request):
        return httpx.Response(200, json={"status": "ok", "result": {"session_id": "s"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as pool:
        sdk = AsyncHTTPClient(url="http://ov.test", http_client=pool)
        await sdk.initialize()
        sdk.initialize = AsyncMock(wraps=sdk.initialize)
        sdk.close = AsyncMock(wraps=sdk.close)
        handle = AsyncClientHandle(OpenVikingConnection(client=sdk))
        try:
            assert await handle.get() is sdk
            await handle.close()
            sdk.initialize.assert_not_awaited()
            sdk.close.assert_not_awaited()
            assert not handle.ready
            assert sdk._http is pool and not pool.is_closed
            assert (await sdk.get_session("s"))["session_id"] == "s"
        finally:
            await sdk.close()
        assert sdk._http is None and not pool.is_closed


async def test_borrowed_sdk_closed_by_application_is_not_reinitialized():
    sdk = AsyncHTTPClient(url="http://ov.test")
    await sdk.initialize()
    pool = sdk._http
    sdk.initialize = AsyncMock(wraps=sdk.initialize)
    handle = AsyncClientHandle(OpenVikingConnection(client=sdk))
    try:
        assert await handle.get() is sdk
        await sdk.close()
        assert pool.is_closed
        with pytest.raises(RuntimeError, match="Initialize the borrowed"):
            await handle.get()
        sdk.initialize.assert_not_awaited()
    finally:
        await sdk.close()


async def test_borrowed_sdk_with_closed_shared_pool_is_rejected():
    async with httpx.AsyncClient() as pool:
        sdk = AsyncHTTPClient(url="http://ov.test", http_client=pool)
        await sdk.initialize()
        handle = AsyncClientHandle(OpenVikingConnection(client=sdk))
        assert await handle.get() is sdk
    try:
        with pytest.raises(RuntimeError, match="application-owned HTTP client is closed"):
            await handle.get()
        assert not handle.ready
    finally:
        await sdk.close()


async def test_owned_sdk_initializes_and_closes_its_own_pool():
    handle = AsyncClientHandle(OpenVikingConnection(url="http://ov.test", api_key="test-key"))
    sdk = await handle.get()
    pool = sdk._http
    try:
        assert await handle.get() is sdk
        assert handle.owned and not pool.is_closed
    finally:
        await handle.close()
    assert pool.is_closed and sdk._http is None
