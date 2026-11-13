"""SDK wrappers borrow application transport without changing shared defaults."""

import asyncio

import httpx
import pytest

from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity


@pytest.mark.asyncio
async def test_run_sdks_share_pool_and_apply_url_credentials_identity_timeout_per_request():
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "ok", "result": []})

    async with httpx.AsyncClient(
        base_url="http://unrelated.invalid/",
        headers={"X-Shared": "yes"},
        params={"shared": "yes"},
        transport=httpx.MockTransport(handle),
    ) as pool:
        defaults = dict(pool.headers), pool.base_url, pool.params, pool.timeout

        async def run(number):
            sdk = AsyncHTTPClient(
                auth_mode="trusted",
                url=f"http://ov{number}.test/prefix",
                api_key="single-credential",
                account="default-account",
                user="default-user",
                actor_peer_id="default-actor",
                profile_enabled=True,
                extra_headers={"X-Run": str(number)},
                timeout=number + 2,
                http_client=pool,
            )
            await sdk.initialize()
            await sdk.initialize()  # Idempotent; never builds another transport.
            assert sdk._http is pool
            await sdk.list_tasks(
                task_type="session_commit",
                status="running",
                resource_id=f"s{number}",
                identity=OVIdentity("acct", f"user{number}", "mail"),
            )
            await sdk.close()
            assert not pool.is_closed and sdk._http is None
            await sdk.initialize()
            await sdk.list_sessions()  # Per-SDK configured default identity.
            await sdk.close()

        await asyncio.gather(*(run(n) for n in (1, 2)))
        assert defaults == (dict(pool.headers), pool.base_url, pool.params, pool.timeout)
        for request in requests:
            number = int(request.headers["X-Run"])
            assert request.url.host == f"ov{number}.test"
            assert request.url.path.startswith("/prefix/api/v1/")
            assert request.headers["X-API-Key"] == "single-credential"
            assert request.headers["X-Shared"] == "yes"
            assert request.url.params["profile"] == "1"
            assert request.url.params["shared"] == "yes"
            assert request.extensions["timeout"]["read"] == number + 2
            if request.url.path.endswith("/tasks"):
                assert request.headers["X-OpenViking-User"] == f"user{number}"
                assert request.headers["X-OpenViking-Actor-Peer"] == "mail"
            else:
                assert request.headers["X-OpenViking-User"] == "default-user"
                assert request.headers["X-OpenViking-Actor-Peer"] == "default-actor"


@pytest.mark.asyncio
async def test_closed_borrowed_pool_fails_without_silently_creating_a_new_pool():
    pool = httpx.AsyncClient()
    await pool.aclose()
    sdk = AsyncHTTPClient(
        auth_mode="trusted", url="http://ov.test", http_client=pool, profile_enabled=False
    )
    with pytest.raises(RuntimeError, match="application-owned HTTP client is closed"):
        await sdk.initialize()
    assert sdk._http is None


@pytest.mark.asyncio
async def test_owned_pool_still_closes_and_can_be_reinitialized():
    sdk = AsyncHTTPClient(auth_mode="trusted", url="http://ov.test", profile_enabled=False)
    await sdk.initialize()
    first = sdk._http
    await sdk.initialize()
    assert sdk._http is first
    await sdk.close()
    assert first.is_closed
    await sdk.initialize()
    assert sdk._http is not first and not sdk._http.is_closed
    await sdk.close()


def test_sync_task_list_forwards_request_identity(monkeypatch):
    from unittest.mock import AsyncMock

    from openviking_cli.client.sync_http import SyncHTTPClient

    monkeypatch.setattr("openviking_cli.client.sync_http.run_async", asyncio.run)
    sdk = SyncHTTPClient(url="http://ov.test", profile_enabled=False)
    operation = AsyncMock(return_value=[{"task_id": "t1", "status": "running"}])
    sdk._async_client.list_tasks = operation
    identity = OVIdentity("acct", "user", "mail")
    assert sdk.list_tasks(
        task_type="session_commit", status="running", resource_id="s", limit=2, identity=identity
    ) == [{"task_id": "t1", "status": "running"}]
    operation.assert_awaited_once_with(
        task_type="session_commit", status="running", resource_id="s", limit=2, identity=identity
    )
