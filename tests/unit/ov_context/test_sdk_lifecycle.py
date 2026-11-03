import httpx

from openviking.integrations.langchain.ov_context import RunIdentity


async def test_owned_sdk_uses_trusted_identity_without_closing_the_borrowed_pool():
    import asyncio

    from openviking.integrations.langchain.client import OpenVikingConnection
    from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle

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
