# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Find retrieves multiple paths in one request under the invoking identity."""

import json

import httpx
import pytest

from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import OpenVikingError

IDENTITY = OVIdentity("tenant", "alice", "doc")
ROOTS = ["viking://user/memories", "viking://user/peers/doc/memories"]


async def invoke(handler, target):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        headers = dict(http.headers)
        sdk = AsyncHTTPClient(
            url="http://fixture",
            api_key="test-key",
            auth_mode="trusted",
            http_client=http,
            profile_enabled=False,
        )
        await sdk.initialize()
        try:
            return await sdk.find("q", target_uri=target, limit=20, identity=IDENTITY)
        finally:
            await sdk.close()
            assert not http.is_closed and dict(http.headers) == headers


@pytest.mark.parametrize("target", [ROOTS, ROOTS[0]])
async def test_find_preserves_paths_limit_and_invocation_identity(target):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/api/v1/search/find"
        assert request.headers["X-OpenViking-Account"] == "tenant"
        assert request.headers["X-OpenViking-User"] == "alice"
        assert request.headers["X-OpenViking-Actor-Peer"] == "doc"
        payload = json.loads(request.content)
        assert payload["target_uri"] == target and payload["limit"] == 20
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "memories": [],
                    "resources": [],
                    "skills": [],
                },
            },
        )

    result = await invoke(handle, target)
    assert result.memories == [] and len(requests) == 1


@pytest.mark.parametrize("code,status", [("PERMISSION_DENIED", 403), ("UNAVAILABLE", 503)])
async def test_failed_multi_path_find_propagates_the_error(code, status):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/api/v1/search/find"
        return httpx.Response(
            status,
            json={
                "status": "error",
                "error": {"code": code, "message": "blocked"},
            },
        )

    with pytest.raises(OpenVikingError):
        await invoke(handle, ROOTS)
    assert len(requests) == 1
