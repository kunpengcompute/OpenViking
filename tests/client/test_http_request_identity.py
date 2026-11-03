"""Native SDK request identity through real HTTPX request construction."""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
import pytest_asyncio

from openviking_cli.client import OVIdentity
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.sync_http import SyncHTTPClient
from openviking_cli.exceptions import PermissionDeniedError

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncHTTPClient, list[httpx.Request]]]:
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)
        if request.url.path.endswith(("/export", "/backup")):
            return httpx.Response(200, content=b"ovpack")
        result: Any = {
            "session_id": "s",
            "archived": True,
            "task_id": "task",
            "temp_file_id": "upload",
            "uri": "viking://resources/imported",
        }
        if request.url.path.startswith("/api/v1/content/") and request.method == "GET":
            result = "text"
        return httpx.Response(200, json={"status": "ok", "result": result})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real(**{"transport": httpx.MockTransport(handle), **kwargs}),
    )
    client = AsyncHTTPClient(
        auth_mode="trusted",
        url="http://ov.test",
        api_key="single-credential",
        account="default-account",
        user="default-user",
        actor_peer_id="default-peer",
        extra_headers={"x-openviking-user": "extra-user", "X-Trace": "kept"},
        profile_enabled=False,
    )
    await client.initialize()
    try:
        yield client, requests
    finally:
        await client.close()


# All public business calls, including nested uploads, share the same contract.
CASES = [
    ("create_session", {"session_id": "s"}),
    ("list_sessions", {}),
    ("get_session", {"session_id": "s", "auto_create": True}),
    ("get_session_context", {"session_id": "s"}),
    ("get_session_archive", {"session_id": "s", "archive_id": "archive"}),
    (
        "add_message",
        {"session_id": "s", "role": "assistant", "content": "answer", "peer_id": "message-owner"},
    ),
    (
        "batch_add_messages",
        {
            "session_id": "s",
            "messages": [{"role": "assistant", "content": "answer", "peer_id": "message-owner"}],
        },
    ),
    ("commit_session", {"session_id": "s", "keep_recent_count": 2}),
    ("get_task", {"task_id": "task"}),
    ("delete_session", {"session_id": "s"}),
    ("session_exists", {"session_id": "s"}),
    ("read_tool_result", {"session_id": "s", "tool_result_id": "result"}),
    ("search_tool_result", {"session_id": "s", "tool_result_id": "result", "query": "needle"}),
    ("list_tool_results", {"session_id": "s"}),
    ("find", {"query": "query"}),
    ("search", {"query": "query", "session_id": "s"}),
    ("read", {"uri": "viking://resources/file.md"}),
    ("abstract", {"uri": "viking://resources/file.md"}),
    ("overview", {"uri": "viking://resources/file.md"}),
    ("write", {"uri": "viking://resources/file.md", "content": "body"}),
    ("ls", {"uri": "viking://resources"}),
    ("tree", {"uri": "viking://resources"}),
    ("stat", {"uri": "viking://resources/file.md"}),
    ("mkdir", {"uri": "viking://resources/folder"}),
    ("rm", {"uri": "viking://resources/file.md"}),
    ("mv", {"from_uri": "viking://resources/old.md", "to_uri": "viking://resources/new.md"}),
    ("grep", {"pattern": "needle", "uri": "viking://resources"}),
    ("glob", {"pattern": "*.md"}),
    ("relations", {"uri": "viking://resources/file.md"}),
    ("link", {"from_uri": "viking://resources/a.md", "to_uris": ["viking://resources/b.md"]}),
    ("unlink", {"from_uri": "viking://resources/a.md", "to_uri": "viking://resources/b.md"}),
    ("reindex", {"uri": "viking://resources/file.md"}),
    ("check_consistency", {"uri": "viking://resources"}),
    ("wait_processed", {}),
    ("add_resource", {"path": "https://example.test/document"}),
    ("add_skill", {"data": {"name": "skill"}}),
    ("export_ovpack", {"uri": "viking://resources", "to": "TMP"}),
    ("backup_ovpack", {"to": "TMP"}),
    ("import_ovpack", {"file_path": "FILE", "parent": "viking://resources"}),
    ("restore_ovpack", {"file_path": "FILE"}),
]


@pytest.mark.parametrize("method,arguments", CASES, ids=[case[0] for case in CASES])
async def test_business_api_uses_request_identity_without_changing_defaults(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]],
    tmp_path: Path,
    method: str,
    arguments: dict[str, Any],
) -> None:
    client, requests = sdk
    assert client._http is not None
    path = tmp_path / "input.ovpack"
    path.write_bytes(b"pack")
    kwargs = {
        key: str(tmp_path / "output.ovpack")
        if value == "TMP"
        else str(path)
        if value == "FILE"
        else value
        for key, value in arguments.items()
    }
    defaults = dict(client._http.headers)
    identity = OVIdentity("account-a", "alice", "agent-a")
    await getattr(client, method)(**kwargs, identity=identity)
    assert requests
    for request in requests:
        assert all(request.headers[key] == value for key, value in identity.headers().items())
        assert request.headers["X-API-Key"] == "single-credential"
        assert request.headers["X-Trace"] == "kept"
        if "application/json" in request.headers.get("content-type", ""):
            body = json.loads(request.content)
            assert (
                "identity" not in body and "account_id" not in body and "actor_peer_id" not in body
            )
    assert dict(client._http.headers) == defaults
    await client.get_session("s")
    assert requests[-1].headers["X-OpenViking-Account"] == "default-account"
    assert requests[-1].headers["X-OpenViking-Actor-Peer"] == "default-peer"


async def test_parallel_identities_keep_headers_and_session_parameters_local(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]],
) -> None:
    client, requests = sdk
    assert client._http is not None
    identities = [
        OVIdentity("account-a", "alice", "mail"),
        OVIdentity("account-b", "bob", "document"),
    ]
    defaults = dict(client._http.headers)

    async def invoke(identity: OVIdentity) -> None:
        await client.search(query=identity.user_id, session_id="same-id", identity=identity)
        await client.add_message(
            "same-id", "assistant", "answer", peer_id="different-attribution", identity=identity
        )

    await asyncio.gather(*(invoke(identity) for identity in identities))
    for request in requests:
        identity = next(
            i for i in identities if i.account_id == request.headers["X-OpenViking-Account"]
        )
        assert all(request.headers[key] == value for key, value in identity.headers().items())
        body = json.loads(request.content)
        if request.url.path.endswith("/search"):
            assert body["query"] == identity.user_id
            assert body["session_id"] == "same-id"
        else:
            assert body["peer_id"] == "different-attribution"
    assert dict(client._http.headers) == defaults


@pytest.mark.parametrize("method", ["add_resource", "add_skill", "import_ovpack", "restore_ovpack"])
async def test_upload_and_followup_request_keep_the_same_identity(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]], tmp_path: Path, method: str
) -> None:
    client, requests = sdk
    assert client._http is not None
    path = tmp_path / "local.md"
    path.write_text("resource")
    identity = OVIdentity("account-a", "alice", "mail")
    kwargs: dict[str, Any]
    if method == "add_resource":
        kwargs = {"path": str(path)}
    elif method == "add_skill":
        kwargs = {"data": str(path)}
    else:
        kwargs = {"file_path": str(path)}
        if method == "import_ovpack":
            kwargs["parent"] = "viking://resources"
    await getattr(client, method)(**kwargs, identity=identity)
    assert len(requests) == 2
    assert requests[0].url.path.endswith("/temp_upload")
    assert all(request.headers["X-OpenViking-User"] == "alice" for request in requests)


async def test_explicit_identity_does_not_apply_legacy_agent_to_body(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]],
) -> None:
    client, requests = sdk
    assert client._http is not None
    client._legacy_agent_id = "old-agent"
    identity = OVIdentity("a", "u", "new-agent")
    await client.find("query", identity=identity)
    await client.search("query", identity=identity)
    await client.add_message("s", "assistant", "answer", peer_id="attribution", identity=identity)
    await client.batch_add_messages(
        "s",
        [{"role": "assistant", "content": "answer", "peer_id": "attribution"}],
        identity=identity,
    )
    for request in requests:
        body = json.loads(request.content)
        assert "agent_id" not in body
        assert all("agent_id" not in message for message in body.get("messages", []))
        assert request.headers["X-OpenViking-Actor-Peer"] == "new-agent"


@pytest.mark.parametrize("field", ["account_id", "user_id", "actor_peer_id"])
@pytest.mark.parametrize("bad", ["", " ", "bad\nheader", "bad/path"])
async def test_invalid_identity_rejected_before_network(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]], field: str, bad: str
) -> None:
    client, requests = sdk
    assert client._http is not None
    values = {"account_id": "a", "user_id": "u", "actor_peer_id": "p", field: bad}
    with pytest.raises(ValueError):
        await client.get_session("s", identity=OVIdentity(**values))
    assert not requests


async def test_error_and_cancellation_do_not_change_default_or_next_identity(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]],
) -> None:
    client, requests = sdk
    assert client._http is not None
    defaults = dict(client._http.headers)
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["X-OpenViking-User"] == "cancelled":
            entered.set()
            await release.wait()
        if request.headers["X-OpenViking-User"] == "denied":
            return httpx.Response(
                403,
                json={
                    "status": "error",
                    "error": {"code": "PERMISSION_DENIED", "message": "denied"},
                },
            )
        return httpx.Response(200, json={"result": {"session_id": "s"}})

    original = client._http
    client._http = httpx.AsyncClient(
        base_url="http://ov.test", headers=defaults, transport=httpx.MockTransport(handle)
    )
    await original.aclose()
    task = asyncio.create_task(client.get_session("s", identity=OVIdentity("a", "cancelled", "p")))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(PermissionDeniedError):
        await client.get_session("s", identity=OVIdentity("a", "denied", "p"))
    await client.get_session("s", identity=OVIdentity("b", "success", "q"))
    assert requests[-1].headers["X-OpenViking-User"] == "success"
    assert dict(client._http.headers) == defaults


async def test_sdk_identity_does_not_import_agent_frameworks() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from openviking_cli.client import OVIdentity; import sys; assert not any(name.startswith(('langchain', 'langgraph')) for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("method,arguments", CASES, ids=[case[0] for case in CASES])
async def test_sync_sdk_forwards_identity(
    sdk: tuple[AsyncHTTPClient, list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    arguments: dict[str, Any],
) -> None:
    identity = OVIdentity("a", "u", "p")
    sync = SyncHTTPClient(url="http://ov.test", api_key="single", profile_enabled=False)
    recorded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def operation(*args: Any, **kwargs: Any) -> object:
        recorded.append((args, kwargs))
        return identity

    monkeypatch.setattr(sync._async_client, method, operation)
    monkeypatch.setattr("openviking_cli.client.sync_http.run_async", lambda result: result)
    arguments = dict(arguments)
    aliases = {
        "link": ("to_uris", "uris"),
        "unlink": ("to_uri", "uri"),
        "import_ovpack": ("parent", "target"),
    }
    if method in aliases:
        original, renamed = aliases[method]
        arguments[renamed] = arguments.pop(original)
    result = getattr(sync, method)(**arguments, identity=identity)
    if method in {"delete_session", "mkdir", "rm", "mv", "link", "unlink"}:
        assert result is None
    else:
        assert result is identity
    assert recorded[0][1]["identity"] is identity
    assert (
        inspect.signature(getattr(sync, method)).parameters["identity"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
