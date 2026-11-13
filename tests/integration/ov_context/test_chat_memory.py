"""Manual extraction reuses scoped OV commit/tasks; failures never imply completion."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from examples.deepagents_ov_context.chat.server import ActiveRun

from .test_chat_api import new_session


def task(session_id: str, status: str = "completed") -> dict[str, Any]:
    return {
        "task_type": "session_commit",
        "resource_id": session_id,
        "status": status,
        "stage": "completed" if status == "completed" else "memory_extraction",
        "result": {
            "memories_extracted": {"memory_write": 3},
            "archive_uri": "viking://test/archive_001",
        },
        "error": "private task detail",
    }


async def test_memory_commit_receipt_task_persistence_identity(
    chat,
) -> None:
    client, rt = chat
    sid = await new_session(client, user_id="alice", agent_id="mail")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        value = (
            {"archived": True, "task_id": "task-1"}
            if request.method == "POST"
            else task(sid)
            if "/tasks/" in request.url.path
            else {"message_count": 2}
        )
        return httpx.Response(200, json={"result": value})

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await client.post(f"/api/sessions/{sid}/memory")
    assert response.status_code == 200 and response.json()["status"] == "pending"
    assert rt.store.get(sid).memory_task_id == "task-1"
    assert not rt.store.get(sid).memory_commit_unknown
    commit = next(request for request in requests if request.method == "POST")
    assert json.loads(commit.content)["keep_recent_count"] == 0
    assert requests[0].headers["x-openviking-user"] == "alice"
    assert requests[0].headers["x-openviking-actor-peer"] == "mail"
    response = await client.get(f"/api/sessions/{sid}/memory")
    assert response.json()["status"] == "completed"
    assert response.json()["memories_extracted"] == {"memory_write": 3}
    assert "private task detail" not in response.text
    assert requests[-1].url.params["account_id"] == "chat-account"
    assert requests[-1].url.params["user_id"] == "alice"
    assert not rt.restoring and not rt.store.get(sid).needs_restore


@pytest.mark.parametrize("status", ["running", "pending", "missing"])
async def test_pending_or_missing_memory_task_cannot_be_submitted_again(chat, status: str) -> None:
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    session.memory_task_id = "task-1"
    rt.store.save(session)
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            404 if status == "missing" else 200, json={"result": task(sid, status)}
        )

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 409
    assert methods == ["GET"]


@pytest.mark.parametrize("failure", ["timeout", "missing-task-id"])
async def test_unknown_commit_result_is_persisted_and_never_automatically_retried(
    chat, failure: str
) -> None:
    client, rt = chat
    sid = await new_session(client)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "GET":
            return httpx.Response(200, json={"result": {"message_count": 2}})
        calls += 1
        if failure == "timeout":
            return httpx.Response(
                503,
                json={
                    "status": "error",
                    "error": {"code": "UNAVAILABLE", "message": "private service error"},
                },
            )
        return httpx.Response(200, json={"result": {"archived": True}})

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await client.post(f"/api/sessions/{sid}/memory")
    assert response.status_code in (502, 503) and "private service error" not in response.text
    assert rt.store.get(sid).memory_commit_unknown
    assert rt.store.get(sid).needs_restore
    assert not rt.restoring
    assert (await client.get(f"/api/sessions/{sid}/memory")).json()["status"] == "unknown"
    assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 409
    assert calls == 1


async def test_extraction_reservation_blocks_chat_and_other_extraction_until_receipt(chat) -> None:
    client, rt = chat
    sid = await new_session(client)
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"result": {"message_count": 2}})
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"result": {"archived": True, "task_id": "task-1"}})

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    attempt = asyncio.create_task(client.post(f"/api/sessions/{sid}/memory"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 409
        assert (
            await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
        ).status_code == 409
    finally:
        release.set()
        assert (await attempt).status_code == 200
    rt.active[sid] = ActiveRun()
    assert (await client.post(f"/api/sessions/{sid}/memory")).status_code == 409
    rt.active.clear()


async def test_memory_task_for_another_session_is_rejected(chat) -> None:
    client, rt = chat
    sid = await new_session(client)
    session = rt.store.get(sid)
    session.memory_task_id = "wrong-task"
    rt.store.save(session)
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": task("other-session")})
        )
    )
    assert (await client.get(f"/api/sessions/{sid}/memory")).status_code == 502


@pytest.mark.parametrize("missing", [False, True])
async def test_empty_or_uncreated_session_does_not_submit_or_mark_unknown(
    chat, missing: bool
) -> None:
    client, rt = chat
    sid = await new_session(client)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return (
            httpx.Response(
                404, json={"status": "error", "error": {"code": "NOT_FOUND", "message": "missing"}}
            )
            if missing
            else httpx.Response(200, json={"result": {"message_count": 0}})
        )

    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await client.post(f"/api/sessions/{sid}/memory")
    assert response.status_code == (409 if missing else 200)
    assert all(request.method == "GET" for request in requests)
    assert not rt.store.get(sid).memory_commit_unknown
    assert not rt.store.get(sid).needs_restore
