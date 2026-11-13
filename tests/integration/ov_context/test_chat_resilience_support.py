"""Real Chat hooks with deterministic external service and model responses."""

from dataclasses import replace

import httpx
import pytest_asyncio

from examples.deepagents_ov_context.chat.config import ChatOVPolicy

from .chat_model_support import NoToolModel


@pytest_asyncio.fixture
async def resilient_chat(chat, env):
    client, runtime = chat
    server = env[0]
    runtime.settings = replace(runtime.settings, ov_policy=ChatOVPolicy())
    runtime.model = NoToolModel()
    await runtime.ov_http.aclose()

    async def external(request):
        if getattr(server, "fault", None):
            failed = await server.fault(request)
            if failed is not None:
                return failed
        response = await server.handle(request)
        if response.status_code == 404:
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "error": {"code": "NOT_FOUND", "message": "session not found"},
                },
            )
        if response.status_code != 200:
            return response
        body = response.json()
        path = request.url.path
        if path.endswith("/context"):
            body["result"]["latest_finished_archive_id"] = None
        elif path.endswith("/tasks/t1"):
            body["result"].update(
                {
                    "resource_id": next(iter(server.sessions)),
                    "task_type": "session_commit",
                    "result": {"memories_extracted": {"preferences": 1}},
                }
            )
        elif path.endswith("/commit"):
            body["result"]["archived"] = True
        return httpx.Response(200, json=body)

    runtime.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(external))
    yield client, runtime, server
