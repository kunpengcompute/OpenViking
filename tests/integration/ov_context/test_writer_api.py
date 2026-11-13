import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest_asyncio
from fastapi import FastAPI

from openviking.client.local import LocalClient
from openviking.integrations._message_content import LANGCHAIN_CONTENT_MIME, decode_tool_content
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking.message import ToolPart
from openviking.server.auth import get_request_context
from openviking.server.routers import sessions as sessions_router
from openviking.server.routers import tasks as tasks_router
from openviking.service.session_service import SessionService
from openviking.service.task_tracker import TaskTracker
from openviking.session.tool_views import add_messages_with_tool_views
from openviking_cli.client.http import AsyncHTTPClient
from tests.ov_context_support.message_examples import exchange
from tests.ov_context_support.session_view import _native_messages
from tests.ov_context_support.writer import make_session


@pytest_asyncio.fixture
async def native_api(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    sessions = SessionService(viking_fs=session._viking_fs)
    sessions.get = AsyncMock(return_value=session)
    sessions.open = AsyncMock(return_value=session)
    sessions.maybe_schedule_auto_commit = AsyncMock(return_value=False)
    monkeypatch.setattr(sessions_router, "get_service", lambda: SimpleNamespace(sessions=sessions))
    app = FastAPI()
    app.include_router(sessions_router.router)
    store = AsyncMock()
    store.list.return_value = []
    tracker = TaskTracker(store)
    monkeypatch.setattr(tasks_router, "get_task_tracker", lambda: tracker)
    app.include_router(tasks_router.router)
    app.dependency_overrides[get_request_context] = lambda: session.ctx
    client = AsyncHTTPClient(url="http://ov-test")
    client._http = httpx.AsyncClient(
        base_url="http://ov-test", transport=httpx.ASGITransport(app=app)
    )
    try:
        yield session, client
    finally:
        await client.close()


async def test_real_http_route_sdk_and_writer_return_short_view(native_api):
    session, client = native_api
    raw = exchange(["small", "订单状态：已发货。\n" * 800])
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    sessions_router.get_service().sessions.maybe_schedule_auto_commit.assert_awaited_once()
    assert result.write_index == 4 and result.written_count == 4
    assert result.model_messages[2] is raw[2]
    assert result.model_messages[3].content != raw[3].content
    assert raw[3].content == "订单状态：已发货。\n" * 800
    part = session.messages[-1].parts[0]
    result_id = part.tool_output_ref.rsplit("/", 1)[-1]
    stored = await client.read_tool_result(session.session_id, result_id, limit=-1)
    assert stored["content"] == raw[3].content
    found = await client.search_tool_result(session.session_id, result_id, query="已发货")
    assert "已发货" in json.dumps(found, ensure_ascii=False)
    listed = await client.list_tool_results(session.session_id)
    assert len(listed["tool_results"]) == 1


async def test_http_empty_view_and_legacy_count_fields(native_api):
    session, client = native_api
    result = await client.batch_add_messages(session.session_id, [{"role": "user", "content": "a"}])
    sessions_router.get_service().sessions.maybe_schedule_auto_commit.assert_awaited_once()
    assert result == {
        "session_id": session.session_id,
        "message_count": 1,
        "added": 1,
        "pending_tokens": 1,
        "tool_views": [],
    }


async def test_local_client_batch_receipt_matches_http_contract(tmp_path):
    session = make_session(tmp_path)
    client = LocalClient.__new__(LocalClient)
    client._ctx = session.ctx
    client._legacy_agent_id = None
    client._service = SimpleNamespace(
        sessions=SimpleNamespace(
            get=AsyncMock(return_value=session), open=AsyncMock(return_value=session)
        )
    )
    raw = exchange(["local result\n" * 800])
    result = await OVMessageWriter(client).write(session.session_id, raw, 0)
    assert result.model_messages[-1].content != raw[-1].content
    assert "viking://" in result.model_messages[-1].content


async def test_structured_externalized_output_decodes_as_model_text(tmp_path):
    session = make_session(tmp_path)
    original = [{"type": "text", "text": "订单详情\n" * 1000}]
    part = ToolPart(
        tool_id="c",
        tool_name="lookup",
        tool_status="completed",
        tool_output=json.dumps(original, ensure_ascii=False),
        tool_output_mime_type=LANGCHAIN_CONTENT_MIME,
    )
    receipt = add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
    assert receipt["tool_views"]
    restored = _native_messages(session.messages[0].to_dict())[0]
    assert restored.content == part.tool_output
    result_id = part.tool_output_ref.rsplit("/", 1)[-1]
    stored = await session.read_tool_result(result_id, limit=-1)
    assert json.loads(stored["content"]) == original
    assert (
        decode_tool_content(
            {
                "tool_output": json.dumps(original),
                "tool_output_mime_type": LANGCHAIN_CONTENT_MIME,
            }
        )
        == original
    )
