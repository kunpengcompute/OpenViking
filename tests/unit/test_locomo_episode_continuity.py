from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmark.locomo.openviking import import_to_ov as ingest


@pytest.mark.asyncio
async def test_reused_session_preserves_timestamp_offset(monkeypatch):
    client = SimpleNamespace(
        initialize=AsyncMock(),
        create_session=AsyncMock(),
        close=AsyncMock(),
        add_message=AsyncMock(),
        commit_session=AsyncMock(return_value={"status": "accepted"}),
    )
    monkeypatch.setattr(ingest.ov, "AsyncHTTPClient", lambda **kwargs: client)
    result = await ingest.viking_ingest(
        [{"role": "user", "text": "hello"}],
        "http://unused",
        "1:00 pm on 8 May, 2023",
        session_id="conversation",
        message_offset=8,
    )
    client.create_session.assert_not_awaited()
    call = client.add_message.call_args.kwargs
    assert call["session_id"] == result["session_id"] == "conversation"
    assert datetime.fromisoformat(call["created_at"]).second == 8


@pytest.mark.asyncio
async def test_import_chunks_share_one_session_and_global_order(monkeypatch, tmp_path):
    calls = []

    async def fake(messages, url, date, **kwargs):
        calls.append(kwargs)
        return {"session_id": "conversation", "token_usage": {}, "task_id": None}

    monkeypatch.setattr(ingest, "viking_ingest", fake)
    monkeypatch.setattr(ingest, "save_ingest_record", lambda *args: None)
    result = await ingest.process_single_session(
        [{"role": "user", "text": str(i)} for i in range(17)],
        "sample",
        "session_1",
        {},
        "now",
        {},
        SimpleNamespace(
            openviking_url="http://unused",
            use_image_url=False,
            success_csv=str(tmp_path / "success.csv"),
            error_log=str(tmp_path / "error.log"),
        ),
    )
    assert result["status"] == "success"
    assert [c["session_id"] for c in calls] == [None, "conversation", "conversation"]
    assert [c["message_offset"] for c in calls] == [0, 8, 16]
