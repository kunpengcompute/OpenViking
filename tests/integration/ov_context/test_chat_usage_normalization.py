"""Provider snapshots have one value through Chat SSE and persisted history."""

import httpx
import pytest

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI, TimedChatOpenAI
from examples.deepagents_ov_context.chat.server import SessionStore

from .test_ark_stream_usage import stream_body
from .test_chat_api import parse_events


@pytest.mark.parametrize("model_type", [TimedChatOpenAI, ArkChatOpenAI])
@pytest.mark.parametrize("mode", ["usage", "both", "updated"])
async def test_chat_usage_matches_final_snapshot_in_sse_message_and_storage(chat, model_type, mode):
    client, runtime = chat
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_body(mode, 1000),
            )
        )
    ) as model_http:
        runtime.model = model_type(
            api_key="offline",
            http_async_client=model_http,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        session = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
        sid = session.json()["session_id"]
        response = await client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    events = parse_events(response.text)
    assert events[-1][0] == "done"
    stats = events[-1][1]["statistics"]
    assert stats["input_tokens"] == 1000
    assert stats["output_tokens"] == 4
    assert stats["total_tokens"] == 1004
    assert stats["model_calls"] == stats["usage_calls"] == 1
    assert "".join(data["text"] for kind, data in events if kind == "token") == "OK"
    trace = (await client.get(f"/api/sessions/{sid}/runs/{stats['run_id']}")).json()
    assert trace["statistics"] == stats
    usage = [
        e["data"]
        for e in trace["events"]
        if e["event"] == "stage" and e["data"].get("name") == "model_usage"
    ]
    assert len(usage) == 1
    assert usage[0]["input_tokens"] == 1000
    assert usage[0]["input_token_details"]["cache_read"] == 32
    assert usage[0]["output_token_details"]["reasoning"] == 2
    reopened = SessionStore(runtime.settings.data_dir / "sessions.sqlite3")
    try:
        assert reopened.get(sid).last_run.model_dump() == stats
    finally:
        reopened.close()
