"""Verify streaming provider usage reaches the durable Run summary once."""

import json
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage

from examples.deepagents_ov_context.chat.history import RunStatistics
from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from examples.deepagents_ov_context.chat.session import SessionInfo, SessionStore
from examples.deepagents_ov_context.chat.stream import ActiveRun, InputTrace


async def test_real_stream_callback_counts_final_usage_once(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream_options"]["include_usage"] is True
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "thinking"}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 23, "completion_tokens": 9, "total_tokens": 32},
            },
        ]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body + "data: [DONE]\n\n"
        )

    store = SessionStore(tmp_path / "sessions.db")
    session = SessionInfo(
        session_id="s",
        account_id="a",
        user_id="u",
        agent_id="p",
        created_at=0,
        last_run=RunStatistics(run_id="r"),
    )
    trace = InputTrace(ActiveRun(), session, store)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            model = ArkChatOpenAI(
                model="deepseek-v4-flash",
                api_key="test",
                base_url="http://model.invalid/v1",
                http_async_client=http,
                stream_usage=True,
            )
            for _ in range(2):
                chunks = [
                    item
                    async for item in model.astream(
                        [HumanMessage(content="question")], config={"callbacks": [trace]}
                    )
                ]
                assert "".join(item.text for item in chunks) == "answer"
        statistics = store.get("s").last_run
        assert statistics.model_calls == statistics.usage_calls == 2
        assert statistics.input_tokens == 46 and statistics.output_tokens == 18
        assert statistics.total_tokens == 64
    finally:
        store.close()


async def test_missing_usage_is_distinguished_from_reported_zero(tmp_path: Path) -> None:
    import uuid

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    store = SessionStore(tmp_path / "sessions.db")
    session = SessionInfo(
        session_id="s",
        account_id="a",
        user_id="u",
        agent_id="p",
        created_at=0,
        last_run=RunStatistics(run_id="r"),
    )
    trace = InputTrace(ActiveRun(), session, store)
    try:
        for usage in (None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}):
            call = uuid.uuid4()
            await trace.on_chat_model_start({}, [[HumanMessage(content="q")]], run_id=call)
            message = AIMessage(content="answer", usage_metadata=usage)
            await trace.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=call
            )
            statistics = store.get("s").last_run
            assert statistics.total_tokens == (None if usage is None else 0)
        assert statistics.model_calls == 2 and statistics.usage_calls == 1
    finally:
        store.close()
