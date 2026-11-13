"""Real graph regressions for malformed model tool arguments (no network)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import Field

from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from examples.deepagents_ov_context.chat.tool_call_recovery import (
    CORRECTION_PROMPT,
    InvalidToolCallError,
    ToolCallRecoveryMiddleware,
)

from .test_graph import CallContext, build_middleware

# Same illegal escape as the live browser eval failures; not a tool range error.
BAD_ARGS = r'{"argv":["eval","/\.js/.test(resource.name)"]}'


def rejected(*, mixed: bool = False) -> AIMessage:
    # Exercise the actual OpenAI parser, rather than manufacturing its error outcome.
    calls = [
        {"id": "bad", "type": "function", "function": {"name": "browser", "arguments": BAD_ARGS}}
    ]
    if mixed:
        calls.append(
            {
                "id": "not-executed",
                "type": "function",
                "function": {"name": "browser", "arguments": '{"argv":["open","wrong"]}'},
            }
        )
    model = ChatOpenAI(api_key="offline-only")
    response = model._create_chat_result(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Inspect scripts",
                        "tool_calls": calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    message = response.generations[0].message
    assert isinstance(message, AIMessage) and message.invalid_tool_calls
    message.id = "rejected"
    message.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    return message


class SequenceModel(BaseChatModel):
    responses: list[AIMessage]
    inputs: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "invalid-json-regression"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> SequenceModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.inputs.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[len(self.inputs) - 1])]
        )


def corrected(name: str = "browser", args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        id="corrected",
        content="",
        tool_calls=[
            {
                "id": "valid",
                "name": name,
                "args": args or {"argv": ["eval", "resource.name.endsWith('.js')"]},
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def final() -> AIMessage:
    return AIMessage(
        id="final",
        content="Task complete",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def request() -> ModelRequest[Any]:
    messages = [HumanMessage(id="user", content="Inspect scripts")]
    return ModelRequest(
        model=SequenceModel(responses=[]),
        messages=messages,
        tools=[],
        state={"messages": messages},
        runtime=Runtime(),
    )


@pytest.mark.parametrize("factory", [create_agent, create_deep_agent])
@pytest.mark.parametrize("mixed", [False, True])
async def test_rejected_response_not_executed_or_checkpointed(factory: Any, mixed: bool) -> None:
    executed: list[list[str]] = []

    @tool
    def browser(argv: list[str]) -> str:
        """Inspect browser resources."""
        executed.append(argv)
        return "script.js"

    model = SequenceModel(responses=[rejected(mixed=mixed), corrected(), final()])
    graph = factory(
        model=model,
        tools=[browser],
        middleware=[ToolCallRecoveryMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread"}}
    result = await graph.ainvoke({"messages": [HumanMessage(content="Inspect scripts")]}, config)
    assert len(executed) == 1 and executed[0][-1] != "wrong"
    assert len(model.inputs) == 3
    assert model.inputs[1][-1].content.startswith(CORRECTION_PROMPT)
    # Temporary feedback only appears in the correction request, never later history.
    assert not any(CORRECTION_PROMPT in str(m.content) for m in model.inputs[2])
    saved = (await graph.aget_state(config)).values["messages"]
    assert result["messages"][-1].content == "Task complete"
    assert not any(m.id == "rejected" or CORRECTION_PROMPT in str(m.content) for m in saved)
    assert [m.tool_call_id for m in saved if isinstance(m, ToolMessage)] == ["valid"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_bounded_retries_exhaust_without_mutating_request(asynchronous: bool) -> None:
    original = request()
    before = [m.model_dump() for m in original.messages]
    observed: list[ModelRequest[Any]] = []
    recovery = ToolCallRecoveryMiddleware()

    def handler(value: ModelRequest[Any]) -> ModelResponse[Any]:
        observed.append(value)
        return ModelResponse(result=[rejected()])

    async def ahandler(value: ModelRequest[Any]) -> ModelResponse[Any]:
        return handler(value)

    with pytest.raises(InvalidToolCallError, match="after 3 model attempts"):
        if asynchronous:
            await recovery.awrap_model_call(original, ahandler)
        else:
            recovery.wrap_model_call(original, handler)
    assert len(observed) == 3
    assert [len(value.messages) for value in observed] == [1, 2, 2]
    assert [m.model_dump() for m in original.messages] == before
    assert original.state["messages"] == original.messages


@pytest.mark.parametrize("exception", [asyncio.CancelledError(), ConnectionError("offline")])
async def test_other_failures_are_not_retried(exception: BaseException) -> None:
    handler = AsyncMock(side_effect=exception)
    with pytest.raises(type(exception)):
        await ToolCallRecoveryMiddleware().awrap_model_call(request(), handler)
    handler.assert_awaited_once()


async def test_correction_keeps_ov_state_command_and_only_writes_valid_messages(env: Any) -> None:
    server, transport, _, identity, _, _ = env
    middleware = build_middleware(env)

    @tool
    def browser(argv: list[str]) -> str:
        """Inspect browser resources."""
        return "script.js"

    model = SequenceModel(responses=[rejected(), corrected(), final()])
    graph = create_deep_agent(
        model=model,
        tools=[browser],
        middleware=[middleware, ToolCallRecoveryMiddleware()],
        context_schema=CallContext,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": identity.session_id}}
    await graph.ainvoke(
        {"messages": [HumanMessage(id="m1", content="Inspect scripts")]},
        config=config,
        context=CallContext(identity),
    )
    state = (await graph.aget_state(config)).values
    assert state["ov_context"]["run_id"] is None
    assert state["ov_context"]["write_index"] == len(state["messages"])
    assert not middleware._ov_session_engine.active
    persisted = json.dumps(server.messages)
    assert "Inspect scripts" in persisted and "script.js" in persisted
    assert "Task complete" in persisted
    assert "Inspect scripts" == model.inputs[0][-1].content[-len("Inspect scripts") :]
    assert "rejected" not in persisted and CORRECTION_PROMPT.rstrip() not in persisted
    await middleware.aclose()
    await transport.http.aclose()


async def test_streaming_openai_parser_recovers_before_graph_update() -> None:
    payloads: list[dict[str, Any]] = []
    executed: list[list[str]] = []
    usage: list[int] = []

    class Trace(AsyncCallbackHandler):
        async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            message = response.generations[0][0].message
            usage.append(message.usage_metadata["total_tokens"])

    @tool
    def browser(argv: list[str]) -> str:
        """Inspect browser resources."""
        executed.append(argv)
        return "script.js"

    def handle(value: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(value.content))
        index = len(payloads)
        delta: dict[str, Any] = {"role": "assistant", "content": "Task complete"}
        if index < 3:
            delta.update(
                content="",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "bad" if index == 1 else "valid",
                        "type": "function",
                        "function": {
                            "name": "browser",
                            "arguments": BAD_ARGS
                            if index == 1
                            else json.dumps({"argv": ["eval", "resource.name.endsWith('.js')"]}),
                        },
                    }
                ],
            )
        chunks = [
            {
                "id": f"completion-{index}",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"completion-{index}",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if index < 3 else "stop",
                    }
                ],
            },
            {
                "id": f"completion-{index}",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        ]
        text = (
            "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ArkChatOpenAI(
            api_key="offline-only",
            base_url="https://model.invalid/v1",
            http_async_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        graph = create_deep_agent(
            model=model, tools=[browser], middleware=[ToolCallRecoveryMiddleware()]
        )
        updates: list[BaseMessage] = []
        async for mode, event in graph.astream(
            {"messages": [HumanMessage(content="Inspect scripts")]},
            config={"callbacks": [Trace()]},
            stream_mode=["messages", "updates"],
        ):
            if mode == "updates":
                for update in event.values():
                    if isinstance(update, dict):
                        updates.extend(update.get("messages", []))
    assert len(payloads) == 3 and len(executed) == 1 and usage == [12, 12, 12]
    assert CORRECTION_PROMPT in payloads[1]["messages"][-1]["content"]
    assert CORRECTION_PROMPT.rstrip() not in json.dumps(payloads[2])
    assert updates[-1].content == "Task complete"
    assert not any(isinstance(m, AIMessage) and m.invalid_tool_calls for m in updates)


async def test_exhaustion_fails_the_real_graph_instead_of_false_done() -> None:
    model = SequenceModel(responses=[rejected(), rejected(), rejected()])
    graph = create_deep_agent(model=model, middleware=[ToolCallRecoveryMiddleware()])
    with pytest.raises(InvalidToolCallError):
        await graph.ainvoke({"messages": [HumanMessage(content="Inspect scripts")]})
    assert len(model.inputs) == 3


@pytest.mark.parametrize("invalid", [True, False])
async def test_chat_rejects_unhandled_invalid_or_missing_tool_call(chat, monkeypatch, invalid):
    from examples.deepagents_ov_context.chat.server import ChatRuntime

    client, runtime = chat
    response = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    sid = response.json()["session_id"]
    message = (
        rejected()
        if invalid
        else AIMessage(content="", response_metadata={"finish_reason": "tool_calls"})
    )

    class BrokenGraph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"model": {"messages": [message]}}

    monkeypatch.setattr(ChatRuntime, "graph", lambda *a, **k: (BrokenGraph(), None))
    response = await client.post("/api/chat", json={"session_id": sid, "message": "Inspect"})
    from .test_chat_api import parse_events

    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error_type"] == "InvalidToolCallError"
    assert "任务未完成" in events[-1][1]["message"]
    assert not any(kind == "done" for kind, _ in events)
    assert runtime.store.get(sid).needs_restore and not runtime.active


async def test_recovery_exhaustion_returns_error_and_counts_every_attempt(chat):
    client, runtime = chat
    runtime.model = SequenceModel(responses=[rejected(), rejected(), rejected()])
    response = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    sid = response.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "Inspect"})
    from .test_chat_api import parse_events

    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error_type"] == "InvalidToolCallError"
    assert events[-1][1]["statistics"]["model_calls"] == 3
    assert events[-1][1]["statistics"]["total_tokens"] == 36
    assert not any(kind == "done" for kind, _ in events)
    metrics = [d for k, d in events if k == "stage" and d.get("name") == "model.tool_call_recovery"]
    assert [d["status"] for d in metrics] == ["retry", "retry", "failed"]


def test_disabled_retry_fails_immediately() -> None:
    calls = 0

    def handler(value: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[rejected()])

    with pytest.raises(InvalidToolCallError, match="after 1 model attempts"):
        ToolCallRecoveryMiddleware(max_retries=0).wrap_model_call(request(), handler)
    assert calls == 1
    with pytest.raises(ValueError):
        ToolCallRecoveryMiddleware(max_retries=-1)


def test_actual_failure_is_invalid_json_not_tool_limit() -> None:
    with pytest.raises(json.JSONDecodeError, match="Invalid.*escape"):
        json.loads(BAD_ARGS)
    message = rejected()
    assert message.tool_calls == [] and message.invalid_tool_calls[0]["name"] == "browser"
