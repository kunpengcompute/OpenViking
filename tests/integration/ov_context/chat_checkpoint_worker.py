"""Independent Chat processes used by the persistent checkpoint contract test."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from examples.deepagents_ov_context.chat.config import load_commit_defaults
from examples.deepagents_ov_context.chat.server import (
    ActiveRun,
    ChatSettings,
    SessionInfo,
    create_app,
)
from openviking.integrations.langchain.ov_context import OVContext


def messages(user: str) -> list[BaseMessage]:
    return [
        HumanMessage(
            id=f"{user}-user",
            name=user,
            content=[
                {"type": "text", "text": f"{user}-question"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
            additional_kwargs={"source": user},
        ),
        AIMessage(
            id=f"{user}-assistant",
            content="",
            name="assistant",
            tool_calls=[{"id": f"{user}-call", "name": "debug_echo", "args": {"text": user}}],
            response_metadata={"model": "test-chat"},
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        ),
        ToolMessage(
            id=f"{user}-tool",
            tool_call_id=f"{user}-call",
            name="debug_echo",
            content=[{"type": "text", "text": f"{user}-original-tool-body"}],
            artifact={"owner": user, "values": [1, 2]},
            status="success",
            response_metadata={"source": user},
        ),
    ]


async def exercise(mode: str, directory: Path) -> dict[str, Any]:
    if mode not in {"write", "write-crash", "read"}:
        raise ValueError("Unknown checkpoint worker mode")
    writing = mode != "read"
    config = directory / "model.json"
    if writing:
        directory.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps(
                {
                    "api_key": "test-only",
                    "api_base": "https://model.invalid/v1",
                    "chat_model": "test-chat",
                    "embedding_model": "test-embedding",
                    "embedding_dimension": 1024,
                }
            )
        )
    example_env = (
        Path(__file__).parents[3] / "examples/deepagents_ov_context/config/chat.example.env"
    )
    settings = ChatSettings(
        config,
        directory,
        account_id="checkpoint-account",
        commit_config=load_commit_defaults(example_env),
    )
    app = create_app(settings)
    requests: list[str] = []

    def unexpected_network(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        raise AssertionError("Restoring a checkpoint must not read OV or call a model")

    async with app.router.lifespan_context(app):
        rt = app.state.runtime
        await rt.ov_http.aclose()
        rt.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(unexpected_network))
        for user in ("alice", "bob"):
            session = SessionInfo(
                session_id=f"session-{user}",
                account_id="checkpoint-account",
                user_id=user,
                agent_id=f"agent-{user}",
                created_at=1,
            )
            if writing:
                session.config_snapshot = rt.snapshot(session)
                rt.store.save(session)
            else:
                session = rt.store.get(session.session_id)
            sdk = rt.sdk()
            graph, middleware = rt.graph(sdk, ActiveRun())
            raw = messages(user)
            cache = [HumanMessage(id=f"{user}-summary", content=f"{user}-summary"), *raw]
            cache[-1] = cache[-1].model_copy(update={"content": f"{user}-short-view"})
            expected = OVContext(
                session_id=session.session_id,
                messages=cache,
                archive_id="archive-9",
                write_index=3,
                completed_runs=2,
                commit_count=1,
                context_status="ready",
                last_completed_run="run-previous",
                last_activity=123.5,
                pending_commit={"task_id": "task-10", "archive_id": "archive-9"},
                owner=session.identity("inspect").session_key,
            )
            try:
                if writing:
                    await graph.aupdate_state(
                        session.graph_config,
                        {"messages": raw, "ov_context": expected.to_state()},
                        as_node="OpenVikingContextMiddleware.after_agent",
                    )
                else:
                    values = (await graph.aget_state(session.graph_config)).values
                    assert [m.model_dump() for m in values["messages"]] == [
                        m.model_dump() for m in raw
                    ]
                    assert [type(m) for m in values["messages"]] == [
                        HumanMessage,
                        AIMessage,
                        ToolMessage,
                    ]
                    restored = OVContext.from_state(values["ov_context"])
                    assert restored.to_state() == expected.to_state()
                    view = await rt.context(session)
                    assert view.source == "checkpoint"
                    assert view.cached_message_count == 4 and view.write_index == 3
                    assert view.messages[-1]["content"] == f"{user}-short-view"
                    assert all(
                        other not in json.dumps(view.model_dump())
                        for other in (("bob",) if user == "alice" else ("alice",))
                    )
            finally:
                await middleware.aclose()
                await sdk.close()
        if mode == "write-crash":
            # Checkpoints have committed; exit without lifespan/connection cleanup.
            print(
                json.dumps({"pid": os.getpid(), "mode": mode, "ov_requests": len(requests)}),
                flush=True,
            )
            os._exit(0)
        connection = rt.saver.conn
    try:
        await connection.execute("SELECT 1")
    except ValueError:
        pass
    else:
        raise AssertionError("Lifespan did not close the SQLite connection")
    return {"pid": os.getpid(), "mode": mode, "ov_requests": len(requests), "sessions": 2}


if __name__ == "__main__":
    result = asyncio.run(exercise(sys.argv[1], Path(sys.argv[2])))
    print(json.dumps(result))
