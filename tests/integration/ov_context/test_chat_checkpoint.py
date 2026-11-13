"""Real SQLite persistence across independently created Chat processes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from examples.deepagents_ov_context.chat.server import ActiveRun, ChatRuntime, create_app
from openviking.integrations.langchain.ov_context import OVContext

from .test_chat_api import new_session


@pytest.mark.parametrize("write_mode", ["write", "write-crash"])
def test_native_messages_and_ovcontext_survive_two_chat_processes(
    tmp_path: Path, write_mode: str
) -> None:
    results: list[dict[str, Any]] = []
    for mode in (write_mode, "read"):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration.ov_context.chat_checkpoint_worker",
                mode,
                str(tmp_path),
            ],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert results[0]["pid"] != results[1]["pid"]
    assert all(item["ov_requests"] == 0 for item in results)
    assert (tmp_path / "checkpoints.sqlite3").is_file()
    assert (tmp_path / "sessions.sqlite3").is_file()


async def test_restore_deletes_only_selected_persistent_checkpoint(
    chat: tuple[httpx.AsyncClient, ChatRuntime],
) -> None:
    client, rt = chat
    assert isinstance(rt.saver, AsyncSqliteSaver)
    session_ids = [await new_session(client, user_id=user) for user in ("alice", "bob")]
    for sid in session_ids:
        session = rt.store.get(sid)
        message = HumanMessage(id=sid, content=session.user_id)
        sdk = rt.sdk()
        graph, middleware = rt.graph(sdk, ActiveRun())
        try:
            await graph.aupdate_state(
                session.graph_config,
                {
                    "messages": [message],
                    "ov_context": OVContext(
                        session_id=sid,
                        messages=[message],
                        write_index=1,
                        owner=session.identity("inspect").session_key,
                    ).to_state(),
                },
                as_node="OpenVikingContextMiddleware.after_agent",
            )
        finally:
            await middleware.aclose()
            await sdk.close()
    await rt.ov_http.aclose()
    rt.ov_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                404, json={"status": "error", "error": {"code": "NOT_FOUND", "message": "missing"}}
            )
        )
    )
    assert (await client.post(f"/api/sessions/{session_ids[0]}/restore")).status_code == 200
    # Reopen the app (new SQLite connection), not just a new graph.
    fresh = create_app(rt.settings)
    async with fresh.router.lifespan_context(fresh):
        other: ChatRuntime = fresh.state.runtime
        for index, sid in enumerate(session_ids):
            session = other.store.get(sid)
            saved = await other.saver.aget_tuple(session.graph_config)
            assert (saved is None) == (index == 0)
        bob = await other.context(other.store.get(session_ids[1]))
        assert bob.source == "checkpoint" and bob.messages[0]["content"] == "bob"
