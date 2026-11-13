# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Verify the live Chat SSE API, checkpoint reuse, OV restore and a real tool call."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


async def stream_chat(client: httpx.AsyncClient, session_id: str, prompt: str) -> dict[str, Any]:
    started = time.monotonic()
    events: list[dict[str, Any]] = []
    async with client.stream(
        "POST", "/api/chat", json={"session_id": session_id, "message": prompt}
    ) as response:
        response.raise_for_status()
        kind, data = "", []
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
            elif not line and data:
                events.append(
                    {
                        "event": kind,
                        "data": json.loads("\n".join(data)),
                        "seconds": round(time.monotonic() - started, 4),
                    }
                )
                data = []
    assert events[-1]["event"] == "done", events[-1]
    tokens = [event for event in events if event["event"] == "token"]
    assert tokens and tokens[0]["seconds"] < events[-1]["seconds"]
    messages = [event["data"]["message"] for event in events if event["event"] == "message"]
    answers = [message["content"] for message in messages if message["type"] == "ai"]
    inputs = [event["data"]["messages"] for event in events if event["event"] == "model_input"]
    return {
        "prompt": prompt,
        "reply": answers[-1],
        "token_events": len(tokens),
        "first_token_seconds": tokens[0]["seconds"],
        "done_seconds": events[-1]["seconds"],
        "model_inputs": inputs,
        "context": events[-1]["data"]["context"],
        "events": events,
    }


async def verify(url: str, output: Path) -> dict[str, Any]:
    marker = "CHAT-" + uuid.uuid4().hex[:12]
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=120) as client:
        health = await client.get("/api/health")
        health.raise_for_status()
        created = await client.post("/api/sessions", json={"user_id": "smoke-user"})
        created.raise_for_status()
        session = created.json()
        sid = session["session_id"]
        blank = await client.get(f"/api/sessions/{sid}")
        blank.raise_for_status()
        assert blank.json()["context"]["source"] == "empty"
        runs = []
        first = await stream_chat(
            client,
            sid,
            f"记住调试口令 {marker}。不要调用任何工具。分三行简要说明什么是会话历史、消息缓存和流式回复。",
        )
        runs.append(first)
        assert first["token_events"] > 3
        first_count = first["context"]["raw_message_count"]
        assert first["context"]["write_index"] == first_count >= 2
        second = await stream_chat(client, sid, "只回复刚才给出的调试口令，不调用工具。")
        runs.append(second)
        assert marker in second["reply"]
        second_count = second["context"]["raw_message_count"]
        assert second_count == second["context"]["cached_message_count"] >= first_count + 2
        assert not any(
            event["data"].get("name") == "context.initial_load"
            for event in second["events"]
            if event["event"] == "stage"
        )
        restored = await client.post(f"/api/sessions/{sid}/restore")
        restored.raise_for_status()
        assert restored.json()["context"]["source"] == "ov"
        third = await stream_chat(
            client,
            sid,
            "调用 debug_echo 工具回显本会话第一轮给出的调试口令。工具成功后，只回复工具返回的口令。",
        )
        runs.append(third)
        assert marker in third["reply"]
        tools = [m for m in third["context"]["messages"][second_count:] if m["type"] == "tool"]
        assert len(tools) == 1 and marker in tools[0]["content"]
        calls = [call for m in third["context"]["messages"] for call in m.get("tool_calls", [])]
        assert any(call["id"] == tools[0]["tool_call_id"] for call in calls)
        assert third["context"]["raw_message_count"] == third["context"]["write_index"] == 4
        assert third["context"]["cached_message_count"] == second_count + 4
        assert len(third["model_inputs"]) == 2
        assert any(m["type"] == "tool" for m in third["model_inputs"][-1])
        # Repeated inspection must not advance the write cursor or persist messages again.
        for _ in range(2):
            detail = (await client.get(f"/api/sessions/{sid}")).json()
            assert detail["context"]["write_index"] == 4 and not detail["busy"]
        report = {
            "url": url,
            "health": health.json(),
            "session": session,
            "runs": runs,
            "checks": {
                "incremental_stream": True,
                "checkpoint_reuse": True,
                "ov_restore": True,
                "real_tool_roundtrip": True,
                "tool_pairing_preserved": True,
                "write_cursor": True,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "session_id": sid,
                    "replies": [run["reply"] for run in runs],
                    "tokens": [run["token_events"] for run in runs],
                    "checks": report["checks"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1934")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.url, args.output))


if __name__ == "__main__":
    main()
