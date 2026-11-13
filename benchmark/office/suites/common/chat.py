"""Read Chat SSE events and preserve a terminal execution receipt."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .files_config import write_json


async def chat_events(
    response: httpx.Response,
    journal: Path | None = None,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if response.is_error and journal is not None:
        body = await response.aread()
        write_json(
            journal.parent / "chat-response-error.json",
            {"status": response.status_code, "body": body.decode("utf-8", errors="replace")},
        )
    response.raise_for_status()
    result: list[dict[str, Any]] = []
    event = "message"
    data: list[str] = []

    def record() -> None:
        result.append({"event": event, "data": json.loads("\n".join(data))})
        if journal is not None:
            with journal.with_suffix(".jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result[-1], ensure_ascii=False) + "\n")
        if on_event is not None:
            on_event(result[-1])

    def consume(line: str) -> None:
        nonlocal event, data
        if not line:
            if data:
                record()
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))

    pending = ""
    try:
        async for chunk in response.aiter_text():
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                consume(line.removesuffix("\r"))
        if pending:
            consume(pending.removesuffix("\r"))
        if data:
            record()
    finally:
        if journal is not None:
            write_json(journal, result)
    if not result or result[-1]["event"] not in {"done", "error", "cancelled"}:
        raise RuntimeError("Chat stream ended without a terminal event")
    return result
