"""Negative controls: recalling from raw history cannot pass the memory case."""

import json

import httpx
import pytest

from tests.st.ov_context.validate_memory import (
    QUERY,
    SEED,
    run_chat,
    wait_extraction,
)
from tests.st.ov_context.validate_multi_session import ValidationError


def event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


@pytest.mark.parametrize("mode", ["injected", "raw-history-only", "archived-id", "other-user"])
async def test_case_requires_injection_and_rejects_archived_or_foreign_history(mode: str) -> None:
    current = {"type": "human", "id": "query", "content": QUERY, "additional_kwargs": {}}
    if mode != "raw-history-only":
        current["content"] = "[参考记忆]\n" + SEED + "\n[用户原文]\n" + QUERY
        current["additional_kwargs"] = {"ov_context_injected": True}
    history = {"type": "human", "id": "archived", "content": SEED}
    messages = [current] if mode != "archived-id" else [history, current]
    body = event("stage", {"name": "memory.recall", "status": "success", "count": 2})
    body += event("model_input", {"messages": messages})
    body += event("message", {"message": {"type": "ai", "content": SEED}})
    body += event(
        "done",
        {
            "context": {"messages": [], "write_index": 2, "raw_message_count": 2},
            "statistics": {"total_tokens": 10},
        },
    )
    async with httpx.AsyncClient(
        base_url="http://chat.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    ) as client:
        kwargs = {"expected_memory": mode != "other-user", "forbidden_id": "archived"}
        if mode == "injected":
            result, _ = await run_chat(client, {"session_id": "s"}, QUERY, mode, **kwargs)
            assert result.injected and result.answer_uses_preferences
        else:
            with pytest.raises(ValidationError):
                await run_chat(client, {"session_id": "s"}, QUERY, mode, **kwargs)


@pytest.mark.parametrize("status,count", [("completed", 0), ("failed", 2), ("unknown", 2)])
async def test_no_memory_or_failed_task_is_not_extraction_success(status: str, count: int) -> None:
    async with httpx.AsyncClient(
        base_url="http://chat.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"status": status, "memories_extracted": {"memory_write": count}}
            )
        ),
    ) as client:
        with pytest.raises(ValidationError):
            await wait_extraction(client, "s", 1)


@pytest.mark.parametrize(
    "source,content,allowed",
    [
        ("viking://user/other/memories/.abstract.md", "Long-term memory storage.", True),
        ("viking://user/seed/memories/.abstract.md", "Long-term memory storage.", False),
        ("viking://user/other/memories/preferences/style.md", "Some preference.", False),
        ("viking://user/other/memories/.abstract.md", SEED, False),
        ("", "Unknown reference.", False),
    ],
)
async def test_other_user_allows_only_own_empty_directory_metadata(source, content, allowed):
    reference = content + (f"\n来源：{source}" if source else "")
    current = {
        "type": "human",
        "id": "query",
        "content": f"[参考记忆]\n{reference}\n[用户原文]\n{QUERY}",
        "additional_kwargs": {"ov_context_injected": True},
    }
    body = event("stage", {"name": "memory.recall", "status": "success", "count": 1})
    body += event("model_input", {"messages": [current]})
    body += event("message", {"message": {"type": "ai", "content": "No known preference."}})
    body += event(
        "done",
        {
            "context": {"messages": [], "write_index": 2, "raw_message_count": 2},
            "statistics": {"total_tokens": 10},
        },
    )
    async with httpx.AsyncClient(
        base_url="http://chat.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    ) as client:
        session = {"session_id": "s", "user_id": "other"}
        if allowed:
            result, _ = await run_chat(client, session, QUERY, "other-user", expected_memory=False)
            assert result.directory_only_reference and not result.injected
        else:
            with pytest.raises(ValidationError):
                await run_chat(client, session, QUERY, "other-user", expected_memory=False)
