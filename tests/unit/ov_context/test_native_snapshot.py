"""Native context snapshot contract needed by the SDK and assembler."""

from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart, ToolPart
from openviking.session import Session
from openviking_cli.exceptions import ResourceExhaustedError


async def test_summary_has_priority_and_oversized_summary_is_an_error():
    session = Session.__new__(Session)
    session.session_id = "s"
    messages = [
        Message(id="old", role="user", parts=[TextPart("old" * 100)]),
        Message(id="new", role="user", parts=[TextPart("new")]),
    ]
    session._collect_session_context_components = AsyncMock(
        return_value={
            "messages": messages,
            "latest_archive": {"overview": "full summary", "overview_tokens": 20},
            "latest_finished_archive_id": "archive_001",
            "total_archives": 1,
            "failed_archives": 0,
        }
    )
    result = await session.get_session_context(20 + messages[-1].estimated_tokens)
    assert result["latest_archive_overview"] == "full summary"
    assert [m["id"] for m in result["messages"]] == ["new"]
    with pytest.raises(ResourceExhaustedError) as raised:
        await session.get_session_context(19)
    assert raised.value.details == {"token_budget": 19, "summary_tokens": 20}


def test_parallel_tool_results_are_selected_as_a_complete_group():
    rows = [
        Message(id="q", role="user", parts=[TextPart("question")]),
        Message(id="a", role="assistant", parts=[ToolPart(tool_id="c1"), ToolPart(tool_id="c2")]),
        Message(id="t2", role="user", parts=[ToolPart(tool_id="c2", tool_output="two")]),
        Message(id="t1", role="user", parts=[ToolPart(tool_id="c1", tool_output="one")]),
    ]
    group_tokens = sum(m.estimated_tokens for m in rows[1:])
    assert Session._budget_recent_messages(rows, group_tokens) == rows[1:]
    assert Session._budget_recent_messages(rows, group_tokens - 1) == []
