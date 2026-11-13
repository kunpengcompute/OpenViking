"""Negative controls for the live acceptance script's independent assertions."""

from copy import deepcopy
from typing import Any

import pytest

from tests.st.ov_context.validate_multi_session import (
    ValidationError,
    validate_messages,
)


def exchange() -> list[dict[str, Any]]:
    return [
        {"type": "system", "content": "fixed rules"},
        {"type": "human", "id": "u", "content": "CASE-OWN"},
        {"type": "ai", "id": "a", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {"type": "tool", "id": "t2", "content": "second", "tool_call_id": "c2"},
        {"type": "tool", "id": "t1", "content": "first", "tool_call_id": "c1"},
    ]


def test_acceptance_checks_allow_parallel_tool_results_in_completion_order() -> None:
    messages = exchange()
    before = deepcopy(messages)
    validate_messages(messages, "CASE-OWN", {"CASE-FOREIGN"})
    assert messages == before


@pytest.mark.parametrize(
    "corruption",
    [
        "foreign",
        "missing-own",
        "duplicate-id",
        "missing-id",
        "orphan",
        "missing-result",
        "interleaved-user",
        "duplicate-call",
        "duplicate-result",
    ],
)
def test_acceptance_checks_reject_corrupted_model_history(corruption: str) -> None:
    messages = exchange()
    if corruption == "foreign":
        messages[1]["content"] += " CASE-FOREIGN"
    elif corruption == "missing-own":
        messages[1]["content"] = "unrelated"
    elif corruption == "duplicate-id":
        messages[4]["id"] = messages[1]["id"]
    elif corruption == "missing-id":
        messages[4].pop("id")
    elif corruption == "orphan":
        messages[4]["tool_call_id"] = "unrelated"
    elif corruption == "missing-result":
        messages.pop()
    elif corruption == "interleaved-user":
        messages.insert(3, {"type": "human", "id": "u2", "content": "interrupt"})
    elif corruption == "duplicate-call":
        messages[2]["tool_calls"][1]["id"] = "c1"
    elif corruption == "duplicate-result":
        messages.append({"type": "tool", "id": "t3", "content": "again", "tool_call_id": "c1"})
    with pytest.raises(ValidationError):
        validate_messages(messages, "CASE-OWN", {"CASE-FOREIGN"})
