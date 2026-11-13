# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Recall evidence for the current PPT case's first model call."""

from typing import Any


def memory_in_first_input(trace: dict[str, Any]) -> bool | None:
    """Require real, nonempty recall text in the first observed model input."""
    events = trace.get("events", [])
    first_index = next((i for i, e in enumerate(events) if e.get("event") == "model_input"), None)
    if first_index is None:
        return None
    recalled = [
        e["data"].get("content")
        for e in events[:first_index]
        if e.get("event") == "stage"
        and e.get("data", {}).get("name") == "memory_recall_result"
        and e["data"].get("count", 0) > 0
    ]
    first_input = events[first_index]["data"]

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    first = "\n".join(strings(first_input))
    return any(isinstance(text, str) and text.strip() and text in first for text in recalled)
