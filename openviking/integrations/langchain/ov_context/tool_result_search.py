# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Bound concurrent literal searches and merge exact original-text windows."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

MAX_QUERIES = 4
MAX_QUERY_CHARS = 200
MAX_MATCHES = 20
MAX_SOURCE_CHARS = 6000
# The existing search_tool_result HTTP contract uses this context on each side.
CONTEXT_CHARS = 300


def _merge(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: (item["start_offset"], item["end_offset"])):
        current = {**window, "queries": list(window["queries"])}
        if not merged or current["start_offset"] > merged[-1]["end_offset"]:
            merged.append(current)
            continue
        previous = merged[-1]
        offset = current["start_offset"] - previous["start_offset"]
        overlap = min(previous["end_offset"], current["end_offset"]) - current["start_offset"]
        if previous["snippet"][offset : offset + overlap] != current["snippet"][:overlap]:
            raise ValueError("Search snippets disagree at overlapping source offsets")
        if current["end_offset"] > previous["end_offset"]:
            previous["snippet"] += current["snippet"][
                previous["end_offset"] - current["start_offset"] :
            ]
            previous["end_offset"] = current["end_offset"]
        previous["queries"] = list(dict.fromkeys(previous["queries"] + current["queries"]))
    return merged


async def search_many(
    search_one: Callable[[str, int], Awaitable[dict[str, Any]]],
    queries: list[str],
    limit: int = MAX_MATCHES,
) -> dict[str, Any]:
    """Search one original with 1..4 terms and return at most 6000 source characters."""
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f"queries must contain 1..{MAX_QUERIES} strings")
    if any(
        not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS
        for query in queries
    ):
        raise ValueError(f"Each query must contain 1..{MAX_QUERY_CHARS} nonblank characters")
    if type(limit) is not int or not 1 <= limit <= MAX_MATCHES:
        raise ValueError(f"limit must be 1..{MAX_MATCHES} per query")
    normalized = list(dict.fromkeys(query.strip() for query in queries))
    results = await asyncio.gather(*(search_one(query, limit) for query in normalized))
    identifiers = {result.get("tool_result_id") for result in results}
    if len(identifiers) != 1 or None in identifiers:
        raise ValueError("Candidate searches must return the same original result")

    windows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for query, result in zip(normalized, results, strict=True):
        matches = result["matches"]
        stats.append(
            {
                "query": query,
                "returned_match_count": len(matches),
                "at_limit": len(matches) >= limit,
            }
        )
        for match in matches:
            offset, text = match["offset"], match["snippet"]
            left = max(0, offset - CONTEXT_CHARS)
            if (
                match.get("offset_unit") != "unicode_code_point"
                or text[offset - left : offset - left + len(query)] != query
            ):
                raise ValueError("Search response does not match the context/offset contract")
            windows.append(
                {
                    "start_offset": left,
                    "end_offset": left + len(text),
                    "snippet": text,
                    "queries": [query],
                    "_match_offset": offset,
                }
            )
    all_merged = _merge(windows)
    # Cover each term's first hit before filling the remaining budget in source order.
    priority: list[dict[str, Any]] = []
    for query in normalized:
        first = next((window for window in windows if query in window["queries"]), None)
        if first is not None:
            priority.append(first)
    priority.extend(sorted(windows, key=lambda item: (item["start_offset"], item["end_offset"])))
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    selected: list[dict[str, Any]] = []
    for window in priority:
        key = (window["start_offset"], window["end_offset"], tuple(window["queries"]))
        if key in seen:
            continue
        seen.add(key)
        candidate = _merge([*selected, window])
        if sum(len(item["snippet"]) for item in candidate) <= MAX_SOURCE_CHARS:
            selected = candidate
    for window in selected:
        for original in windows:
            query, hit = original["queries"][0], original["_match_offset"]
            if (
                window["start_offset"] <= hit
                and hit + len(query) <= window["end_offset"]
                and query not in window["queries"]
            ):
                window["queries"].append(query)
        window.pop("_match_offset", None)
    omitted = sum(
        not any(
            window["start_offset"] <= candidate["start_offset"]
            and candidate["end_offset"] <= window["end_offset"]
            for window in selected
        )
        for candidate in windows
    )
    return {
        "tool_result_id": next(iter(identifiers)),
        "query_stats": stats,
        "windows": selected,
        "offset_unit": "unicode_code_point",
        "offset_range": "start_inclusive_end_exclusive",
        "returned_source_chars": sum(len(window["snippet"]) for window in selected),
        "unmerged_snippet_chars": sum(len(window["snippet"]) for window in windows),
        "merged_source_chars_before_budget": sum(len(window["snippet"]) for window in all_merged),
        "omitted_match_windows": omitted,
        "partial": bool(omitted or any(stat["at_limit"] for stat in stats)),
    }
