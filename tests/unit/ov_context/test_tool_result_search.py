"""Exact source offsets, concurrent requests and bounded merged output."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest

from openviking.integrations.langchain.ov_context.tool_result_search import search_many


def response(content: str, query: str, limit: int = 20) -> dict[str, Any]:
    return {
        "tool_result_id": "result-1",
        "matches": [
            {
                "offset": match.start(),
                "offset_unit": "unicode_code_point",
                "snippet": content[max(0, match.start() - 300) : match.end() + 300],
            }
            for match in list(re.finditer(re.escape(query), content))[:limit]
        ],
    }


async def test_searches_are_concurrent_and_merge_exact_unicode_windows() -> None:
    content = "🙂" * 350 + "北京 AQI 上海 空气" + "尾" * 350
    entered: set[str] = set()
    ready = asyncio.Event()

    async def one(query: str, limit: int) -> dict[str, Any]:
        entered.add(query)
        if len(entered) == 3:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        return response(content, query, limit)

    result = await search_many(one, ["北京", "上海", "AQI"])
    assert len(result["windows"]) == 1
    window = result["windows"][0]
    assert window["snippet"] == content[window["start_offset"] : window["end_offset"]]
    assert set(window["queries"]) == entered
    assert result["returned_source_chars"] < result["unmerged_snippet_chars"]
    assert result["partial"] is False
    assert "_match_offset" not in window


async def test_source_budget_preserves_first_hit_for_each_term_and_signals_partial() -> None:
    content = ("x" * 800 + "needle") * 25 + "y" * 800 + "稀有词" + "z" * 300

    async def one(query: str, limit: int) -> dict[str, Any]:
        return response(content, query, limit)

    result = await search_many(one, ["needle", "稀有词"])
    assert result["returned_source_chars"] <= 6000
    assert result["returned_source_chars"] == sum(len(w["snippet"]) for w in result["windows"])
    assert result["omitted_match_windows"] > 0 and result["partial"] is True
    assert any("稀有词" in window["queries"] for window in result["windows"])
    for window in result["windows"]:
        assert window["snippet"] == content[window["start_offset"] : window["end_offset"]]


async def test_duplicates_empty_matches_and_literal_queries() -> None:
    content = "a|b is a literal, not an OR expression"
    calls: list[str] = []

    async def one(query: str, limit: int) -> dict[str, Any]:
        calls.append(query)
        return response(content, query, limit)

    result = await search_many(one, [" a|b ", "a|b", "absent"])
    assert calls == ["a|b", "absent"]
    assert [item["returned_match_count"] for item in result["query_stats"]] == [1, 0]
    assert result["windows"][0]["snippet"] == content
    assert result["partial"] is False


async def test_no_hits_still_identifies_original() -> None:
    async def one(query: str, limit: int) -> dict[str, Any]:
        return response("no matches", query, limit)

    result = await search_many(one, ["missing"])
    assert result["tool_result_id"] == "result-1"
    assert result["windows"] == [] and result["returned_source_chars"] == 0
    assert result["partial"] is False


@pytest.mark.parametrize("fault", ["result_id", "offset_unit", "overlap"])
async def test_inconsistent_search_responses_are_not_silently_merged(fault: str) -> None:
    async def one(query: str, limit: int) -> dict[str, Any]:
        result = response("alpha middle beta", query, limit)
        if query == "beta":
            if fault == "result_id":
                result["tool_result_id"] = "another-result"
            elif fault == "offset_unit":
                result["matches"][0]["offset_unit"] = "bytes"
            else:
                result["matches"][0]["snippet"] = "X" + result["matches"][0]["snippet"][1:]
        return result

    with pytest.raises(ValueError):
        await search_many(one, ["alpha", "beta"])


async def test_sdk_failure_is_propagated_without_partial_success() -> None:
    async def one(query: str, limit: int) -> dict[str, Any]:
        if query == "beta":
            raise RuntimeError("OV unavailable")
        return response("alpha beta", query, limit)

    with pytest.raises(RuntimeError, match="OV unavailable"):
        await search_many(one, ["alpha", "beta"])
