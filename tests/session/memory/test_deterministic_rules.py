# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Date anchors preserve precision and always use the recorded message date."""

from datetime import datetime

import pytest

from openviking.session.memory.deterministic_rules import (
    _build_ranges,
    _canonicalize,
    _merge_entity_content,
    _resolve_dates_in_text,
)


@pytest.mark.parametrize(
    "text, dates",
    [
        ("昨天去了北京", ["2023-05-07"]),
        ("I went yesterday", ["2023-05-07"]),
        ("last year", ["2022-01-01"]),
        ("May 7 2023", ["2023-05-07"]),
        ("上周一", ["2023-05-01"]),
        ("2023-02-30", []),
    ],
)
def test_dates_are_anchored_to_message(text, dates):
    _, actual = _resolve_dates_in_text(text, datetime(2023, 5, 8))
    assert actual == dates


def test_unknown_relative_date_has_no_wall_clock_fallback():
    text, dates = _resolve_dates_in_text("yesterday", None)
    assert text == "yesterday"
    assert dates == []
    assert _resolve_dates_in_text("2023-05-07", None)[1] == ["2023-05-07"]


def test_ranges_keep_all_chunks_and_gaps():
    assert _build_ranges([0, 1, 3, 3, 4, 7]) == "0-1,3-4,7"
    assert _build_ranges([]) == ""


def test_entity_append_preserves_old_bullets():
    old = "# caroline\n\n## 原始事实\n- I like painting.\n"
    new = "# caroline\n\n## 原始事实\n- I like painting.\n- I went yesterday.\n"
    result = _merge_entity_content(old, new)
    assert result.count("I like painting.") == 1
    assert "I went yesterday." in result
    assert _merge_entity_content(result, new) == result


@pytest.mark.parametrize(
    "name, key", [("[Melanie]", "melanie"), ("John Smith", "john_smith"), ("小明", "小明")]
)
def test_entity_keys(name, key):
    assert _canonicalize(name) == key
