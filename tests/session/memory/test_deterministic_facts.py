# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source attribution, stable identities and conservative rule fallback."""

from datetime import datetime

import pytest

from openviking.session.memory import fact_records
from openviking.session.memory.fact_records import (
    alias_in_text,
    assertion_type,
    make_fact_records,
    merge_fact_records,
)
from openviking.utils.spacy_utils import detect_language


@pytest.mark.parametrize(
    "text, expected",
    [
        ("我昨天参加了活动。", "zh"),
        ("I attended yesterday.", "en"),
        ("", "en"),
    ],
)
def test_language_routing(text, expected):
    assert detect_language(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Did I go?", "question"),
        ("If I went there", "conditional"),
        ("I might go", "uncertain"),
        ("I never went", "negation"),
        ("I will go tomorrow", "plan"),
        ("I go every week", "habit"),
        ("She said I went", "report"),
        ("我昨天去了北京。", "fact"),
    ],
)
def test_assertion_scope(text, expected):
    assert assertion_type(text) == expected


def test_source_identity_and_verbatim_quotes_without_nlp(monkeypatch):
    monkeypatch.setattr(fact_records, "_parse", lambda text: None)

    def resolve(text, ts):
        return text, []

    ts = datetime(2023, 5, 8)
    text = "[Caroline]: 我昨天去了北京。我的儿子叫小明。"
    rows = make_fact_records(text, "Caroline", "D1:3", ts, resolve)
    assert rows
    for row in rows:
        assert row["source_turn_ids"] == ["D1:3"]
        assert row["source_quotes"]["D1:3"] in text
        assert row["speaker"] == "Caroline"
        assert row["recorded_at"] == ts.isoformat()
        assert row["time"] is None
    relation = next(row for row in rows if row["predicate"] == "has_son")
    assert (relation["subject"], relation["object"]) == ("Caroline", "小明")
    again = make_fact_records(text, "Caroline", "D1:3", ts, resolve)
    assert again == rows
    assert len(merge_fact_records(rows, again)) == len(rows)
    other = make_fact_records(text, "Caroline", "D2:3", ts, resolve)
    assert {r["id"] for r in rows}.isdisjoint(r["id"] for r in other)


def test_missing_source_is_not_a_fact():
    assert make_fact_records("我去了北京。", "user", "", None, lambda t, ts: (t, [])) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I completed 4 essays this year.", "this year"),
        ("I have now completed 7 essays this year.", "this year"),
        ("We finished 3 reports last month.", "last month"),
        ("I read 2 books this week.", "this week"),
        ("I completed 4 essays since May.", "since May"),
        ("我今年完成了4篇论文。", "今年"),
        ("I have 3 books on this yearly plan.", None),
    ],
)
def test_personal_fact_calendar_scope(text, expected):
    from openviking.session.memory.atomic_facts import _scope

    assert _scope(text) == expected


def test_alias_matching_does_not_invent_english_entities_without_nlp(monkeypatch):
    monkeypatch.setattr(fact_records, "_parse", lambda text: None)
    assert not alias_in_text("son", "I like this song")
    assert not alias_in_text("John", "John Smith went home", "person")
    assert alias_in_text("John", "John", "person")
    assert alias_in_text("小明", "小明昨天去了北京", "person")


def test_model_name_is_part_of_parse_cache_key(monkeypatch):
    from types import SimpleNamespace

    import openviking.utils.spacy_utils as spacy_utils
    import openviking_cli.utils.config as config

    model = SimpleNamespace(spacy_model_en="first")
    monkeypatch.setattr(config, "get_openviking_config", lambda: SimpleNamespace(memory=model))
    calls = []

    def load(name):
        calls.append(name)
        return lambda text: (name, text)

    monkeypatch.setattr(spacy_utils, "load_spacy_model", load)
    fact_records._parse_with_model.cache_clear()
    try:
        assert fact_records._parse("hello") == ("first", "hello")
        assert fact_records._parse("hello") == ("first", "hello")
        model.spacy_model_en = "second"
        assert fact_records._parse("hello") == ("second", "hello")
        assert calls == ["first", "second"]
    finally:
        fact_records._parse_with_model.cache_clear()
