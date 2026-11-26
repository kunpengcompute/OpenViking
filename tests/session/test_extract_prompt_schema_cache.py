# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""记忆写入侧 schema 指纹缓存的单元测试。

被测对象：`openviking/session/memory/extract_loop.py` 中的
  - `_prompt_schema_cache_enabled()` 开关判定
  - `_schemas_fingerprint()` schema 内容指纹
  - `_PROMPT_SCHEMA_CACHE` 进程内 LRU

缓存只与「schema 内容 / 输出语言 / 输出格式 / link 开关 / peer 范围」这些
静态条件有关，与对话内容无关，因此跨 commit 可以复用同一批 Pydantic 模型类。
"""

from types import SimpleNamespace

import pytest

from openviking.session.memory import extract_loop as mod

ENV = "OPENVIKING_PROMPT_SCHEMA_CACHE"


class FakeSchema:
    """只需要 model_dump()，与真实 MemorySchema 的行为一致。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def model_dump(self):
        return {"name": self.name, "fields": [{"name": "content", "type": "str"}]}


@pytest.fixture(autouse=True)
def _reset_schema_cache(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    mod._PROMPT_SCHEMA_CACHE.clear()
    yield
    mod._PROMPT_SCHEMA_CACHE.clear()


# ---------------------------------------------------------------------------
# 开关
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _enable_kunpeng_opt(monkeypatch):
    """单测在任意架构上开启鲲鹏优化总开关（CI 多为 x86）。"""
    monkeypatch.setattr("openviking._kunpeng_opt.enabled", lambda: True)


def test_cache_enabled_by_default():
    assert mod._prompt_schema_cache_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "disable", "disabled"])
def test_off_values_disable_cache(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert mod._prompt_schema_cache_enabled() is False


def test_other_values_keep_cache_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert mod._prompt_schema_cache_enabled() is True


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------


def test_same_schema_yields_same_fingerprint():
    a = [FakeSchema("profile"), FakeSchema("event")]
    b = [FakeSchema("profile"), FakeSchema("event")]
    assert mod._schemas_fingerprint(a) == mod._schemas_fingerprint(b)


def test_changed_schema_yields_different_fingerprint():
    before = [FakeSchema("profile")]
    after = [FakeSchema("profile"), FakeSchema("event")]
    assert mod._schemas_fingerprint(before) != mod._schemas_fingerprint(after)


def test_fingerprint_is_short_and_stable():
    fp = mod._schemas_fingerprint([FakeSchema("profile")])
    assert len(fp) == 16
    assert fp == mod._schemas_fingerprint([FakeSchema("profile")])


def test_non_serializable_schema_falls_back_to_str():
    """schema 里出现非 JSON 原生类型时不能抛错，用 default=str 兜住。"""

    class WeirdSchema:
        def model_dump(self):
            return {"when": object()}

    fp = mod._schemas_fingerprint([WeirdSchema()])
    assert len(fp) == 16


# ---------------------------------------------------------------------------
# 缓存键与 LRU
# ---------------------------------------------------------------------------


def _key(schemas, language="zh", output_format="python", link=True, peers=()):
    return (
        mod._schemas_fingerprint(schemas),
        language,
        output_format,
        link,
        tuple(peers),
    )


def test_same_static_conditions_produce_same_key():
    schemas = [FakeSchema("profile")]
    assert _key(schemas) == _key([FakeSchema("profile")])


def test_output_settings_change_the_key():
    schemas = [FakeSchema("profile")]
    base = _key(schemas)
    assert base != _key(schemas, language="en")
    assert base != _key(schemas, output_format="json")
    assert base != _key(schemas, link=False)
    assert base != _key(schemas, peers=("alice",))


def test_cache_is_bounded_and_evicts_oldest():
    limit = mod._PROMPT_SCHEMA_CACHE_MAX
    for i in range(limit):
        mod._PROMPT_SCHEMA_CACHE[_key([FakeSchema(f"s{i}")])] = object()
    assert len(mod._PROMPT_SCHEMA_CACHE) == limit

    oldest = next(iter(mod._PROMPT_SCHEMA_CACHE))
    while len(mod._PROMPT_SCHEMA_CACHE) >= limit:
        mod._PROMPT_SCHEMA_CACHE.popitem(last=False)
    mod._PROMPT_SCHEMA_CACHE[_key([FakeSchema("new")])] = object()

    assert len(mod._PROMPT_SCHEMA_CACHE) == limit
    assert oldest not in mod._PROMPT_SCHEMA_CACHE


def test_hit_moves_key_to_the_end():
    k1 = _key([FakeSchema("a")])
    k2 = _key([FakeSchema("b")])
    mod._PROMPT_SCHEMA_CACHE[k1] = SimpleNamespace(tag="a")
    mod._PROMPT_SCHEMA_CACHE[k2] = SimpleNamespace(tag="b")

    mod._PROMPT_SCHEMA_CACHE.move_to_end(k1)

    assert list(mod._PROMPT_SCHEMA_CACHE)[-1] == k1
