# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Shared token estimation tests."""

import math
import random

import pytest

from openviking.message import Message, TextPart
from openviking.session import Session
from openviking.utils.token_estimation import estimate_text_tokens, text_token_weight

# Keep the original weights as an independent oracle for optimized counting.
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xFF00, 0xFFEF),
    (0x3000, 0x303F),
)


def _reference_weight(text):
    return sum(
        1.5
        if any(start <= ord(char) <= end for start, end in _CJK_RANGES)
        else 2.0
        if ord(char) > 0xFFFF
        else 0.25
        for char in text
    )


@pytest.mark.parametrize(
    "text,weight",
    [(None, 0.0), ("", 0.0), ("abc\n\x00", 1.25), ("汉😀𠀀a", 5.25)],
)
def test_text_weight_and_rounding(text, weight):
    assert text_token_weight(text) == weight
    assert estimate_text_tokens(text) == math.ceil(weight)


@pytest.mark.parametrize(
    "point",
    sorted(
        {0, 0x7F, 0x80, 0xD800, 0xDFFF, 0xFFFF, 0x10000, 0x10FFFF}
        | {point for start, end in _CJK_RANGES for point in (start - 1, start, end, end + 1)}
    ),
)
def test_character_class_boundaries_keep_original_weight(point):
    text = chr(point)
    assert text_token_weight(text) == _reference_weight(text)


def test_all_unicode_code_points_keep_original_weight():
    for start in range(0, 0x110000, 4096):
        text = "".join(chr(point) for point in range(start, min(start + 4096, 0x110000)))
        assert text_token_weight(text) == _reference_weight(text), hex(start)


def test_mixed_text_keeps_additive_weights_and_rounds_once():
    rng = random.Random(20261105)
    for _ in range(300):
        text = "".join(chr(rng.randrange(0x110000)) for _ in range(rng.randrange(1024)))
        expected = _reference_weight(text)
        split = len(text) // 2
        assert text_token_weight(text) == expected
        assert text_token_weight(text[:split]) + text_token_weight(text[split:]) == expected
        assert estimate_text_tokens(text) == math.ceil(expected)


def test_message_estimated_tokens_is_cjk_aware():
    """Chinese text should not be estimated with the ASCII chars/4 fallback."""
    msg = Message(id="msg-cjk", role="user", parts=[TextPart("\u4f60\u597d\u4e16\u754c")])

    assert msg.estimated_tokens == 6


async def test_archive_overview_tokens_do_not_trust_stale_low_metadata():
    class FakeFS:
        async def read_file(self, uri, ctx=None):
            del uri, ctx
            return '{"overview_tokens": 1}'

    fake_session = type("FakeSession", (), {"_viking_fs": FakeFS(), "ctx": None})()

    tokens = await Session._read_archive_overview_tokens(
        fake_session,
        "viking://session/test/history/archive_001",
        "\u4f60\u597d\u4e16\u754c",
    )

    assert tokens == 6


async def test_archive_overview_tokens_keep_higher_metadata_estimate():
    class FakeFS:
        async def read_file(self, uri, ctx=None):
            del uri, ctx
            return '{"overview_tokens": 5}'

    fake_session = type("FakeSession", (), {"_viking_fs": FakeFS(), "ctx": None})()

    tokens = await Session._read_archive_overview_tokens(
        fake_session,
        "viking://session/test/history/archive_001",
        "abcd",
    )

    assert tokens == 5
