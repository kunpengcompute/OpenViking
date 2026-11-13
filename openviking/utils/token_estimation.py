# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Shared conservative token estimation helpers."""

from __future__ import annotations

import math
import re
from typing import Any

_CJK_CODE_POINTS = re.compile(
    "[\u1100-\u11ff\u3000-\u30ff\u3130-\u318f\u31f0-\u31ff"
    "\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
    "\uff00-\uffef\U00020000-\U0002ebef]"
)
_NON_BMP_CODE_POINTS = re.compile("[\U00010000-\U0010ffff]")
_CJK_NON_BMP_CODE_POINTS = re.compile("[\U00020000-\U0002ebef]")


def text_token_weight(text: str | None) -> float:
    """Return additive token weight before rounding, including rendered separators."""
    if not text:
        return 0.0
    if text.isascii():
        return len(text) * 0.25
    cjk = len(_CJK_CODE_POINTS.findall(text))
    other_non_bmp = len(_NON_BMP_CODE_POINTS.findall(text)) - len(
        _CJK_NON_BMP_CODE_POINTS.findall(text)
    )
    # Count character classes in the regex engine instead of a Python loop.
    # CJK weighs 1.5, other non-BMP 2.0, and all remaining characters 0.25.
    return len(text) * 0.25 + cjk * 1.25 + other_non_bmp * 1.75


def estimate_text_tokens(text: str | None) -> int:
    """Estimate tokens with a CJK-aware fallback."""
    return math.ceil(text_token_weight(text))


def estimate_serialized_tokens(value: Any) -> int:
    """Estimate tokens for already-structured prompt-like values."""
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_text_tokens(value)
    return estimate_text_tokens(str(value))
