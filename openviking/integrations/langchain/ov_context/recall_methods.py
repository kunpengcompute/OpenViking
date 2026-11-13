# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Literal procedure evidence shared by ordinary-memory parsing and selection."""

from __future__ import annotations

import re
from collections.abc import Iterable

_ACTION = re.compile(
    r"使用|采用|通过|调用|读取|加载|解析|提取|返回|保留|保存|输出|区分|筛选|归并|计算|校验|核验|核对|验证|复算|"
    r"先.+再|按章节|逐章|\b(?:use|read|load|parse|extract|return|preserve|write|validate|assert)\b|"
    r"\b(?:read|load|parse|extract|write)_[a-z_]+|\b(?:PdfReader|Document)\s*\(",
    re.I,
)
_TECHNICAL = re.compile(
    r"\b[a-z_]\w*\([^\n)]*\)|\.\b(?:xlsx?|csv|pdf|docx?|pptx?|json|md)\b|"
    r"工作簿|工作表|原字段|原行号|分子|分母|公式|JSON|\b(?:raw|source_row|event_id|seq|retry_of)\b",
    re.I,
)
_NEGATIVE = re.compile(r"^\s*(?:不要|禁止|不得|无需|不必|do not\b|never\b)", re.I)
_REQUEST = re.compile(r"用户(?:要求|希望|下达)|\buser (?:asked|requested)\b", re.I)
_ROLE = re.compile(r"^\s*\[([^]\n]+)\]:\s*(.*)$")
_SUMMARY = re.compile(r"^(?:summary|摘要)\s*[:：]", re.I)
_MECHANISM = re.compile(r"\b[a-z_]\w*\([^\n)]*\)|source_row|原行号|先.+再", re.I)


def procedure_units(units: Iterable[str]) -> tuple[str, ...]:
    """Keep source-backed actions, including prose lists and complete code fences.

    User quotations and unfinished fences cannot establish a provided method.
    No operation family, business domain or successful outcome is inferred.
    """
    result: list[str] = []
    user_quote = False
    for unit in units:
        role = _ROLE.match(unit)
        if role:
            user_quote = role[1].lower() in {"user", "human", "system"}
            unit = role[2]
        if user_quote:
            continue
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", unit).strip()
        if not text or text.startswith("#") or _NEGATIVE.search(text) or _REQUEST.search(text):
            continue
        # Completion summaries describe outcomes. Only explicit mechanisms can
        # establish a reusable method; a filename or computed value is insufficient.
        if _SUMMARY.match(text) and not _MECHANISM.search(text):
            continue
        if _ACTION.search(text) and _TECHNICAL.search(text) and text not in result:
            result.append(text)
    return tuple(result)


def file_method_capabilities(text: str) -> set[str]:
    """Recognize concrete file/data operations without task-specific names."""
    caps: set[str] = set()

    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, re.I | re.S))

    if has(
        r"读取|加载|解析|提取|\bread\b|\bload\b|load_|read_|parse|extract|PdfReader|Document"
    ) and has(
        r"工作簿|工作表|原文件|原始资料|\.xlsx?\b|\.csv\b|\.pdf\b|\.docx?\b|\.json\b|"
        r"workbook|spreadsheet|source file|read_csv|PdfReader"
    ):
        caps.add("read_source_file")
    if has(r"原字段|全部字段|完整记录|全部原字段|\braw\b") and has(
        r"原行号|行号|source_row|row.number"
    ):
        caps.add("preserve_source_records")
    if (
        has(r"去重|重复导出|dedup|drop_duplicates")
        and has(r"期末|截止|截断|跨期|cutoff")
        and has(r"最大.{0,12}seq|最大.{0,8}状态|状态修订|逻辑尝试|max.{0,12}seq|latest.state")
    ):
        caps.add("resolve_event_state")
    if (
        has(r"区分|分开|分离|独立|separat|distinguish")
        and has(r"首次|first")
        and has(r"重试|retry")
    ):
        caps.add("separate_attempts")
    if has(r"计算|复算|公式|calculate|compute|formula") and has(
        r"指标|成功率|完成率|分子|分母|统计|总额|metric|ratio|numerator|denominator"
    ):
        caps.add("calculate_metric")
    data_context = has(
        r"工作簿|工作表|文件|原字段|原行号|source_row|(?<!-)json|CSV|PDF|DOCX|XLSX|"
        r"分子|分母|公式|数据|dataframe|workbook"
    )
    if (
        data_context
        and has(r"校验|核验|核对|验证|复算|validate|assert")
        and has(
            r"记录|结果|数值|字段|公式|集合|覆盖|JSON|数据|result|record|data|ratio|numerator|denominator"
        )
    ):
        caps.add("validate_result")
    return caps
