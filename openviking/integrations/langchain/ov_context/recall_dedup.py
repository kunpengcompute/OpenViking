# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Invocation-local content deduplication, committed only after budget admission."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .recall_content import has_information

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})[^\n]*$")
_CHATLOG = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[^\n]*?\s+)?(?:ChatLog|聊天记录|对话记录)\s*[:：]\s*$",
    re.I,
)
_ROLE = re.compile(r"^\s*\[(?:user|assistant|human|ai)\]\s*[:：]\s*", re.I)


def _closes(line: str, marker: str) -> bool:
    return bool(re.fullmatch(rf"\s*{re.escape(marker[0])}{{{len(marker)},}}\s*", line))


def _blocks(text: str):
    paragraph: list[str] = []
    code: list[str] = []
    marker = ""
    for line in text.replace("\r\n", "\n").splitlines():
        fence = _FENCE.match(line)
        if code:
            code.append(line)
            if _closes(line, marker):
                yield True, "\n".join(code)
                code, marker = [], ""
        elif fence:
            if paragraph:
                yield False, "\n".join(paragraph)
                paragraph = []
            code, marker = [line], fence[1]
        elif not line.strip() or _CHATLOG.match(line):
            if paragraph:
                yield False, "\n".join(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if code:
        # An incomplete fence is also atomic; never deduplicate individual code lines.
        yield True, "\n".join(code)
    if paragraph:
        yield False, "\n".join(paragraph)


def _key(code: bool, text: str) -> str:
    if code:
        lines = text.splitlines()
        marker = _FENCE.match(lines[0])[1]
        end = -1 if len(lines) > 1 and _closes(lines[-1], marker) else None
        language = lines[0].strip()[len(marker) :].strip()
        body = "\n".join(lines[1:end])
        return language + "\n" + body if body.strip() else ""
    lines = []
    for line in text.splitlines():
        line = _ROLE.sub("", line)
        line = re.sub(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)、]\s+|>\s*)", "", line)
        lines.append(line)
    value = " ".join(lines).strip()
    value = re.sub(r"^Summary\s*[:：]\s*", "", value, flags=re.I)
    value = re.sub(
        r"^(?:用户(?:要求|请求|希望)|[Tt]he user (?:asked|requested)(?: to)?)\s*", "", value
    )
    value = re.sub(r"^请(?:帮我)?\s*", "", value)
    return re.sub(r"\s+", " ", value.replace("`", "").replace("**", "")).strip()


def _covered(key: str, references: set[str]) -> bool:
    # Short answers such as "已付款" are not duplicates of a question containing them.
    return key in references or (len(key) >= 48 and any(key in text for text in references))


@dataclass(frozen=True)
class DeduplicatedContent:
    text: str
    query_blocks: int = 0
    memory_blocks: int = 0


class RecallDeduplicator:
    """Compare paragraphs and whole code blocks without changing stored memories.

    Query normalization removes presentation wrappers, never numbers, dates or
    code spelling/indentation. ``filter`` is speculative: ``remember`` is called
    only after a rendered entry has passed all output budgets.
    """

    def __init__(self, query: str, *, enabled: bool = True):
        self.enabled = enabled
        self.query = {False: set(), True: set()}
        self.accepted: dict[str, dict[bool, set[str]]] = {}
        for code, block in _blocks(query):
            self.query[code].add(_key(code, block))

    def filter(self, text: str, *, scope: str = "") -> DeduplicatedContent:
        # Event lookups can retain occurrence identity even when visible facts
        # match. New execution tasks share the empty scope for content dedup.
        if not self.enabled:
            return DeduplicatedContent(text)
        kept = []
        kept_code = False
        accepted = self.accepted.get(scope, {False: set(), True: set()})
        seen = {kind: set(values) for kind, values in accepted.items()}
        query_blocks = memory_blocks = 0
        for code, block in _blocks(text):
            key = _key(code, block)
            if not key:
                continue
            query_duplicate = key in self.query[True] if code else _covered(key, self.query[False])
            memory_duplicate = key in seen[True] if code else _covered(key, seen[False])
            if query_duplicate:
                query_blocks += 1
            elif memory_duplicate:
                memory_blocks += 1
            else:
                kept.append(block)
                seen[code].add(key)
                kept_code = kept_code or code
        content = "\n\n".join(kept)
        return DeduplicatedContent(
            content if kept_code or has_information(content) else "", query_blocks, memory_blocks
        )

    def remember(self, text: str, *, scope: str = "") -> None:
        if self.enabled:
            accepted = self.accepted.setdefault(scope, {False: set(), True: set()})
            for code, block in _blocks(text):
                accepted[code].add(_key(code, block))
