# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic recall previews built from stored facts and operation fields."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from openviking.session.memory.utils.messages import parse_memory_file_with_fields

from .recall_methods import procedure_units

# Keep the existing 400-character preview size, selecting complete lines.
_PREVIEW_CHARS = 400
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_FIELD = re.compile(r"^(?:[-*]\s+)?([^:：\n]+)[:：]\s*(.*)$")
_GROUPS = (
    ("summary", "摘要", "概述"),
    (
        "when to use",
        "best for",
        "trigger",
        "situation",
        "applicability boundary",
        "适用条件",
        "适用场景",
    ),
    ("negative applicability", "适用限制"),
    ("optimal params", "recommended flow", "procedure", "approach", "操作步骤", "关键参数"),
    ("guidelines", "方法", "操作要点"),
    ("common failures", "anti-patterns", "reflect", "注意事项", "失败教训"),
    ("key dependencies", "preconditions", "前置条件"),
    ("recommendation", "建议"),
    ("good cases", "bad cases"),
)
_NAMES = {"tool", "skill"}
_SECTION_KEYS = {key for group in _GROUPS for key in group}
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _key(label: str) -> str:
    return label.strip().strip("*").replace("_", " ").lower()


def memory_view(text: str) -> dict[str, Any]:
    """Use the same hidden-field cleanup as the content read endpoint."""
    view = parse_memory_file_with_fields(text)
    # Indexed excerpts can end in the middle of an internal comment.
    view["content"] = re.sub(r"<!--\s*MEMORY_FIELDS\b.*\Z", "", view["content"], flags=re.S).strip()
    return view


def _units(text: str) -> Iterator[str]:
    """Treat fenced commands and JSON as indivisible source units."""
    block: list[str] = []
    marker = ""
    for line in text.splitlines():
        fence = _FENCE.match(line)
        if block:
            block.append(line)
            if fence and fence[1][0] == marker[0] and len(fence[1]) >= len(marker):
                yield "\n".join(block)
                block, marker = [], ""
        elif fence:
            block, marker = [line], fence[1]
        else:
            yield line
    # An unfinished fenced command is kept in the full view, not the preview.


def _factual_lines(text: str) -> Iterator[str]:
    for line in _units(text):
        if not line.strip() or _HEADING.match(line) or line.strip() in {"```", "---"}:
            continue
        if _FENCE.match(line) and not any(part.strip() for part in line.splitlines()[1:-1]):
            continue
        field = _FIELD.match(line)
        if field and (_key(field[1]) in _NAMES or not field[2].strip()):
            continue
        yield line


def has_information(text: str) -> bool:
    """Reject known heading/name-only placeholders without interpreting facts."""
    return next(_factual_lines(text), None) is not None


def _sections(
    text: str, fields: Mapping[str, Any] | None = None
) -> tuple[str, dict[str, tuple[str, list[str]]]]:
    sections: dict[str, tuple[str, list[str]]] = {}
    current = None
    title = ""
    for line in _units(text):
        heading = _HEADING.match(line)
        field = _FIELD.match(line) if line and not line[0].isspace() else None
        match = heading or field
        if match is not None:
            current = _key(match[1])
            prefix = line.strip() if heading else line.split(":" if ":" in line else "：", 1)[0]
            if field:
                prefix += ":"
            sections.setdefault(current, (prefix, []))
            if field and field[2].strip():
                sections[current][1].append(field[2].strip())
            if not title and (
                current in _NAMES
                or (heading and line.startswith("# ") and current not in _SECTION_KEYS)
            ):
                title = line.strip()
        elif current is not None and has_information(line):
            sections[current][1].append(line.rstrip())

    # Fields are a fallback for records whose visible template omits them.
    for label, value in (fields or {}).items():
        key = _key(label)
        if isinstance(value, str) and has_information(value) and key not in sections:
            sections[key] = (f"{label}:", [value.strip()])
    return title, sections


@dataclass(frozen=True)
class MemoryFeatures:
    """Invocation-local source fields, not a new persisted memory schema."""

    title: str = ""
    domain: str = ""
    trigger: str = ""
    family: str = ""
    target: str = ""
    steps: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    historical_role: str = "unknown"
    has_artifacts: bool = False

    @property
    def methods(self) -> tuple[str, ...]:
        return self.steps + self.parameters + self.warnings


def memory_features(text: str) -> MemoryFeatures:
    """Extract explicit fields and literal procedure evidence without inventing steps."""
    fields = memory_view(text)
    content = fields["content"]
    title, sections = _sections(content, fields)

    def units(*names: str) -> tuple[str, ...]:
        return tuple(
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for name in names
            for line in sections.get(name, ("", []))[1]
            if has_information(line)
        )

    operation = "\n".join(units("operation intent"))

    def operation_field(name: str) -> str:
        match = re.search(rf"(?im)^\s*(?:[-*]\s*)?{name}:\s*(.+)$", operation)
        return match[1].strip() if match else ""

    summary = content.split("\n\n", 1)[0]
    role = "unknown"
    if re.search(
        r"全部.*(?:完成|成功)|已(?:保存|创建|完成|确认)|完成后生成|\b(?:completed|created|saved)\b",
        summary,
        re.I,
    ):
        role = "result"
    elif re.search(r"用户要求|用户希望|\b(?:requested|asked)\b", summary, re.I):
        role = "request"
    return MemoryFeatures(
        title=title,
        domain="\n".join(units("domain")),
        trigger="\n".join(units("trigger", "when to use", "best for", "applicability boundary")),
        family=operation_field("family"),
        target=operation_field("primary object"),
        steps=units("procedure", "recommended flow", "guidelines", "操作步骤", "方法")
        or procedure_units(_units(content)),
        parameters=units("optimal params", "关键参数", "recommendation"),
        conditions=units("preconditions", "前置条件"),
        warnings=units("anti-patterns", "common failures", "失败教训"),
        exclusions=units("negative applicability", "适用限制"),
        historical_role=role,
        has_artifacts=bool(
            re.search(
                r"/api/sessions/[^\s]+/artifacts/|\[[^\]]+\]\([^)]*\.(?:png|csv|txt|xlsx|pdf|docx)",
                content,
                re.I,
            )
        ),
    )


def overview(text: str, fields: Mapping[str, Any] | None = None) -> str:
    """Select useful sections and complete lines without inventing instructions.

    One line per section is considered before adding detail to any section.
    This only constructs a preview; recall ordering and output budgets belong
    to the caller.
    """
    title, sections = _sections(text, fields)

    selected = []
    for group in _GROUPS:
        for key in group:
            if key in sections and sections[key][1]:
                selected.append(sections[key])
                break
    primary_count = sum(
        any(key in sections and sections[key][1] for key in group) for group in _GROUPS[:-2]
    )

    if selected:
        kept: dict[int, list[str]] = {}

        def render() -> str:
            blocks = [
                prefix + "\n" + "\n".join(kept[i])
                for i, (prefix, _) in enumerate(selected)
                if i in kept
            ]
            return "\n\n".join(([title] if title else []) + blocks)

        # Preserve conditions, actions and failures before optional examples
        # and recommendations that often repeat the same instructions.
        for phase in (range(primary_count), range(primary_count, len(selected))):
            for index in range(max((len(selected[i][1]) for i in phase), default=0)):
                for i in phase:
                    lines = selected[i][1]
                    if index >= len(lines):
                        continue
                    kept.setdefault(i, []).append(lines[index])
                    if len(render()) > _PREVIEW_CHARS:
                        kept[i].pop()
                        if not kept[i]:
                            del kept[i]
        if kept:
            return render()

    # General documents retain factual text rather than a heading list.
    complete = []
    for line in _factual_lines(text):
        if len("\n".join([*complete, line])) <= _PREVIEW_CHARS:
            complete.append(line)
    return "\n".join(complete)
