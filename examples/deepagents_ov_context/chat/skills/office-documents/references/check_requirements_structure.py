#!/usr/bin/env python3
"""Generic Markdown structure check, with no case facts or expected numbers.

Usage: python3 check_requirements_structure.py output/requirements.md [--json]
Passing proves only structure. Read the actual sources to verify business
rules, metric values, calculations, source support and executable acceptance.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SECTIONS = (
    "需求背景",
    "需求目标",
    "需求价值",
    "关键指标",
    "功能概述",
    "详细业务规则",
    "预研范围",
    "验收标准",
    "待确认事项",
    "资料来源",
)
ALIASES = {
    "id": ("指标ID", "指标编号", "指标标识"),
    "name": ("指标名称",),
    "description": ("指标说明", "说明", "统计口径"),
    "target": ("目标值", "目标"),
    "baseline": ("基准值", "基准"),
}


def plain(value: str) -> str:
    return re.sub(r"[*`_]", "", value).strip()


def markdown_tables(text: str) -> list[dict[str, Any]]:
    """Read pipe tables, allowing reordered columns and escaped pipe content."""
    tables = []
    current: dict[str, Any] | None = None
    previous: list[str] | None = None
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            current, previous = None, None
            continue
        cells = [
            plain(c.replace(r"\|", "|")) for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]
        separator = bool(cells) and all(
            re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells
        )
        if separator and previous is not None and len(previous) == len(cells):
            current = {"headers": previous, "rows": []}
            tables.append(current)
        elif current is not None:
            current["rows"].append(cells)
        previous = cells
    return tables


def metric_tables(text: str) -> list[dict[str, Any]]:
    result = []
    for table in markdown_tables(text):
        columns = {}
        for name, aliases in ALIASES.items():
            hits = [
                i
                for i, cell in enumerate(table["headers"])
                if any(a.casefold() in re.sub(r"\s+", "", cell).casefold() for a in aliases)
            ]
            if len(hits) == 1:
                columns[name] = hits[0]
        if not {"name", "description", "target", "baseline"}.issubset(columns):
            continue
        rows = []
        for cells in table["rows"]:
            if len(cells) != len(table["headers"]):
                rows.append({"malformed": True, "cells": cells})
                continue
            row = {name: cells[index] for name, index in columns.items()}
            identity = row.get("id", row["name"])
            match = re.search(r"(?<![A-Za-z0-9-])([A-Za-z]+-\d+)(?![A-Za-z0-9-])", identity)
            row["id"] = match.group(1) if match else None
            row["cells"] = cells
            rows.append(row)
        result.append({**table, "columns": columns, "metric_rows": rows})
    return result


def assess(text: str) -> dict[str, Any]:
    headings = [plain(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)]
    checks = {
        "utf8_text_nonempty": bool(text.strip()) and "\x00" not in text,
        **{f"section:{name}": any(name in h for h in headings) for name in SECTIONS},
    }
    tables = metric_tables(text)
    checks["one_main_metric_table"] = len(tables) == 1
    rows = tables[0]["metric_rows"] if len(tables) == 1 else []
    checks["metric_rows_present"] = bool(rows)
    checks["metric_rows_well_formed"] = bool(rows) and all(not r.get("malformed") for r in rows)
    checks["metric_identifiers_unique"] = (
        bool(rows)
        and all(r.get("id") for r in rows)
        and len({r.get("id") for r in rows}) == len(rows)
    )
    checks["metric_cells_nonempty"] = bool(rows) and all(
        all(r.get(k, "").strip() for k in ("name", "description", "target", "baseline"))
        for r in rows
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scope": "structure only; metric correctness, business semantics and source support require independent review",
    }


def check(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise ValueError("expected a regular output file")
        return assess(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        return {"passed": False, "checks": {"readable_utf8_file": False}, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("output/requirements.md"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for name, passed in result["checks"].items():
            print(("PASS " if passed else "FAIL ") + name)
        print("STRUCTURE_PASS" if result["passed"] else "STRUCTURE_FAIL")
        print(
            "Structure only: independently verify facts, calculations, source support and acceptance."
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
