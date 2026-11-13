# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Verify frozen source materials and independently recompute their metric oracle."""

import datetime as dt
import json
import statistics
from typing import Any

from benchmark.office.cases.reqdoc_generation.files import CASE_PATH, load_case
from benchmark.paths import PROJECT_ROOT

CASE = CASE_PATH
DATASET = (PROJECT_ROOT / json.loads(CASE.read_text())["input_dir"]).parent


def test_independent_event_reduction_matches_private_oracle() -> None:
    """Read actual XLSX, not builder's Python rows; check filters and each set."""
    from openpyxl import load_workbook

    book = load_workbook(DATASET / "inputs/metrics.xlsx", data_only=True)
    assert book["事件导出"].sheet_state == "hidden"

    def rows(name: str) -> list[dict[str, Any]]:
        data = list(book[name].values)
        return [dict(zip(data[0], row, strict=True)) for row in data[1:]]

    start = dt.datetime.fromisoformat("2026-10-01T00:00:00+08:00")
    end = dt.datetime.fromisoformat("2026-10-08T00:00:00+08:00")
    registry = {
        r["intent_id"]: r
        for r in rows("意图台账")
        if r["status"] == "submitted"
        and not r["is_test"]
        and start <= dt.datetime.fromisoformat(r["started_at"]) < end
    }
    by_event = {}
    payloads = {}
    for row in rows("事件导出"):
        payload = {k: v for k, v in row.items() if k != "export_batch"}
        assert row["event_id"] not in payloads or payloads[row["event_id"]] == payload
        payloads[row["event_id"]] = payload
        if row["intent_id"] in registry and dt.datetime.fromisoformat(row["event_time"]) < end:
            by_event.setdefault(row["event_id"], row)
    attempts = {}
    for row in by_event.values():
        key = (row["intent_id"], row["phase"], row["attempt_id"])
        if key not in attempts or row["seq"] > attempts[key]["seq"]:
            attempts[key] = row
    saved = [r for r in attempts.values() if r["phase"] == "market" and r["result"] == "saved"]
    first = [r for r in attempts.values() if r["phase"] == "install" and not r["retry_of"]]
    retry = [r for r in attempts.values() if r["phase"] == "install" and r["retry_of"]]
    terminal = [r for r in first if r["result"] in {"installed", "failed"}]
    assert set(registry) == {f"I{i:03}" for i in range(1, 31)}
    assert {r["intent_id"] for r in saved} == {f"I{i:03}" for i in range(1, 25)}
    assert {r["attempt_id"] for r in first if r["result"] == "pending_approval"} == {"A013", "A014"}
    assert {r["attempt_id"] for r in retry if r["result"] == "installed"} == {
        "R015",
        "R016",
        "R018",
    }
    durations = [
        (
            dt.datetime.fromisoformat(r["event_time"])
            - dt.datetime.fromisoformat(registry[r["intent_id"]]["started_at"])
        ).total_seconds()
        for r in saved
    ]
    values = [
        len(saved) / len(registry) * 100,
        sum(r["result"] == "installed" for r in terminal) / len(terminal) * 100,
        statistics.median(durations),
        sum(r["result"] == "installed" for r in retry) / len(retry) * 100,
        None,
    ]
    oracle = json.loads((DATASET / "oracle.json").read_text())
    assert values == [m["baseline"] for m in oracle["metrics"]]
    assert len(durations) == 24 and sorted(durations)[11:13] == [260, 280]


def test_all_oracle_sources_exist_in_actual_attachments() -> None:
    from docx import Document
    from openpyxl import load_workbook
    from pypdf import PdfReader

    case, inputs = load_case(CASE)
    text = "\n".join(p.read_text() for p in inputs if p.suffix in {".md", ".json"})
    doc = Document(DATASET / "inputs/change_log.docx")
    text += "\n".join(p.text for p in doc.paragraphs)
    text += "\n".join(c.text for table in doc.tables for row in table.rows for c in row.cells)
    book = load_workbook(DATASET / "inputs/metrics.xlsx", data_only=True)
    text += "\n".join(
        str(value) for row in book["统计口径"].values for value in row if value is not None
    )
    pdf = PdfReader(DATASET / "inputs/prototype.pdf")
    assert len(pdf.pages) == 5
    for i, page in enumerate(pdf.pages, 1):
        content = page.extract_text()
        assert f"P-0{i}" in content and "模拟业务" in content
        text += content
    oracle = json.loads((DATASET / "oracle.json").read_text())
    for fact in oracle["facts"]:
        assert all(source in text for source in fact["sources"]), fact["id"]
    assert len(inputs) == 7 and case["output"] == "requirements.md"
    assert sum(p.suffix != ".py" for p in inputs) == 6
    assert any(p.name == "record_helpers.py" for p in inputs)
    assert not any("requirements.md" == p.name for p in inputs)
