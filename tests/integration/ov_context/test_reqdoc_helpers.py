# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared helper preserves evidence and is available equally in both Q arms."""

import copy
import importlib.util
from datetime import datetime

import pytest
from openpyxl import Workbook, load_workbook

from benchmark.office.cases.reqdoc_generation.files import load_case
from benchmark.paths import PROJECT_ROOT


@pytest.fixture
def helper():
    _, sources = load_case()
    path = next(p for p in sources if p.name == "record_helpers.py")
    spec = importlib.util.spec_from_file_location("record_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_arms_upload_same_helper_without_embedded_function_body():
    spec, sources = load_case()
    assert len(sources) == 7 and len({p.name for p in sources}) == 7
    assert "record_helpers.py" in {item["filename"] for item in spec["inputs"]}
    h, q = (PROJECT_ROOT / spec["phases"][phase]["prompt"] for phase in ["H", "Q"])
    assert h.read_text().startswith(q.read_text())
    assert "inputs/record_helpers.py" in q.read_text()
    assert "可在计算脚本中按需使用" in q.read_text()
    assert "from record_helpers import read_records, effective_events" in h.read_text()
    assert "(工作表名, source_row)" in h.read_text()
    assert "from record_helpers import" not in q.read_text()
    assert "def read_records" not in h.read_text()
    assert "def effective_events" not in h.read_text()


def test_original_fields_and_excel_rows_survive_blank_rows(tmp_path, helper):
    path = tmp_path / "records.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "records"
    ws.append(["id", "notice", "retry_of", "workspace_kind"])
    ws.append(["first", "原文提示", None, "controlled"])
    ws.append([None, None, None, None])
    ws.append(["retry", None, "first", "ordinary"])
    wb.save(path)
    wb.close()
    records = helper.read_records(path, "records")
    assert [r["source_row"] for r in records] == [2, 4]
    assert records[0]["raw"]["notice"] == "原文提示"
    assert records[1]["raw"]["retry_of"] == "first"
    assert all(set(r["raw"]) == {"id", "notice", "retry_of", "workspace_kind"} for r in records)


@pytest.mark.parametrize("headers", [["id", "id"], ["id", None]])
def test_invalid_headers_fail_before_records_are_silently_overwritten(tmp_path, helper, headers):
    path = tmp_path / "bad.xlsx"
    wb = Workbook()
    wb.active.append(headers)
    wb.active.append(["a", "b"])
    wb.save(path)
    wb.close()
    with pytest.raises(ValueError, match="唯一的表头"):
        helper.read_records(path, wb.sheetnames[0])


def event(row, event_id, seq, at, *, export_batch="batch-a"):
    return {
        "source_row": row,
        "raw": {
            "event_id": event_id,
            "intent_id": "sample-intent",
            "phase": "install",
            "attempt_id": "sample-attempt",
            "seq": seq,
            "event_time": at,
            "export_batch": export_batch,
            "notice": "完整原文",
            "retry_of": None,
            "workspace_kind": "controlled",
        },
    }


def test_cutoff_precedes_sequence_merge_and_exclusions_preserve_full_records(helper):
    before = "2026-10-07T23:59:00+08:00"
    end = datetime.fromisoformat("2026-10-08T00:00:00+08:00")
    rows = [
        event(2, "old", 1, before),
        event(3, "current", 2, before),
        event(4, "current", 2, before, export_batch="batch-b"),
        event(5, "future", 3, end),
        event(6, "lower", 1, before),
    ]
    original = copy.deepcopy(rows)
    kept, excluded = helper.effective_events(rows, end)
    assert kept == [rows[1]]
    assert {r["record"]["source_row"]: r["reason"] for r in excluded} == {
        2: "被更大seq取代",
        4: "重复导出",
        5: "期末及以后",
        6: "seq不高于已保留状态",
    }
    assert {r["source_row"] for r in kept} | {r["record"]["source_row"] for r in excluded} == {
        2,
        3,
        4,
        5,
        6,
    }
    assert rows == original
    assert all(r["record"] in rows for r in excluded)


def test_conflicting_duplicate_export_is_rejected(helper):
    rows = [event(2, "same", 1, "2026-10-01T00:00:00+08:00")]
    rows.append(copy.deepcopy(rows[0]))
    rows[1]["raw"]["notice"] = "发生冲突"
    with pytest.raises(ValueError, match="字段冲突"):
        helper.effective_events(rows, datetime.fromisoformat("2026-10-08T00:00:00+08:00"))


def test_actual_workbook_keeps_all_fields_and_partitions_original_rows(helper):
    spec, _ = load_case()
    path = PROJECT_ROOT / spec["input_dir"] / "metrics.xlsx"
    wb = load_workbook(path, data_only=True)
    try:
        for sheet in ["意图台账", "事件导出"]:
            records = helper.read_records(path, sheet)
            expected = {
                row: dict(zip(next(wb[sheet].values), values, strict=True))
                for row, values in enumerate(wb[sheet].iter_rows(min_row=2, values_only=True), 2)
                if any(v is not None for v in values)
            }
            assert {r["source_row"]: r["raw"] for r in records} == expected
        kept, excluded = helper.effective_events(
            records, datetime.fromisoformat("2026-10-08T00:00:00+08:00")
        )
        combined = kept + [r["record"] for r in excluded]
        assert len(combined) == len(records)
        assert {r["source_row"]: r["raw"] for r in combined} == expected
    finally:
        wb.close()
