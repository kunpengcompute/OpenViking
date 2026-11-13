# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Reusable record loading and event reduction; business rules come from the case inputs."""

from datetime import datetime

from openpyxl import load_workbook


def read_records(path, sheet):
    """第一行为唯一表头；返回全部原字段及实际Excel行号。"""
    wb = load_workbook(path, data_only=True)
    try:
        ws = wb[sheet]
        headers = [c.value for c in ws[1]]
        if any(h is None for h in headers) or len(set(headers)) != len(headers):
            raise ValueError("需要完整、唯一的表头")
        return [
            {"raw": dict(zip(headers, values, strict=True)), "source_row": row_no}
            for row_no, values in enumerate(ws.iter_rows(min_row=2, values_only=True), 2)
            if any(v is not None for v in values)
        ]
    finally:
        wb.close()


def effective_events(records, period_end):
    """按期末截断、event_id去重、同一尝试最大seq处理；保留被排除原记录。"""
    seen, latest, excluded = {}, {}, []
    for record in records:
        raw = record["raw"]
        value = raw["event_time"]
        at = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if at >= period_end:
            excluded.append({"record": record, "reason": "期末及以后"})
            continue
        event_id = raw["event_id"]
        if event_id in seen:
            old = seen[event_id]["raw"]
            if any(raw[k] != old[k] for k in raw if k != "export_batch"):
                raise ValueError("相同event_id存在非导出批次字段冲突")
            excluded.append({"record": record, "reason": "重复导出"})
            continue
        seen[event_id] = record
        key = (raw["intent_id"], raw["phase"], raw["attempt_id"])
        old = latest.get(key)
        if old is None or raw["seq"] > old["raw"]["seq"]:
            if old is not None:
                excluded.append({"record": old, "reason": "被更大seq取代"})
            latest[key] = record
        else:
            excluded.append({"record": record, "reason": "seq不高于已保留状态"})
    return list(latest.values()), excluded
