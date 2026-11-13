"""Administrator-only scenario data and objective checks. No source task data."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark.office.datasets.schedule.data import EVENT_LABELS

ZONE = ZoneInfo("Asia/Shanghai")
NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
QUERY = "请把明天这个时间的“产品评审”安排到我的日程里，持续5分钟。请保留工作台原标签页，在另一个新标签页中办理日程。完成并核对结果后，先关闭日程标签页，再关闭工作台，最后告诉我安排结果。"


def connect(folder, namespace):
    if not NAMESPACE.fullmatch(namespace):
        raise ValueError("Invalid namespace")
    directory = Path(folder) / "schedule-state"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(directory / (namespace + ".sqlite3"), timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def expected(anchor):
    when = datetime.fromisoformat(anchor)
    if when.tzinfo is None or when.utcoffset() != timedelta(hours=8):
        raise ValueError("Anchor must include Asia/Shanghai offset")
    start = (when + timedelta(days=1)).replace(second=0, microsecond=0)
    return {
        "owner": "linche-product",
        "calendar_id": "personal",
        "title": "产品评审",
        "start": start.isoformat(timespec="minutes"),
        "end": (start + timedelta(minutes=5)).isoformat(timespec="minutes"),
        "timezone": "Asia/Shanghai",
    }


def provision(folder, namespace, anchor=None):
    if not NAMESPACE.fullmatch(namespace):
        raise ValueError("Invalid namespace")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    registry = folder / (namespace + ".json")
    if registry.exists():
        raise FileExistsError("Refuse to replace existing test data")
    when = (
        datetime.fromisoformat(anchor)
        if anchor
        else datetime.now(ZONE).replace(second=0, microsecond=0)
    )
    target = expected(when.isoformat())
    config = {
        "namespace": namespace,
        "case": "schedule",
        "projects": [],
        "owner": {"id": "linche-product", "name": "林澈", "department": "产品部"},
        "anchor": when.isoformat(timespec="minutes"),
        "timezone": "Asia/Shanghai",
        "scenario_version": "office-schedule-v3-explicit-tabs",
    }
    db = connect(folder, namespace)
    try:
        db.executescript("""
        CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
          calendar_id TEXT NOT NULL, title TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
          notes TEXT NOT NULL DEFAULT '', source TEXT NOT NULL, nonce TEXT UNIQUE NOT NULL);
        CREATE TABLE sessions(token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE audit(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
          event_id INTEGER, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)

        first = when.replace(day=1, hour=9, minute=0, second=0, microsecond=0)
        for i in range(42):
            start = first + timedelta(days=i % 28, hours=(i % 4) * 2)
            # Existing meetings are read-only and include same-title distractors;
            # none is the target event at the exact required minute.
            calendar = ["personal", "department", "rooms"][i % 3]
            if (
                calendar == "personal"
                and EVENT_LABELS[i % len(EVENT_LABELS)] == target["title"]
                and start.isoformat(timespec="minutes") == target["start"]
            ):
                start += timedelta(minutes=45)
            db.execute(
                "INSERT INTO events(owner,calendar_id,title,start,end,notes,source,nonce) VALUES(?,?,?,?,?,?,?,?)",
                (
                    config["owner"]["id"] if calendar == "personal" else "department-team",
                    calendar,
                    EVENT_LABELS[i % len(EVENT_LABELS)],
                    start.isoformat(timespec="minutes"),
                    (start + timedelta(minutes=30)).isoformat(timespec="minutes"),
                    "历史办公安排，非本次待办。",
                    "seed",
                    f"seed-{i}",
                ),
            )
        db.commit()
    finally:
        db.close()
    registry.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    return {
        "namespace": namespace,
        "query": QUERY,
        "expected": target,
        "anchor": config["anchor"],
        "workbench_path": "/workbench/" + namespace,
        "scenario_version": config["scenario_version"],
    }


def inventory(folder, namespace):
    db = connect(folder, namespace)
    try:
        events = [dict(r) for r in db.execute("SELECT * FROM events ORDER BY id")]
        return {
            "events": events,
            "audit": [dict(r) for r in db.execute("SELECT * FROM audit ORDER BY id")],
            "background_sha256": hashlib.sha256(
                json.dumps(
                    [r for r in events if r["source"] == "seed"], sort_keys=True, ensure_ascii=False
                ).encode()
            ).hexdigest(),
        }
    finally:
        db.close()


def objective_checks(reference, before, after):
    """Report facts, never substitute backend state for final-answer quality."""
    created = [e for e in after["events"] if e["source"] == "agent"]
    exact = [
        e
        for e in created
        if all(e[k] == reference[k] for k in ("owner", "calendar_id", "title", "start", "end"))
    ]
    verified_ids = {a["event_id"] for a in after["audit"] if a["action"] == "detail_read"}
    return {
        "created_count": len(created),
        "correct_event_count": len(exact),
        "correct_event_ids": [e["id"] for e in exact],
        "no_extra_events": len(created) == 1 and len(exact) == 1,
        "background_unchanged": before["background_sha256"] == after["background_sha256"],
        "detail_page_loaded": bool(exact) and all(e["id"] in verified_ids for e in exact),
        "backend_task_completed": len(created) == 1
        and len(exact) == 1
        and before["background_sha256"] == after["background_sha256"],
        "final_answer_review": "pending",
        "task_score": None,
    }
