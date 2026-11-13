"""Timesheet fixture initialization, upgrade and persisted work-log checks."""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from benchmark.office.datasets.timesheet.catalog import CARD_VERSION, card_specs

from ..portal.store import connect


def initialize(registry, namespace):
    """Create work-log tables and seed their initial records."""
    db = connect(registry, namespace)
    try:
        db.executescript("""
        CREATE TABLE entries(id INTEGER PRIMARY KEY AUTOINCREMENT, card TEXT NOT NULL,
          minutes INTEGER NOT NULL CHECK(minutes>0), content TEXT NOT NULL, notes TEXT NOT NULL,
          source TEXT NOT NULL, nonce TEXT UNIQUE NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE audit(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
                           entry_id INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        rows = [
            ("demand-ops", 30, "办公平台的验证测试", "办公平台的验证测试"),
            ("demand-ops", 60, "需求梳理", "已完成的历史工作"),
            ("version-maintenance", 45, "版本核对", "历史维护记录"),
            ("training", 60, "团队培训", "历史培训记录"),
        ]
        with db:
            for i, row in enumerate(rows):
                db.execute(
                    "INSERT INTO entries(card,minutes,content,notes,source,nonce) VALUES(?,?,?,?,?,?)",
                    (*row, "seed", "seed-" + str(i)),
                )
    finally:
        db.close()
    upgrade_timesheet(registry, namespace)


def upgrade_timesheet(registry, namespace):
    """Administrator-only, transactional fixture upgrade preserving existing rows."""
    config = json.loads((Path(registry) / (namespace + ".json")).read_text())
    if config.get("case") != "office" or config.get("namespace") != namespace:
        raise ValueError("Office workspace required")
    anchor = config["anchor"]
    day = date.fromisoformat(anchor[:10])
    specs = card_specs(anchor)
    db = connect(registry, namespace)
    try:
        db.execute("BEGIN IMMEDIATE")
        if "work_date" not in {r["name"] for r in db.execute("PRAGMA table_info(entries)")}:
            db.execute("ALTER TABLE entries ADD COLUMN work_date TEXT")
        db.execute("CREATE TABLE IF NOT EXISTS fixture_versions(version TEXT PRIMARY KEY)")
        if db.execute("SELECT 1 FROM fixture_versions WHERE version=?", (CARD_VERSION,)).fetchone():
            db.commit()
            return False
        # Ten target records total 10.5 hours; plan estimate is separately 44.5.
        rows = []
        activities = [
            "需求清单梳理",
            "运营问题跟进",
            "版本需求对齐",
            "阶段方案评审",
            "沟通资料整理",
            "验证记录核对",
            "需求验收跟进",
            "阶段工作复盘",
        ]
        for i, minutes in enumerate([60, 90, 30, 120, 45, 60, 75, 60]):
            rows.append(
                (
                    "demand-ops",
                    minutes,
                    activities[i],
                    activities[i] + "已完成",
                    "board-target-" + str(i),
                    (day - timedelta(days=i + 1)).isoformat(),
                )
            )
        for i, spec in enumerate(specs[1:], 1):
            count = 1 if spec["key"] in ("version-maintenance", "training") else 2
            for j in range(count):
                rows.append(
                    (
                        spec["key"],
                        30 + 15 * ((i + j) % 5),
                        spec["title"] + "阶段核对",
                        "核对资料、记录结果并完成沟通",
                        "board-" + spec["key"] + "-" + str(j),
                        (day - timedelta(days=(i + j) % 15 + 1)).isoformat(),
                    )
                )
        for card, minutes, content, notes, nonce, work_date in rows:
            db.execute(
                "INSERT INTO entries(card,minutes,content,notes,source,nonce,work_date) VALUES(?,?,?,?,?,?,?)",
                (card, minutes, content, notes, "seed", nonce, work_date),
            )
        db.execute("INSERT INTO fixture_versions(version) VALUES(?)", (CARD_VERSION,))
        db.commit()
        return True
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def inventory(registry, namespace):
    db = connect(registry, namespace)
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM entries ORDER BY id")]
        return {
            "entries": rows,
            "demand_minutes": sum(r["minutes"] for r in rows if r["card"] == "demand-ops"),
            "seed_sha256": hashlib.sha256(
                json.dumps(
                    [r for r in rows if r["source"] == "seed"], ensure_ascii=False, sort_keys=True
                ).encode()
            ).hexdigest(),
        }
    finally:
        db.close()


def timesheet_checks(before, after):
    old = {r["id"] for r in before["entries"]}
    added = [r for r in after["entries"] if r["id"] not in old]
    return {
        "created_count": len(added),
        "delta_minutes": after["demand_minutes"] - before["demand_minutes"],
        "correct_entry": len(added) == 1
        and added[0]["card"] == "demand-ops"
        and added[0]["minutes"] == 30
        and added[0]["content"] == "办公平台的验证测试"
        and added[0]["notes"] == "办公平台的验证测试",
        "seed_unchanged": before["seed_sha256"] == after["seed_sha256"],
    }
