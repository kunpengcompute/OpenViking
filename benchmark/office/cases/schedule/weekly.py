"""Eight-result weekly scheduling fixture and deterministic backend checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from benchmark.office.datasets.schedule.data import BACKGROUND_MEETINGS as MEETINGS
from benchmark.office.datasets.schedule.data import CALENDARS
from benchmark.office.datasets.schedule.data import PROJECT_NOTE as NOTE

from ..portal.data import provision as provision_office
from . import data

VERSION = "schedule-weekly-v1"


def task_description(login_url: str) -> str:
    rows = []
    for title, day, slots, minutes, calendar in MEETINGS:
        rows.append(
            f"| {title} | {day} | {'或'.join(slots)}开始，持续{minutes}分钟 | {CALENDARS[calendar]} |"
        )
    return (
        f"测试环境入口：{login_url}\n测试账号：100001，林澈（产品部），通过页面选择模拟登录。"
        "操作范围为本次独立测试工作区。通过 browser 的可见网页控件办理；eval 仅用于只读观察。\n\n"
        "请通过办公工作台，为林澈安排2026年11月2日至11月6日的 OV 集成验收会议，时区为北京时间。"
        "请先使用 write_todos 制定计划，执行过程中更新进度，完成以下八项安排：\n\n"
        "| 日程标题 | 日期 | 时间要求 | 保存位置 |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n执行要求：\n"
        "1. 先检查两个日历中的已有安排。目标日历中已经存在标题、日期和起止时间完全一致的日程时，直接复用。\n"
        "2. 有两个候选时段的会议，选择目标日历中与已有日程不重叠的最早时段。\n"
        f"3. 新增日程的备注统一填写“{NOTE}”，保留其他已有日程。\n"
        "4. 完成后逐项核对实际保存结果，给出一张排期表，包含标题、日期、起止时间、所属日历、日程编号，"
        "以及“新增／已存在／未完成”状态。未完成项说明原因。"
    )


def event(title, day, start, minutes, calendar, nonce):
    begin = datetime.fromisoformat(f"{day}T{start}+08:00")
    return {
        "owner": "linche-product" if calendar == "personal" else "department-team",
        "calendar_id": calendar,
        "title": title,
        "start": begin.isoformat(timespec="minutes"),
        "end": (begin + timedelta(minutes=minutes)).isoformat(timespec="minutes"),
        "notes": NOTE,
        "source": "seed",
        "nonce": nonce,
    }


def prepare(registry: Path, namespace: str, page_url: str) -> dict:
    provision_office(registry, namespace, anchor="2026-10-31T09:00:00+08:00")
    seeds = [
        event("需求澄清", "2026-11-02", "09:30", 30, "personal", "weekly-existing-1"),
        event("验收演练", "2026-11-04", "15:00", 30, "personal", "weekly-existing-5"),
        event("部门例会", "2026-11-02", "10:00", 105, "department", "weekly-conflict-2"),
        event("季度资源沟通", "2026-11-05", "09:45", 75, "department", "weekly-conflict-6"),
        event("接口联调", "2026-11-03", "10:00", 45, "department", "weekly-other-calendar-3"),
        event("数据核对", "2026-11-03", "14:00", 30, "department", "weekly-other-calendar-4"),
        event("项目复盘", "2026-11-06", "16:00", 30, "personal", "weekly-other-calendar-8"),
        event("问题收敛", "2026-11-05", "10:00", 30, "personal", "weekly-other-day-7"),
        event("会前资料整理", "2026-11-02", "16:00", 30, "personal", "weekly-background-1"),
        event("跨团队协作", "2026-11-03", "16:00", 30, "department", "weekly-background-2"),
        event("客户需求沟通", "2026-11-04", "10:00", 30, "personal", "weekly-background-3"),
        event("上线评审资料准备", "2026-11-05", "14:00", 30, "personal", "weekly-background-4"),
        event("产品部例会", "2026-11-06", "09:00", 30, "department", "weekly-background-5"),
    ]
    db = data.connect(registry, namespace)
    try:
        with db:
            for row in seeds:
                db.execute(
                    "INSERT INTO events(owner,calendar_id,title,start,end,notes,source,nonce) VALUES(:owner,:calendar_id,:title,:start,:end,:notes,:source,:nonce)",
                    row,
                )
    finally:
        db.close()
    before = data.inventory(registry, namespace)
    targets = []
    # Derive the earliest free candidate using the same frozen pre-run state.
    for title, day, slots, minutes, calendar in MEETINGS:
        selected = None
        for slot in slots:
            proposed = event(title, day, slot, minutes, calendar, "reference")
            exact = [
                e
                for e in before["events"]
                if all(
                    e[k] == proposed[k] for k in ("owner", "calendar_id", "title", "start", "end")
                )
            ]
            busy = any(
                e["calendar_id"] == calendar
                and e["start"] < proposed["end"]
                and e["end"] > proposed["start"]
                for e in before["events"]
            )
            if exact or not busy:
                selected = {
                    k: proposed[k]
                    for k in ("owner", "calendar_id", "title", "start", "end", "notes")
                }
                selected["existing_id"] = exact[0]["id"] if exact else None
                break
        if selected is None:
            raise ValueError(f"No valid candidate for {title}")
        targets.append(selected)
    assert sum(t["existing_id"] is not None for t in targets) == 2
    assert targets[1]["start"][11:16] == "14:00" and targets[5]["start"][11:16] == "15:00"
    return {
        "version": VERSION,
        "namespace": namespace,
        "targets": targets,
        "before": before,
        "task": task_description(f"{page_url.rstrip('/')}/mock/login?workspace={namespace}"),
    }


def check(reference: dict, after: dict) -> dict:
    before = reference["before"]
    old = {e["id"]: e for e in before["events"]}
    new_ids = {e["id"] for e in after["events"]} - set(old)
    checks = []
    accounted = set()
    for target in reference["targets"]:
        exact = [
            e
            for e in after["events"]
            if all(e[k] == target[k] for k in ("owner", "calendar_id", "title", "start", "end"))
        ]
        same_title_day = [
            e
            for e in after["events"]
            if e["calendar_id"] == target["calendar_id"]
            and e["title"] == target["title"]
            and e["start"][:10] == target["start"][:10]
        ]
        unique = len(exact) == len(same_title_day) == 1
        is_existing = target["existing_id"] is not None
        disposition = unique and (
            exact[0]["id"] == target["existing_id"] if is_existing else exact[0]["id"] in new_ids
        )
        notes = unique and (is_existing or exact[0]["notes"] == target["notes"])
        completed = bool(unique and disposition and notes)
        if completed:
            accounted.add(exact[0]["id"])
        checks.append(
            {
                "title": target["title"],
                "completed": completed,
                "unique": unique,
                "correct_disposition": bool(disposition),
                "notes_correct": bool(notes),
                "expected_status": "已存在" if is_existing else "新增",
                "actual_ids": [e["id"] for e in exact],
                "expected": target,
            }
        )
    unchanged = before["background_sha256"] == after["background_sha256"]
    extra = sorted(new_ids - accounted)
    return {
        "version": VERSION,
        "items": checks,
        "completed": sum(i["completed"] for i in checks),
        "total": len(checks),
        "completion_rate": sum(i["completed"] for i in checks) / len(checks),
        "new_event_count": len(new_ids),
        "expected_new_event_count": 6,
        "extra_event_ids": extra,
        "background_unchanged": unchanged,
        "all_backend_requirements_met": all(i["completed"] for i in checks)
        and unchanged
        and not extra,
        "final_delivery_review": "pending",
        "rounds_affect_score": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check"])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--page-url", default="http://127.0.0.1:19482")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = (
        prepare(args.registry, args.namespace, args.page_url)
        if args.command == "prepare"
        else check(
            json.loads(args.reference.read_text()), data.inventory(args.registry, args.namespace)
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    args.output.chmod(0o600)
    print(
        json.dumps(
            {"command": args.command, "namespace": args.namespace, "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
