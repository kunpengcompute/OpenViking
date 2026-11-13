"""Twelve-result scheduling case with relative dates and persisted ID dependencies."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

from benchmark.office.datasets.schedule.data import DEPENDENCY_SPECS as SPECS

from . import data, weekly

VERSION = "schedule-fortnight-v2"
NOTE = weekly.NOTE

KEYS = ("owner", "calendar_id", "title", "start", "end")


def task_description(login_url: str) -> str:
    rows = [
        f"| {s['title']} | {s['date_rule']} | {'或'.join(s['slots'])} | {s['minutes']}分钟 | {weekly.CALENDARS[s['calendar']]} | {s.get('parent', '—')} |"
        for s in SPECS
    ]
    return (
        f"测试环境入口：{login_url}\n测试账号：100001，林澈（产品部），通过页面选择模拟登录。"
        "操作范围为本次独立测试工作区。通过 browser 的可见网页控件办理；eval 仅用于只读观察。\n\n"
        "请通过办公工作台，为林澈安排2026年11月2日至11月13日的 OV 集成验收会议，时区为北京时间。"
        "请先使用 write_todos 制定计划，执行过程中更新进度，完成以下十二项安排：\n\n"
        "| 日程标题 | 日期或关联规则 | 候选开始时间 | 持续时间 | 保存位置 | 前置会议 |\n|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n执行要求：\n"
        "1. 日期规则中的当天、次日和之后第2天均按自然日计算，依据对应目标会议满足要求的排期日期推导。\n"
        "2. 检查两个日历中的已有安排。目标日历中已经存在标题、日期和起止时间完全一致的日程时直接复用，保留其他已有日程。\n"
        "3. 有多个候选开始时间的会议，选择目标日历中与已有日程及本次新增日程均不重叠的最早时段。\n"
        f"4. 新增日程的备注填写“{NOTE}”。表中有前置会议的新增日程，还须填写前置会议标题和它在系统中的实际日程编号，"
        "格式为“项目：OV集成验收；前置会议：会议标题；前置编号：CAL-实际编号”。前置会议编号取自本次核验的正确日程，已有和本次新增日程均可作为前置。\n"
        "5. 完成后逐项核对实际保存结果，给出十二项排期表，包含标题、推导后的具体日期、起止时间、所属日历、日程编号、"
        "前置会议编号，以及“新增／已存在／未完成”状态。未完成项说明原因。"
    )


def prepare(registry: Path, namespace: str, page_url: str) -> dict:
    # Preserve the first case's page, login, background and distractor definitions.
    weekly.prepare(registry, namespace, page_url)
    seeds = [
        weekly.event("项目复盘", "2026-11-06", "16:00", 30, "department", "fortnight-existing-8"),
        weekly.event("二轮需求确认", "2026-11-09", "09:30", 30, "personal", "fortnight-existing-9"),
        weekly.event("回归联调", "2026-11-10", "10:00", 30, "personal", "fortnight-duration-10"),
        weekly.event("回归验收", "2026-11-11", "14:00", 15, "department", "fortnight-duration-11"),
        weekly.event("交付复盘", "2026-11-12", "16:00", 45, "department", "fortnight-other-day-12"),
        weekly.event(
            "二轮需求确认", "2026-11-09", "09:30", 30, "department", "fortnight-other-calendar-9"
        ),
        weekly.event(
            "回归联调", "2026-11-10", "14:00", 45, "department", "fortnight-other-calendar-10"
        ),
        weekly.event("需求跟进", "2026-11-09", "15:00", 30, "personal", "fortnight-background-1"),
        weekly.event("技术例会", "2026-11-12", "10:00", 60, "department", "fortnight-background-2"),
        weekly.event("业务同步", "2026-11-13", "09:00", 30, "personal", "fortnight-background-3"),
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
    resolved = {}
    # Include prior resolved targets when checking candidate conflicts.
    for spec in SPECS:
        day = (
            spec.get("day")
            or (
                date.fromisoformat(resolved[spec["relative_to"]]["start"][:10])
                + timedelta(days=spec["days"])
            ).isoformat()
        )
        selected = None
        for slot in spec["slots"]:
            proposed = weekly.event(
                spec["title"], day, slot, spec["minutes"], spec["calendar"], "reference"
            )
            exact = [e for e in before["events"] if all(e[k] == proposed[k] for k in KEYS)]
            assert len(exact) <= 1
            occupied = any(
                e["calendar_id"] == proposed["calendar_id"]
                and e["start"] < proposed["end"]
                and e["end"] > proposed["start"]
                for e in [*before["events"], *targets]
            )
            if exact or not occupied:
                selected = {k: proposed[k] for k in (*KEYS, "notes")}
                selected.update(
                    existing_id=exact[0]["id"] if exact else None,
                    parent=spec.get("parent"),
                    date_rule=spec["date_rule"],
                )
                break
        if selected is None:
            raise ValueError(f"No valid slot for {spec['title']}")
        targets.append(selected)
        resolved[spec["title"]] = selected
    assert len(targets) == 12 and sum(t["existing_id"] is not None for t in targets) == 4
    assert resolved["回归联调"]["start"] == "2026-11-10T14:00+08:00"
    assert resolved["回归验收"]["start"] == "2026-11-11T15:00+08:00"
    assert resolved["交付复盘"]["start"] == "2026-11-13T16:00+08:00"
    return {
        "version": VERSION,
        "namespace": namespace,
        "targets": targets,
        "before": before,
        "task": task_description(f"{page_url.rstrip('/')}/mock/login?workspace={namespace}"),
        "forced_two_phase_execution": False,
    }


def check(reference: dict, after: dict) -> dict:
    before = reference["before"]
    old = {e["id"] for e in before["events"]}
    new_ids = {e["id"] for e in after["events"]} - old
    by_title = {
        t["title"]: [e for e in after["events"] if all(e[k] == t[k] for k in KEYS)]
        for t in reference["targets"]
    }
    items = []
    accounted = set()
    for target in reference["targets"]:
        exact = by_title[target["title"]]
        # Pre-existing near matches remain valid background. New competing records do not.
        competing_new = [
            e
            for e in after["events"]
            if e["id"] in new_ids
            and e["calendar_id"] == target["calendar_id"]
            and e["title"] == target["title"]
            and e["start"][:10] == target["start"][:10]
            and e not in exact
        ]
        unique = len(exact) == 1 and not competing_new
        existing = target["existing_id"] is not None
        disposition = bool(
            unique
            and (exact[0]["id"] == target["existing_id"] if existing else exact[0]["id"] in new_ids)
        )
        if disposition:
            accounted.add(exact[0]["id"])
        notes_correct = bool(unique and (existing or NOTE in exact[0]["notes"]))
        parent = target.get("parent")
        parent_ids = []
        if parent:
            parents = by_title[parent]
            if len(parents) == 1:
                parent_ids = [parents[0]["id"]]
        reference_correct = bool(
            existing
            or not parent
            or (
                unique
                and len(parent_ids) == 1
                and parent in exact[0]["notes"]
                and set(re.findall(r"CAL-\d+", exact[0]["notes"])) == {f"CAL-{parent_ids[0]}"}
            )
        )
        completed = bool(unique and disposition and notes_correct and reference_correct)
        items.append(
            {
                "title": target["title"],
                "completed": completed,
                "unique": bool(unique),
                "correct_disposition": disposition,
                "notes_correct": notes_correct,
                "parent_reference_correct": reference_correct,
                "expected_parent_ids": parent_ids,
                "actual_ids": [e["id"] for e in exact],
                "expected_status": "已存在" if existing else "新增",
                "expected": target,
            }
        )
    unchanged = before["background_sha256"] == after["background_sha256"]
    extra = sorted(new_ids - accounted)
    return {
        "version": VERSION,
        "items": items,
        "completed": sum(i["completed"] for i in items),
        "total": 12,
        "completion_rate": sum(i["completed"] for i in items) / 12,
        "new_event_count": len(new_ids),
        "expected_new_event_count": 8,
        "extra_event_ids": extra,
        "background_unchanged": unchanged,
        "all_backend_requirements_met": all(i["completed"] for i in items)
        and unchanged
        and not extra,
        "final_delivery_review": "pending",
        "rounds_affect_score": False,
        "forced_two_phase_execution": False,
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
