"""Four new bookings with same-name history and a genuine free-slot decision."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from benchmark.office.datasets.schedule.data import BOOKING_RULES as RULES
from benchmark.office.datasets.schedule.data import BOOKING_SPECS as SPECS
from benchmark.office.datasets.schedule.data import RULE_TITLE

from . import data, fortnight, weekly

VERSION = "schedule-booking-rules-v3"


KEYS = ("owner", "calendar_id", "title", "start", "end", "notes")


def earliest_start(rows, calendar, begin, finish, minutes):
    lower, upper = datetime.fromisoformat(begin), datetime.fromisoformat(finish)
    candidates = {lower}
    for row in rows:
        if row["calendar_id"] == calendar:
            candidates.add(datetime.fromisoformat(row["end"]))
    for candidate in sorted(candidates):
        end = candidate + timedelta(minutes=minutes)
        if candidate < lower or end > upper:
            continue
        if not any(
            row["calendar_id"] == calendar
            and datetime.fromisoformat(row["start"]) < end
            and datetime.fromisoformat(row["end"]) > candidate
            for row in rows
        ):
            return candidate
    raise ValueError("No continuous free interval within the requested window")


def prepare(registry, namespace, page_url="http://127.0.0.1:19482"):
    registry = Path(registry)
    baseline = fortnight.prepare(registry, namespace, page_url)
    assert len(baseline["before"]["events"]) == 65
    path = registry / (namespace + ".json")
    config = json.loads(path.read_text())
    config["calendar_policy"] = {"reject_overlaps": True}
    config["scenario_version"] = VERSION
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    seeds = []
    rule = weekly.event(RULE_TITLE, "2026-10-31", "08:00", 5, "personal", "rules-document-v1")
    rule["notes"] = RULES
    seeds.append(rule)
    for i, (subject, kind, _day, _start, minutes, cal, prefix) in enumerate(SPECS):
        old = weekly.event(
            prefix + subject, "2026-11-14", f"{9 + i:02}:00", minutes, cal, f"booking-history-{i}"
        )
        old["notes"] = "项目：OV集成验收；会议类型：" + kind
        seeds.append(old)
    seeds.extend(
        [
            weekly.event("部门例会", "2026-11-19", "14:00", 30, "department", "booking-busy-1"),
            weekly.event("上线沟通", "2026-11-19", "14:45", 15, "department", "booking-busy-2"),
        ]
    )
    rows = [*baseline["before"]["events"], *seeds]
    chosen = earliest_start(
        rows, "department", "2026-11-19T14:00+08:00", "2026-11-19T16:00+08:00", 30
    )
    targets = []
    for i, (subject, kind, day, start, minutes, cal, prefix) in enumerate(SPECS):
        if i == 3:
            start = chosen.strftime("%H:%M")
        target = weekly.event(prefix + subject, day, start, minutes, cal, "reference")
        target["notes"] = "项目：OV集成验收；会议类型：" + kind
        targets.append({k: target[k] for k in KEYS})
        other = weekly.event(
            prefix + subject,
            day,
            start,
            minutes,
            "personal" if cal == "department" else "department",
            f"booking-other-calendar-{i}",
        )
        other["notes"] = target["notes"]
        seeds.append(other)
    db = data.connect(registry, namespace)
    try:
        with db:
            for seed in seeds:
                cur = db.execute(
                    "INSERT INTO events(owner,calendar_id,title,start,end,notes,source,nonce) VALUES(:owner,:calendar_id,:title,:start,:end,:notes,:source,:nonce)",
                    seed,
                )
                if seed["title"] == RULE_TITLE:
                    rule_id = cur.lastrowid
    finally:
        db.close()
    before = data.inventory(registry, namespace)
    return {
        "version": VERSION,
        "namespace": namespace,
        "rule_id": rule_id,
        "before": before,
        "targets": targets,
        "task": task_description(page_url, namespace),
    }


def task_description(page_url, namespace):
    return f"""【环境信息】
入口：{page_url}/mock/login?workspace={namespace}
账号：100001，林澈（产品部），通过页面选择登录。
操作范围：本轮独立工作区，通过浏览器页面办理。

【任务】
请为林澈新增以下四场会议，项目名称为“OV集成验收”。请查阅日程管理中的资料日程《产品部会议办理规范》，按规范补齐标题、结束时间、保存日历和备注。

| 会议主题 | 会议类型 | 日期 | 时间要求 |
|---|---|---|---|
| 支付接口方案 | 需求评审 | 2026年11月16日 | 15:00开始 |
| 清算链路 | 技术联调 | 2026年11月17日 | 10:00开始 |
| 对账异常 | 问题跟进 | 2026年11月18日 | 10:30开始 |
| 版本V2 | 阶段验收 | 2026年11月19日 | 14:00—16:00内最早可用的连续30分钟 |

其中“对账异常”这场指定持续20分钟，保存到产品部共享日历。时间均为北京时间，前三场时段已确认可用；第四场按目标日历的实际占用安排。按表顺序逐项新增，保留已有日程。
完成后从日历定位本次新增记录，读取详情并汇总四场的标题、起止时间、所属日历和实际日程编号；未完成项说明原因。"""


def check(reference, after):
    old = {r["id"]: r for r in reference["before"]["events"]}
    current = {r["id"]: r for r in after["events"]}
    new_ids = set(current) - set(old)
    matched = set()
    items = []
    for target in reference["targets"]:
        exact = [r for r in after["events"] if all(r[k] == target[k] for k in KEYS)]
        ids = [r["id"] for r in exact]
        correct = len(exact) == 1 and ids[0] in new_ids and exact[0]["source"] == "agent"
        if correct:
            matched.update(ids)
        items.append(
            {
                "title": target["title"],
                "completed": correct,
                "actual_ids": ids,
                "expected": target,
                "same_title_candidates": [
                    r["id"] for r in after["events"] if r["title"] == target["title"]
                ],
            }
        )
    unchanged = all(current.get(i) == r for i, r in old.items())
    completed = sum(r["completed"] for r in items)
    return {
        "items": items,
        "completed": completed,
        "total": 4,
        "completion_rate": completed / 4,
        "new_event_count": len(new_ids),
        "background_unchanged": unchanged,
        "unmatched_new_event_ids": sorted(new_ids - matched),
        "all_backend_requirements_met": completed == 4 and len(new_ids) == 4 and unchanged,
    }
