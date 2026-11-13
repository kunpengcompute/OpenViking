"""Four independent business goals sharing one resource-booking workflow."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from benchmark.office.datasets.schedule.data import ATTENDEE_COUNTS as COUNTS
from benchmark.office.datasets.schedule.data import MEETING_ROOMS, RESOURCE_BUSY, RESOURCE_RULES

from . import booking_rules, data, resources

VERSION = "schedule-resource-booking-v1"


def prepare(registry, namespace, page_url="http://127.0.0.1:19482"):
    ref = booking_rules.prepare(registry, namespace, page_url)
    path = Path(registry) / (namespace + ".json")
    config = json.loads(path.read_text())
    config["resource_booking"] = True
    config["scenario_version"] = VERSION
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    db = data.connect(registry, namespace)
    try:
        resources.initialize(db)
        with db:
            db.execute(
                "UPDATE events SET notes=notes||? WHERE id=?", (RESOURCE_RULES, ref["rule_id"])
            )

            db.executemany("INSERT INTO meeting_rooms VALUES(?,?,?,?,?,?,?,?)", MEETING_ROOMS)
            for room, day, begin, finish in RESOURCE_BUSY:
                db.execute(
                    "INSERT INTO room_busy(room_id,title,start,end) VALUES(?,?,?,?)",
                    (
                        room,
                        "已有资源预约",
                        day + "T" + begin + "+08:00",
                        day + "T" + finish + "+08:00",
                    ),
                )
        before = data.inventory(registry, namespace)
        rstate = resources.snapshot(db)
    finally:
        db.close()
    targets = []
    for i, original in enumerate(ref["targets"]):
        target = dict(original)
        kind = booking_rules.SPECS[i][1]
        count = COUNTS[i]
        lower = datetime.fromisoformat(target["start"] if i < 3 else "2026-11-19T14:00+08:00")
        upper = datetime.fromisoformat(target["end"] if i < 3 else "2026-11-19T16:00+08:00")
        minutes = booking_rules.SPECS[i][4]
        duration = timedelta(minutes=minutes)
        candidates = {lower}
        if i == 3:
            candidates.update(
                datetime.fromisoformat(r["end"]) for r in [*before["events"], *rstate["room_busy"]]
            )
        choices = []
        for begin in sorted(candidates):
            finish = begin + duration
            if begin < lower or finish > upper:
                continue
            start, end = begin.isoformat(timespec="minutes"), finish.isoformat(timespec="minutes")
            if any(
                r["calendar_id"] == target["calendar_id"] and r["start"] < end and r["end"] > start
                for r in before["events"]
            ):
                continue
            for room in rstate["meeting_rooms"]:
                if resources.eligible(room, kind, count) and not any(
                    r["room_id"] == room["id"] and r["start"] < end and r["end"] > start
                    for r in rstate["room_busy"]
                ):
                    choices.append((start, room["priority"], room["id"], end))
        start, _priority, room_id, end = min(choices)
        target.update(start=start, end=end, room_id=room_id, kind=kind, attendees=count)
        targets.append(target)
    return {
        "version": VERSION,
        "namespace": namespace,
        "rule_id": ref["rule_id"],
        "before": before,
        "resources_before": rstate,
        "targets": targets,
        "task": task_description(page_url, namespace),
    }


def task_description(page_url, namespace):
    return f"""【环境信息】
入口：{page_url}/mock/login?workspace={namespace}
账号：100001，林澈（产品部），通过页面选择登录。操作范围为本轮独立工作区，通过浏览器页面办理。

【任务】
请为“OV集成验收”项目逐项预约以下四场会议，创建日程并落实会议室。请查阅日程管理中的资料日程《产品部会议办理规范》，按规范补齐标题、时长、日历、备注和资源条件，并按其资源选择优先级办理。

| 会议主题 | 类型 | 日期 | 时间要求 | 参会人数 |
|---|---|---|---|---:|
| 支付接口方案 | 需求评审 | 2026年11月16日 | 15:00开始 | 6 |
| 清算链路 | 技术联调 | 2026年11月17日 | 10:00开始 | 4 |
| 对账异常 | 问题跟进 | 2026年11月18日 | 10:30开始 | 3 |
| 版本V2 | 阶段验收 | 2026年11月19日 | 14:00—16:00内最早可用的连续30分钟 | 6 |

“对账异常”指定持续20分钟，保存到产品部共享日历。时间均为北京时间，前三场开始时间固定；第四场根据日历和会议室实际占用选时。每项只创建一条日程及对应会议室预约，保留已有记录。完成后简要列出各项结果；未完成的说明原因。"""


def inventory(registry, namespace):
    result = data.inventory(registry, namespace)
    db = data.connect(registry, namespace)
    try:
        result["resources"] = resources.snapshot(db)
    finally:
        db.close()
    return result


def check(ref, after):
    basic = dict(ref)
    basic["targets"] = [{k: t[k] for k in booking_rules.KEYS} for t in ref["targets"]]
    result = booking_rules.check(basic, after)
    reservations = {r["event_id"]: r for r in after["resources"]["room_reservations"]}
    for item, target in zip(result["items"], ref["targets"], strict=True):
        row = reservations.get(item["actual_ids"][0]) if len(item["actual_ids"]) == 1 else None
        item["resource_correct"] = bool(
            row and all(row[k] == target[k] for k in ("room_id", "attendees", "kind"))
        )
        item["expected_resource"] = {k: target[k] for k in ("room_id", "attendees", "kind")}
        item["completed"] = item["completed"] and item["resource_correct"]
    unchanged = all(
        after["resources"][k] == ref["resources_before"][k] for k in ("meeting_rooms", "room_busy")
    )
    result.update(
        completed=sum(x["completed"] for x in result["items"]),
        resource_background_unchanged=unchanged,
    )
    result["completion_rate"] = result["completed"] / 4
    result["all_backend_requirements_met"] = (
        result["completed"] == 4
        and result["new_event_count"] == 4
        and result["background_unchanged"]
        and unchanged
        and len(reservations) == 4
    )
    return result
