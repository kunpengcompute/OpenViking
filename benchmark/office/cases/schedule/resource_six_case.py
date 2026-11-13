"""Six-goal booking with per-item verification and optional deadline scoring."""

import json
from pathlib import Path

from benchmark.office.datasets.schedule.data import EXTRA_RESOURCE_BUSY, EXTRA_TARGETS
from benchmark.office.tasks import render_task

from . import booking_rules, data, resource_case, weekly

VERSION = "schedule-resource-six-v3-explicit-times"


def prepare(registry, namespace, page_url="http://127.0.0.1:19482"):
    ref = resource_case.prepare(registry, namespace, page_url)
    db = data.connect(registry, namespace)
    try:
        with db:
            for room, day, begin, finish in EXTRA_RESOURCE_BUSY:
                db.execute(
                    "INSERT INTO room_busy(room_id,title,start,end) VALUES(?,?,?,?)",
                    (
                        room,
                        "已有资源预约",
                        day + "T" + begin + "+08:00",
                        day + "T" + finish + "+08:00",
                    ),
                )
    finally:
        db.close()
    ref["resources_before"] = resource_case.inventory(registry, namespace)["resources"]
    for subject, kind, day, start, minutes, calendar, prefix, count, room in EXTRA_TARGETS:
        event = weekly.event(prefix + subject, day, start, minutes, calendar, "reference")
        event["notes"] = "项目：OV集成验收；会议类型：" + kind
        target = {key: event[key] for key in booking_rules.KEYS}
        target.update(kind=kind, attendees=count, room_id=room)
        ref["targets"].append(target)
    config_path = Path(registry) / (namespace + ".json")
    config = json.loads(config_path.read_text())
    config["scenario_version"] = VERSION
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    ref.update(
        version=VERSION,
        task=task_description(page_url, namespace),
        training_task=training_task_description(page_url, namespace),
    )
    return ref


def task_description(page_url, namespace):
    return render_task("schedule", "Q", page_url=page_url, workspace=namespace)


def training_task_description(page_url, namespace):
    return render_task("schedule", "H", page_url=page_url, workspace=namespace)


def inventory(registry, namespace):
    return resource_case.inventory(registry, namespace)
