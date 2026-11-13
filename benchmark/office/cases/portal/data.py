"""Administrator-only preparation; references stay outside browser-visible pages."""

import json
from pathlib import Path

from ..mail.data import metadata as mail_metadata
from ..schedule.data import provision as calendar_provision
from ..timesheet.data import initialize as initialize_timesheet
from .store import connect


def provision(registry, namespace, projects=None, anchor=None):
    projects = projects or []
    if projects and (
        len(projects) != 2
        or any(
            p["identifier"]
            not in {namespace + "-northstar-platform", namespace + "-northstar-data"}
            for p in projects
        )
    ):
        raise ValueError("Invalid legacy project mapping")
    spec = calendar_provision(registry, namespace, anchor)
    path = Path(registry) / (namespace + ".json")
    data = json.loads(path.read_text())
    data.update(
        case="office",
        projects=projects,
        scenario_version="office-unified-v2",
        mail_read=mail_metadata(),
        employee={"number": "100001", "name": "林澈", "department": "产品部"},
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    db = connect(registry, namespace)
    try:
        db.executescript("""
        CREATE TABLE sessions(token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, created REAL NOT NULL,
                              cookie_names TEXT NOT NULL);
        CREATE TABLE challenges(token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, created REAL NOT NULL);
        """)
    finally:
        db.close()
    initialize_timesheet(registry, namespace)
    return {
        **spec,
        "scenario_version": "office-unified-v2",
        "mail_read": mail_metadata(),
        "login_path": "/mock/login?workspace=" + namespace,
        "employee_number": "100001",
        "timesheet_query": "请在电子看板的“需求与运营工作”卡片报工0.5小时，报工内容与备注均为“办公平台的验证测试”。已有相同内容仍需新增一次。请核对报工前后累计时长，保存验证截图和 before=x, after=y 格式的文本，并告诉我结果。",
    }
