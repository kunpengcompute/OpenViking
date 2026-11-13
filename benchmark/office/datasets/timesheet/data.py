"""H/Q business references for the six-card task; task prose lives in tasks/."""

VERSION = "office-timesheet-six-hq-v1"
CARDS = (
    "demand-ops",
    "version-maintenance",
    "training",
    "demand-review",
    "ops-support",
    "data-quality",
)
VALUES = {
    "H": (
        (30, "办公平台的验证测试"),
        (60, "完成版本变更清单核对"),
        (90, "参加办公平台操作培训"),
        (45, "完成本期需求范围评审"),
        (30, "跟进运营反馈并整理处理记录"),
        (30, "核对报工统计口径与汇总数据"),
    ),
    "Q": (
        (45, "整理本周需求变更与核验清单"),
        (30, "复核维护版本与发布记录"),
        (60, "参加操作规范复训"),
        (90, "核对新增需求与范围边界"),
        (45, "整理运营反馈跟进结果"),
        (15, "复核报工汇总与统计口径"),
    ),
}


def goals(phase):
    if phase not in VALUES:
        raise ValueError("phase must be H or Q")
    return [
        {
            "card": card,
            "minutes": minutes,
            "content": content,
            "notes": content,
            "work_date": "2026-11-03" if phase == "H" else "2026-11-04",
        }
        for card, (minutes, content) in zip(CARDS, VALUES[phase], strict=True)
    ]
