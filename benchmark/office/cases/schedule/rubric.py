"""Predeclared partial-credit rubric; final-output dimensions require final output."""

VERSION = "schedule-v2-tabs"
CHECKS = {
    "C1": (
        "completion",
        15,
        "已持久化日程1；标题正确5；归属当前用户个人日历6；只新建一个且不改历史记录3。未创建则0。",
    ),
    "C2": (
        "completion",
        20,
        "已创建日程的日期8、开始时分6、结束日期时分4、时区2；逐字段核对任务发起时刻。未创建则0。",
    ),
    "S1": ("structure", 8, "最终回答明确处理状态3、标题2、日期起止时间2、时长1。没有最终回答则0。"),
    "S2": (
        "structure",
        7,
        "最终回答时间表达清楚且标明时区2、起止成对2、有可定位日程编号或详情链接3。无最终回答则0。",
    ),
    "P1": (
        "process",
        8,
        "通过可见网页控件完成日程创建8；只到表单/部分填好4；仅到日程应用2；未到0。脚本或API直接写入不计可见控件完成分。",
    ),
    "P2": (
        "process",
        6,
        "保存后实际读取详情：标题、日期、开始时分、结束时分/时长、归属、保存状态，各1。自动跳转本身不是已读取证据。",
    ),
    "P3": (
        "process",
        6,
        "日程在独立于工作台的新标签中办理2；Agent先用tab close逐个关闭本次业务标签，再关闭只剩最后一页的浏览器4，只关闭部分2，未主动关闭0。CLI不能tab close最后一页，允许此时close收尾；多页直接close及运行时自动清理不计逐个关闭。",
    ),
    "A1": (
        "artifact",
        15,
        "仅评最终回答与持久化结果一致性：归属、标题、日期、开始时间、结束时间/时长，各3；无结果/无最终回答的对象项0。",
    ),
    "A2": (
        "artifact",
        10,
        "最终结果陈述准确5、编号/链接真实3、未完成或不确定事项如实披露2；无最终回答0；没披露任何处理状态不自动给分。",
    ),
    "A3": (
        "artifact",
        5,
        "最终回答紧扣本次安排3；没有虚构参会人、会议室、通知已发等额外业务事实2。无最终回答0。",
    ),
}
assert sum(v[1] for v in CHECKS.values()) == 100


def completion_points(reference, before, after):
    created = [e for e in after["events"] if e["source"] == "agent"]
    if not created:
        return {"C1": 0, "C2": 0}
    e = max(
        created,
        key=lambda r: sum(
            r[k] == reference[k] for k in ["owner", "calendar_id", "title", "start", "end"]
        ),
    )
    preserved = before["background_sha256"] == after["background_sha256"]
    c1 = (
        1
        + 5 * (e["title"] == reference["title"])
        + 6 * (e["owner"] == reference["owner"] and e["calendar_id"] == reference["calendar_id"])
        + 3 * (len(created) == 1 and preserved)
    )
    c2 = (
        8 * (e["start"][:10] == reference["start"][:10])
        + 6 * (e["start"][11:16] == reference["start"][11:16])
        + 4 * (e["end"] == reference["end"])
        + 2 * (e["start"].endswith("+08:00") and e["end"].endswith("+08:00"))
    )
    return {"C1": c1, "C2": c2}


def validate_and_score(judged, completion, final_answer_present, event_ids):
    atoms = judged["checks"]
    expected = set(CHECKS) - {"C1", "C2"}
    if {a["id"] for a in atoms} != expected or len(atoms) != len(expected):
        raise ValueError("Missing or duplicate judge items")
    points = dict(completion)
    evidence = {}
    for atom in atoms:
        key = atom["id"]
        dimension, maximum, _ = CHECKS[key]
        value = atom["points"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= maximum
        ):
            raise ValueError("Invalid points")
        if dimension in ("structure", "artifact") and not final_answer_present and value != 0:
            raise ValueError("No final answer: output dimensions must be zero")
        if not atom.get("reason") or not atom.get("evidence"):
            raise ValueError("Evidence required")
        if any(i not in event_ids for i in atom["evidence"]):
            raise ValueError("Unknown evidence event")
        points[key] = value
        evidence[key] = atom
    dimensions = {
        name: sum(points[k] for k, v in CHECKS.items() if v[0] == name)
        for name in ["completion", "structure", "process", "artifact"]
    }
    return {
        "rubric_version": VERSION,
        "points": points,
        "dimension_points": dimensions,
        "task_score": sum(dimensions.values()),
        "final_answer_present": final_answer_present,
        "judge_evidence": evidence,
    }
