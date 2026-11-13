"""Administrator fixture and task contract for mail reading/body export only."""

import hashlib
import re
from pathlib import Path

VERSION = "office-mail-read-v1"
CASES = {
    "H": {
        "subject": "关于进行业务网服务器漏洞扫描的通知",
        "sender": "许舟",
        "date": "2026-10-28",
        "scope": "业务网",
        "window": "2026年10月30日20:00至22:00",
        "contact": "周禾",
    },
    "Q": {
        "subject": "关于开展办公网终端安全核查的通知",
        "sender": "赵远",
        "date": "2026-11-03",
        "scope": "办公网",
        "window": "2026年11月5日19:00至21:00",
        "contact": "林澈",
    },
}


def metadata(phase="H"):
    row = CASES[phase]
    return {
        "version": VERSION,
        "phase": phase,
        "subject": row["subject"],
        "sender": row["sender"],
        "output": "content.txt",
    }


def body(phase):
    row = CASES[phase]
    return "\n".join(
        [
            "各位同事：",
            "",
            "为核对办公环境的安全状态，现安排"
            + row["scope"]
            + "相关设备的例行核查。本通知为合成测试数据。",
            "",
            "一、工作时间",
            row["window"] + "。",
            "",
            "二、覆盖范围",
            row["scope"] + "测试环境内已登记的服务器或终端；生产环境和未登记设备另行安排。",
            "",
            "三、配合事项",
            "1. 核对设备登记信息及值班联系方式。",
            "2. 在核查窗口内保持测试设备在线。",
            "3. 遇到业务影响时，先记录现象、设备标识和发生时间，再联系值班人员。",
            "",
            "四、注意事项",
            "本次核查用于记录现状，后续整改另行通知。附件及其他同主题邮件以各自通知范围为准。",
            "",
            "联系人：" + row["contact"] + "（测试协调组）",
            "办公安全服务组",
            row["date"],
        ]
    )


def fixture(recipient):
    rows = []
    snapshots = {}
    for phase, rule in CASES.items():
        prefix = phase.lower()
        ids = []

        def add(key, subject, sender, text, hour=9, *, prefix=prefix, ids=ids, date=rule["date"]):
            mid = prefix + "-" + key
            ids.append(mid)
            rows.append(
                dict(
                    id=mid,
                    subject=subject,
                    from_name=sender,
                    **{"from": "office-notice@example.test"},
                    date=date + f"T{hour:02d}:00:00+08:00",
                    body=text,
                )
            )

        add("target", rule["subject"], rule["sender"], body(phase), 10)
        add(
            "same-subject",
            rule["subject"],
            "周禾" if rule["sender"] != "周禾" else "林澈",
            "这是一封同主题的历史转发，发件人与任务指定人员不同。请以相应通知为准。",
            8,
        )
        for i, suffix in enumerate(["（预告）", "（补充说明）", "：范围征集", "：会议纪要"]):
            add(
                "similar-" + str(i),
                rule["subject"] + suffix,
                rule["sender"],
                "本邮件仅涉及" + suffix.strip("（）：") + "。完整安排以正式通知为准。",
                9,
            )
        for i in range(26):
            add(
                "background-" + str(i),
                ["培训报名提醒", "办公资料归档", "设备登记核对", "团队活动通知"][i % 4]
                + f" {i + 1:02d}",
                ["周禾", "林澈", "赵远", "许舟"][i % 4],
                f"测试背景通知第{i + 1}项：请按各部门日常安排处理相关资料。",
                8,
            )
        snapshots["history" if phase == "H" else "target"] = ids
    return {"fixture_id": VERSION, "recipient": recipient, "messages": rows, "snapshots": snapshots}


def task(config):
    target = config["mail_read"]
    if target.get("version") != VERSION:
        raise ValueError("Mail-read fixture must be prepared first")
    return (
        f"登录后点击顶部导航的“邮件”进入邮箱，在新标签页中打开主题为“{target['subject']}”、发件人为“{target['sender']}”的邮件。"
        "读取邮件正文（正文可能在iframe中），将完整正文保存为本轮output/content.txt。"
        "按主题与发件人共同匹配，保留正文中的编号、时间和签名。完成后按打开顺序逐个关闭本任务的邮箱和登录/工作台标签页，最后给我文件链接。"
    )


def check_export(path, phase="H"):
    p = Path(path)
    actual = p.read_text(encoding="utf-8") if p.is_file() else ""

    def normalize(s):
        return re.sub(r"\s+", "", s)

    expected = body(phase)
    return {
        "file_exists": p.is_file(),
        "filename_correct": p.name == "content.txt",
        "body_complete": bool(actual) and normalize(actual) == normalize(expected),
        "actual_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "comparison": "Ignore whitespace only; require all body text and no extra headers.",
    }
