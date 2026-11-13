"""Synthetic mailbox: three H preparation targets and six Q evaluation targets."""

import hashlib
import random
from datetime import datetime, timedelta

VERSION = "office-mail-h3-q6-v2"
Q_VERSION = "mail-read-six-v3-20261103"
H_TITLES = (
    "测试终端接入核对通知",
    "办公服务窗口调整安排",
    "账号有效期核验安排",
    "历史版本变更说明",
    "历史轮值协调通知",
    "历史演练检查清单",
)

PEOPLE = {
    "赵远": "zhaoyuan",
    "林澈": "linche",
    "许舟": "xuzhou",
    "周禾": "zhouhe",
    "陈宁": "chenning",
}

TASKS = [
    {
        "id": "Q1",
        "name": "指定邮件",
        "subject": "关于开展办公网终端安全核查的通知",
        "sender": "赵远",
        "date": "2026-11-03T10:00:00+08:00",
    },
    {
        "id": "Q2",
        "name": "区分发件人",
        "subject": "系统维护安排",
        "sender": "林澈",
        "date": "2026-11-02T16:00:00+08:00",
    },
    {
        "id": "Q3",
        "name": "区分日期",
        "subject": "设备登记核对通知",
        "sender": "许舟",
        "date": "2026-11-04T09:30:00+08:00",
    },
    {
        "id": "Q4",
        "name": "最新通知",
        "subject": "办公平台升级通知",
        "sender": "周禾",
        "date": "2026-11-05T18:30:00+08:00",
    },
    {
        "id": "Q5",
        "name": "原件与转发",
        "subject": "关于调整值班安排的通知",
        "sender": "赵远",
        "date": "2026-11-03T14:10:00+08:00",
    },
    {
        "id": "Q6",
        "name": "长正文",
        "subject": "季度安全检查实施细则",
        "sender": "林澈",
        "date": "2026-11-04T15:45:00+08:00",
    },
]


def mail(mid, subject, sender, date, body):
    return {
        "id": mid,
        "subject": subject,
        "from_name": sender,
        "from": PEOPLE[sender] + "@example.test",
        "date": date,
        "body": body,
    }


def text_for(t):
    i = int(t["id"][1:])
    day = t["date"][:10]
    if i == 6:
        sections = []
        for n in range(1, 33):
            sections.append(
                f"{n}、第{n:02d}项检查："
                + [
                    "资产登记与台账核对",
                    "终端补丁与版本核查",
                    "账号授权和有效期核查",
                    "数据备份与恢复演练",
                ][n % 4]
                + "\n"
                f"检查范围为办公测试环境第{n:02d}组设备，设备编号SEC-{n:03d}至SEC-{n + 2:03d}；生产设备不在本通知范围。"
                f"执行人员须对照登记台账检查设备归属、版本记录、最近一次维护时间及审批编号，将结果登记在季度检查清单第{n:02d}行。"
                f"发现缺失信息时记录“待补充”，不得将空值填写为0，也不得推断未提供的维护结果。"
                f"本项的取证编号为EV-26-{n:03d}，复核人应核对原始记录与截图中的设备编号一致。"
                f"如与历史登记存在差异，应保留原登记值和本次核验值，并说明差异来源。"
                f"本项完成期限为2026年11月{6 + n % 5}日17:00；超出期限的情况单独登记，不覆盖已完成记录。"
            )
        return "\n\n".join(
            [
                "各位同事：",
                "以下为本季度办公测试环境安全检查实施细则，请按条款落实并保留原始记录。",
                *sections,
                "附录A：检查结果代码\nPASS=已核对一致；GAP=存在缺项；PENDING=尚未核实。三类状态分别登记，不能合并。",
                "附录B：材料清单\n1. 资产登记表；2. 版本核查记录；3. 账号授权清单；4. 备份恢复记录。",
                "补充说明：本细则不授权执行生产变更。涉及生产设备的操作另行审批。",
                "联系人：林澈（安全协调组）",
                "办公安全服务组",
                day,
            ]
        )
    scope = ["办公网终端", "数据服务平台", "设备登记台账", "办公平台版本4.8", "周三夜间值班"][i - 1]
    return "\n".join(
        [
            "各位同事：",
            "",
            f"现对{scope}安排如下，请以本封通知中的范围和时间执行。",
            "",
            "一、工作时间",
            f"2026年11月{5 + i}日19:00至21:00。",
            "",
            "二、覆盖范围",
            scope + "的已登记测试资源，未登记设备及生产环境另行审批。",
            "",
            "三、配合事项",
            "1. 核对登记信息和联系方式。",
            f"2. 在安排的时间内完成第{i}组核验并保留原始记录。",
            "3. 发现业务影响时，记录设备标识、现象和发生时间后联系值班人员。",
            "",
            "四、核验要求",
            f"本次登记编号为OFFICE-2026-{110 + i}；后续调整将以新的正式通知发布。",
            "附件及其他同主题邮件按各自通知范围处理，不替代本次安排。",
            "",
            f"联系人：{t['sender']}（办公协调组）",
            "办公运行服务组",
            day,
        ]
    )


def pool():
    rows = [
        mail(t["id"].lower() + "-original", t["subject"], t["sender"], t["date"], text_for(t))
        for t in TASKS
    ]

    def add(mid, subject, sender, date, body):
        rows.append(mail(mid, subject, sender, date, body))

    for i, t in enumerate(TASKS, 1):
        add(
            f"q{i}-other-sender",
            t["subject"],
            "陈宁",
            "2026-11-05T20:00:00+08:00",
            "各位同事：\n本邮件为区域服务组的安排，适用范围为区域测试环境。请核对所在区域后处理。\n联系人：陈宁\n2026年11月5日",
        )
        add(
            f"q{i}-preview",
            t["subject"] + "（预告）",
            t["sender"],
            "2026-11-01T09:00:00+08:00",
            "各位同事：\n相关安排仍在汇总中，具体时间待正式通知。本邮件用于征求参与名单。\n办公运行服务组\n2026年11月1日",
        )
    for day in ["2026-11-03", "2026-11-05"]:
        add(
            "q3-" + day,
            "设备登记核对通知",
            "许舟",
            day + "T09:30:00+08:00",
            f"各位同事：\n本次设备登记核对涉及研发测试区，登记批次为{day}，请完成该批次的台账核对。\n许舟\n"
            + day,
        )
    for date, version in [
        ("2026-11-01T18:30:00+08:00", "4.6"),
        ("2026-11-04T19:00:00+08:00", "4.7"),
    ]:
        add(
            "q4-v" + version,
            "办公平台升级通知",
            "周禾",
            date,
            f"各位同事：\n办公平台计划升级至版本{version}，升级窗口以本次安排为准。完成后核对登录与消息通知。\n周禾\n"
            + date[:10],
        )
    original = rows[4]["body"]
    add(
        "q5-forward",
        "转发：关于调整值班安排的通知",
        "陈宁",
        "2026-11-05T11:00:00+08:00",
        "请区域团队补充值班联系方式。\n----------转发的邮件----------\n发件人：赵远\n主题：关于调整值班安排的通知\n"
        + original,
    )
    add(
        "q5-reply",
        "回复：关于调整值班安排的通知",
        "赵远",
        "2026-11-04T10:10:00+08:00",
        "已收到名单，实际替班人员请按确认表登记。\n> " + original.replace("\n", "\n> "),
    )
    for n in range(40):
        subject = [
            "培训报名提醒",
            "办公资料归档",
            "设备登记核对",
            "团队活动通知",
            "月度值班汇总",
            "办公平台使用建议",
            "安全检查问题反馈",
            "项目会议纪要",
        ][n % 8] + f" 第{n + 1}批"
        add(
            f"background-{n + 1:02d}",
            subject,
            list(PEOPLE)[n % 5],
            f"2026-11-{1 + n % 5:02d}T{8 + n % 10:02d}:15:00+08:00",
            f"各位同事：\n请核对第{n + 1}批日常资料，保留原始记录，处理后按部门安排归档。\n办公协作组",
        )
    for row in rows:
        row["_admin_key"] = row["id"]
        row["id"] = hashlib.sha256((Q_VERSION + ":" + row["id"]).encode()).hexdigest()[:28]
    random.Random(1733).shuffle(rows)
    return rows


def phase_rows(phase):
    if phase not in {"H", "Q"}:
        raise ValueError("phase must be H or Q")
    rows = pool()
    if phase == "Q":
        return rows
    for index, row in enumerate(rows, 1):
        for target, title in zip(TASKS, H_TITLES, strict=True):
            row["subject"] = row["subject"].replace(target["subject"], title)
        row["id"] = hashlib.sha256(("history:" + row["id"]).encode()).hexdigest()[:28]
        row["date"] = (datetime.fromisoformat(row["date"]) - timedelta(days=14)).isoformat()
        row["body"] = (
            f"各位同事：\n\n本次仅核对历史演练批次 H-{index:03d}，范围为历史测试区。\n"
            f"请核对第{index}组登记编号、负责人及复核状态。\n"
            "一、核对登记编号是否完整。\n二、保留复核结果原始记录。\n"
            "三、遇到缺项登记为待补充，不代填未知值。\n\n"
            f"历史协调组\n{row['date'][:10]}"
        )
        if row["_admin_key"] == "q6-original":
            sections = [
                f"{n}、历史演练第{n:02d}项：核对H-{n:03d}设备登记、演练日志、"
                f"恢复验证及责任人签认，保留原始值与本次核验值。记录编号HIST-{n:03d}。"
                "缺项填写待补充，复核结论分别记录一致、差异和待核验；每项保留完整依据。"
                for n in range(1, 33)
            ]
            row["body"] += "\n\n" + "\n\n".join(sections)
            row["body"] += (
                "\n\n附录A：演练材料清单\n1.设备台账；2.历史日志；3.核验记录。\n历史协调组\n2026年10月21日"
            )
    random.Random(7349).shuffle(rows)
    return rows


def fixture(recipient, phase):
    rows = phase_rows(phase)
    key = "history" if phase == "H" else "target"
    return {
        "fixture_id": VERSION + "-" + phase.lower(),
        "recipient": recipient,
        "messages": rows,
        "snapshots": {key: [row["id"] for row in rows]},
    }


def goals(phase):
    rows = phase_rows(phase)
    selected = (1, 2, 4) if phase == "H" else range(1, 7)
    return [
        {
            "id": f"{phase}{i}",
            "filename": f"{'history' if phase == 'H' else 'mail'}_{i:02d}.txt",
            "message": next(row for row in rows if row["_admin_key"] == f"q{target}-original"),
        }
        for i, target in enumerate(selected, 1)
    ]
