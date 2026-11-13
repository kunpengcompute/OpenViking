"""Synthetic app catalogue; names and background content are portable test data."""

import os


def webmail_url():
    port = int(os.environ.get("OV_CASE_WEBMAIL_PORT", "19480"))
    return f"http://127.0.0.1:{port}/?_task=mail&_mbox=INBOX"


DISTRACTORS = {
    "knowledge": (
        "知识库",
        "知识文档、操作指引与常见问题",
        "办公知识库已收录通用操作说明。示例文档：文档命名约定、知识分类索引。",
    ),
    "contacts": (
        "通讯录",
        "组织部门、办公联系人与分组",
        "示例部门：产品部、研发部、综合支持部。联系人：许舟、周禾、赵远。",
    ),
    "news": (
        "公告通知",
        "企业公告与办公通知",
        "示例公告：办公系统维护安排、培训通知、资料归档提醒。",
    ),
    "files": ("文件空间", "共享文档、个人文件与目录", "示例目录：公共模板、培训资料、团队文档。"),
    "approvals": (
        "审批中心",
        "流程查询、审批记录与待办",
        "示例流程：设备领用、资料申请、费用审核。当前没有待审批记录。",
    ),
    "employee": ("员工服务", "常见办公服务与帮助", "服务目录：办公设备、园区服务、常见问题。"),
}

# Similar names intentionally require an exact application choice. These are
# synthetic extensions, not reconstructed internal application names or counts.
EXTRA_APPS = (
    ("visitors", "访客预约", "行政服务", "来访登记、访问时段与预约记录"),
    ("purchasing", "采购商城", "行政服务", "办公用品目录、采购申请与订单查询"),
    ("board-history", "电子看板（历史查询）", "研发协作", "查询已归档项目、卡片状态及历史统计"),
    ("board-training", "电子看板（培训）", "研发协作", "培训任务卡片、课程进度与学习记录"),
    ("work-summary", "报工统计", "研发协作", "按团队和日期查看汇总工时及趋势"),
    ("requirements", "需求看板", "研发协作", "需求列表、评审进度与版本计划"),
    ("mail-archive", "邮件归档查询", "办公协作", "查询已归档邮件及保管期限"),
    ("mail-subscriptions", "邮件订阅管理", "办公协作", "订阅栏目、推送设置与发送记录"),
    ("meeting-rooms", "会议室预约", "办公协作", "会议室容量、设备与空闲时段"),
    ("schedule-query", "日程查询", "办公协作", "查询公共活动安排与节假日历"),
    ("ticket-statistics", "工单统计", "研发协作", "工单数量、处理时长与分类统计"),
    ("assets", "资产台账", "行政服务", "设备清单、领用状态与盘点记录"),
    ("learning", "学习中心", "员工成长", "课程目录、报名记录与学习进度"),
    ("surveys", "问卷调研", "员工成长", "活动问卷、提交记录与统计说明"),
)


def applications(namespace):
    prefix = "/office/" + namespace + "/"
    rows = [
        {
            "key": "mail",
            "name": "办公邮箱",
            "category": "办公协作",
            "description": "查找指定邮件、读取正文与文件导出",
            "url": webmail_url(),
            "keywords": "mail email bangongyouxiang",
        },
        {
            "key": "calendar",
            "name": "日程管理",
            "category": "办公协作",
            "description": "个人日历、部门日历与会议安排",
            "url": "/calendar/" + namespace,
            "keywords": "richeng meeting schedule",
        },
        {
            "key": "timesheet",
            "name": "电子看板（ST）",
            "category": "研发协作",
            "description": "工作卡片、报工记录与累计时长",
            "url": "/timesheet/" + namespace,
            "keywords": "dianzikanban timesheet baogong",
        },
    ]
    rows += [
        {
            "key": key,
            "name": name,
            "category": "通用服务",
            "description": desc,
            "url": prefix + key,
            "keywords": key,
        }
        for key, (name, desc, _) in DISTRACTORS.items()
    ]
    for row in rows:
        row.update(subscribed=True, publisher="办公平台服务组")
    rows += [
        {
            "key": key,
            "name": name,
            "category": category,
            "description": desc,
            "url": None,
            "keywords": key,
            "subscribed": False,
            "publisher": "企业应用服务组",
        }
        for key, name, category, desc in EXTRA_APPS
    ]
    return rows
