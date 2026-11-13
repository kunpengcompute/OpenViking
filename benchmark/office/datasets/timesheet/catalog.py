"""Synthetic work cards and workflow metadata for office workload fixtures."""

from datetime import date, timedelta

COLUMNS = ("需求待办", "开发中", "开发自测-todo", "联调验证", "已完成")
CARD_VERSION = "timesheet-board-v2"


def card_specs(anchor):
    day = date.fromisoformat(anchor[:10])
    specs = [
        ("demand-ops", "需求与运营工作", 2, "林澈", "产品运营", "高", "44.5"),
        ("version-maintenance", "版本维护", 3, "许舟", "版本管理", "中", "24"),
        ("training", "团队培训", 4, "周禾", "团队建设", "低", "8"),
        ("demand-review", "需求评审与范围确认", 0, "林澈", "产品运营", "高", "12"),
        ("ops-support", "运营支持工作", 1, "林澈", "产品运营", "中", "16"),
        ("demand-history", "需求与运营工作（归档）", 4, "林澈", "产品运营", "中", "40"),
        ("ops-test", "运营需求验证", 2, "赵远", "质量保障", "高", "20"),
        ("platform-test", "办公平台验证", 2, "许舟", "质量保障", "中", "18"),
        ("report-template", "周报模板维护", 0, "周禾", "办公协作", "低", "6"),
        ("access-review", "权限申请核对", 0, "赵远", "基础服务", "中", "8"),
        ("calendar-sync", "日历同步优化", 1, "许舟", "办公协作", "中", "16"),
        ("mail-index", "邮件检索索引整理", 1, "赵远", "办公协作", "高", "24"),
        ("ticket-fields", "工单字段校验", 3, "周禾", "质量保障", "高", "12"),
        ("release-notes", "发布说明核对", 3, "林澈", "版本管理", "中", "8"),
        ("document-review", "文档目录评审", 0, "许舟", "知识管理", "低", "10"),
        ("meeting-notes", "会议纪要整理", 4, "周禾", "办公协作", "低", "4"),
        ("data-quality", "统计口径核验", 2, "林澈", "质量保障", "高", "20"),
        ("workflow-check", "流程分支回归", 2, "赵远", "质量保障", "中", "18"),
        ("dashboard-export", "统计报表导出", 1, "周禾", "产品运营", "中", "14"),
        ("archive-review", "归档清单复核", 4, "许舟", "知识管理", "低", "6"),
        ("api-contract", "接口契约联调", 3, "赵远", "基础服务", "高", "32"),
        ("equipment", "设备信息登记", 0, "周禾", "基础服务", "低", "8"),
        ("handoff", "上线交接材料", 3, "林澈", "版本管理", "中", "12"),
        ("old-validation", "历史验证记录整理", 4, "赵远", "知识管理", "低", "6"),
    ]
    result = []
    for i, (key, title, column, owner, group, priority, estimate) in enumerate(specs):
        result.append(
            {
                "key": key,
                "title": title,
                "status": COLUMNS[column],
                "owner": owner,
                "group": group,
                "priority": priority,
                "estimate": estimate,
                "code": "WK-" + str(101 + i),
                "start": (day - timedelta(days=21 + i % 7)).isoformat(),
                "due": (day + timedelta(days=(i % 9) - 4)).isoformat(),
                "description": title
                + "：梳理事项范围，核对处理记录，维护相关资料，并完成阶段交接。",
                "version": "办公协作2026.10",
                "project": "办公协作平台",
            }
        )
    # This plan field deliberately differs from already reported time.
    target_due = day.replace(month=9, day=30)
    result[0].update(
        start=(target_due - timedelta(days=21)).isoformat(),
        due=target_due.isoformat(),
        description="跟进需求梳理与运营支持事项，维护需求清单、验证记录和阶段沟通资料。按实际投入登记报工，记录具体工作内容与备注。",
    )
    return result
