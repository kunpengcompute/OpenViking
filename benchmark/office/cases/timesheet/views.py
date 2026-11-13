"""Task board/detail rendering from scoped fixture data and persisted work logs."""

from decimal import Decimal
from urllib.parse import urlencode

from benchmark.office.datasets.timesheet.catalog import CARD_VERSION, COLUMNS, card_specs

from ..portal.app import esc, layout


def hours(minutes):
    value = Decimal(minutes) / 60
    return format(value, "f").rstrip("0").rstrip(".") if minutes % 60 else str(minutes // 60)


def chrome(namespace, body, view="board"):
    base = "/timesheet/" + namespace
    nav = '<aside class="work-nav"><h2>电子看板（ST）</h2>'
    for title, url, selected in [
        ("任务看板", base, view == "board"),
        ("我的工作", base + "?owner=" + esc("林澈"), view == "mine"),
        ("全部任务", base + "?view=list", view == "list"),
        ("报工台账", base + "?view=ledger", view == "ledger"),
        ("返回工作台", "/workbench/" + namespace, False),
    ]:
        nav += (
            '<a href="'
            + url
            + '"'
            + (' aria-current="page"' if selected else "")
            + ">"
            + title
            + "</a>"
        )
    nav += "<p>产品部 · 办公协作平台<br>当前版本：2026.10</p></aside>"
    return layout(
        "任务面板 · 产品部",
        '<link rel="stylesheet" href="'
        + base
        + '/ui.css"><div class="work-app" data-page-version="'
        + CARD_VERSION
        + '">'
        + nav
        + '<div class="work-main">'
        + body
        + "</div></div>",
    )


def filters(base, query, view):
    body = (
        '<form class="work-toolbar" method="get"><input type="hidden" name="view" value="'
        + view
        + '">'
    )
    body += (
        '<div><label for="work-search">搜索任务</label><input id="work-search" name="q" placeholder="任务名称或编号" value="'
        + esc(query.get("q", [""])[0])
        + '"></div>'
    )
    for field, label, values in [
        ("status", "状态", COLUMNS),
        ("owner", "经办人", ("林澈", "许舟", "周禾", "赵远")),
    ]:
        value = query.get(field, [""])[0]
        body += (
            '<div><label for="work-'
            + field
            + '">'
            + label
            + '</label><select id="work-'
            + field
            + '" name="'
            + field
            + '"><option value="">全部</option>'
        )
        body += (
            "".join(
                '<option value="'
                + esc(x)
                + '"'
                + (" selected" if x == value else "")
                + ">"
                + esc(x)
                + "</option>"
                for x in values
            )
            + "</select></div>"
        )
    return (
        body
        + '<button class="primary">筛选</button><a href="'
        + base
        + "?view="
        + view
        + '">重置筛选</a></form>'
    )


def board(namespace, anchor, entries, query):
    base = "/timesheet/" + namespace
    view = query.get("view", ["board"])[0]
    if view not in ("board", "list", "ledger"):
        view = "board"
    q = query.get("q", [""])[0][:200].strip().lower()
    owner = query.get("owner", [""])[0]
    status = query.get("status", [""])[0]
    specs = card_specs(anchor)
    selected = [
        s
        for s in specs
        if (not q or q in (s["title"] + " " + s["code"]).lower())
        and (not owner or owner == s["owner"])
        and (not status or status == s["status"])
    ]

    def total(key):
        return hours(sum(r["minutes"] for r in entries if r["card"] == key))

    head = (
        '<div class="work-top"><div><h1>任务面板 · 产品部</h1><p>办公协作平台 / 2026.10 · '
        + str(len(selected))
        + ' 项任务</p></div><div class="work-switch">'
    )
    for key, title in [("board", "看板视图"), ("list", "列表视图")]:
        values = {k: v[0] for k, v in query.items() if k in ("q", "owner", "status")}
        values["view"] = key
        head += (
            '<a href="'
            + base
            + "?"
            + esc(urlencode(values))
            + '"'
            + (' aria-current="page"' if view == key else "")
            + ">"
            + title
            + "</a>"
        )
    body = head + "</div></div>" + filters(base, query, view)
    if view == "ledger":
        visible = {s["key"] for s in selected}
        logs = [r for r in entries if r["card"] in visible]
        names = {s["key"]: s["title"] for s in specs}
        body += (
            '<section class="work-panel"><h2>报工台账</h2><p>筛选范围累计已报工 '
            + hours(sum(r["minutes"] for r in logs))
            + " 小时 · "
            + str(len(logs))
            + ' 条</p><table class="work-records"><thead><tr><th>任务</th><th>报工日期</th><th>内容</th><th>时长（小时）</th><th>我的备注</th></tr></thead><tbody>'
        )
        for row in logs:
            body += (
                '<tr><td><a href="'
                + base
                + "/cards/"
                + row["card"]
                + '?tab=records">'
                + esc(names[row["card"]])
                + "</a></td><td>"
                + esc(row.get("work_date") or anchor[:10])
                + "</td><td>"
                + esc(row["content"])
                + "</td><td>"
                + hours(row["minutes"])
                + "</td><td>"
                + esc(row["notes"])
                + "</td></tr>"
            )
        body += "</tbody></table></section>"
    elif view == "list":
        body += '<section class="work-panel"><h2>全部任务</h2><table class="work-records"><thead><tr><th>编号 / 标题</th><th>状态</th><th>经办人</th><th>计划完成</th><th>预计工时</th><th>累计已报工时</th></tr></thead><tbody>'
        for s in selected:
            body += (
                '<tr><td><a href="'
                + base
                + "/cards/"
                + s["key"]
                + '">'
                + esc(s["code"] + " " + s["title"])
                + "</a></td><td>"
                + esc(s["status"])
                + "</td><td>"
                + s["owner"]
                + "</td><td>"
                + s["due"]
                + "</td><td>"
                + s["estimate"]
                + "</td><td>"
                + total(s["key"])
                + "</td></tr>"
            )
        body += "</tbody></table></section>"
    else:
        body += '<div class="work-board" aria-label="任务流程看板">'
        for column in COLUMNS:
            cards = [s for s in selected if s["status"] == column]
            body += (
                '<section class="work-column" aria-label="'
                + column
                + '"><h2>'
                + column
                + " <span>"
                + str(len(cards))
                + "</span></h2>"
            )
            for s in cards:
                body += (
                    '<article class="work-card"><span class="work-tag">'
                    + s["group"]
                    + '</span> <span class="work-priority">'
                    + s["priority"]
                    + '优先级</span><h3><a href="'
                    + base
                    + "/cards/"
                    + s["key"]
                    + '">'
                    + esc(s["title"])
                    + "</a></h3>"
                )
                body += (
                    "<p>"
                    + s["code"]
                    + " · "
                    + s["version"]
                    + "</p><p>计划完成 "
                    + s["due"]
                    + "</p><p>预计工时 "
                    + s["estimate"]
                    + " 小时 · 已报 "
                    + total(s["key"])
                    + ' 小时</p><div class="work-foot"><span>'
                    + s["owner"]
                    + '</span><a class="work-details-link" href="'
                    + base
                    + "/cards/"
                    + s["key"]
                    + '">查看工作卡片</a></div></article>'
                )
            body += "</section>"
        body += "</div>"
    if not selected:
        body += '<p class="work-empty" role="status">当前筛选条件下没有任务。</p>'
    return chrome(namespace, body, "mine" if owner == "林澈" and view == "board" else view)


def detail(namespace, anchor, spec, entries, audit, current, query):
    base = "/timesheet/" + namespace
    path = base + "/cards/" + spec["key"]
    saved = query.get("saved", [""])[0] == "1"
    tab = query.get("tab", ["records" if saved else "overview"])[0]
    if tab not in ("overview", "records", "activity"):
        tab = "overview"
    total = sum(r["minutes"] for r in entries)
    body = (
        '<nav class="work-crumb"><a href="'
        + base
        + '">任务看板</a><span> / '
        + spec["status"]
        + " / "
        + spec["code"]
        + "</span></nav>"
    )
    body += (
        '<div class="work-detail-title"><span class="work-tag">任务</span><h1>'
        + esc(spec["title"])
        + '</h1><span class="pill">'
        + spec["status"]
        + "</span></div>"
    )
    body += (
        '<div class="work-summary"><div><p>经办人</p><strong>'
        + spec["owner"]
        + "</strong></div><div><p>计划完成日期</p><strong>"
        + spec["due"]
        + "</strong></div><div><p>预计工时（小时）</p><strong>"
        + spec["estimate"]
        + "</strong></div><div><p>累计已报工时（小时）</p><strong>"
        + hours(total)
        + "</strong></div></div>"
    )
    if saved:
        body += '<p class="work-notice" role="status">报工已保存，请核对新增记录和累计时长。</p>'
    body += '<nav role="tablist" aria-label="工作卡片内容" class="tabs work-panels">'
    for key, title in [("overview", "卡片详情"), ("records", "报工记录"), ("activity", "活动记录")]:
        body += (
            '<button type="button" role="tab" id="tab-'
            + key
            + '" data-work-tab aria-selected="'
            + str(tab == key).lower()
            + '" aria-controls="work-'
            + key
            + '">'
            + title
            + "</button>"
        )
    body += "</nav>"

    def panel(key):
        return (
            '<section id="work-'
            + key
            + '" role="tabpanel" aria-labelledby="tab-'
            + key
            + '" class="panel work-panel"'
            + (" hidden" if tab != key else "")
            + ">"
        )

    body += panel("overview") + '<h2>卡片详情</h2><dl class="work-fields">'
    for label, value in [
        ("工作名称", spec["title"]),
        ("任务编号", spec["code"]),
        ("所属项目", spec["project"]),
        ("所属版本", spec["version"]),
        ("状态", spec["status"]),
        ("优先级", spec["priority"]),
        ("经办人", spec["owner"]),
        ("创建人", "周禾"),
        ("业务分类", spec["group"]),
        ("计划开始日期", spec["start"]),
        ("计划完成日期", spec["due"]),
        ("预计工时（小时）", spec["estimate"]),
    ]:
        body += "<div><dt>" + label + "</dt><dd>" + esc(value) + "</dd></div>"
    body += (
        "</dl><h2>任务描述</h2><p>"
        + esc(spec["description"])
        + "</p><h2>工作范围</h2><ul><li>事项清单、处理记录与资料核对</li><li>跨团队沟通及阶段进度维护</li><li>验证结果归档与交接说明</li></ul></section>"
    )
    body += (
        panel("records")
        + '<div class="work-record-header"><div><h2>报工记录</h2><p>累计报工时长：<strong id="total-hours">'
        + hours(total)
        + "</strong> 小时 · 共 "
        + str(len(entries))
        + ' 条记录</p></div><button id="add-work-entry" class="primary" type="button">添加报工</button></div>'
    )
    needle = query.get("entry_q", [""])[0][:200]
    shown = [r for r in entries if needle.lower() in (r["content"] + " " + r["notes"]).lower()]
    pages = max(1, (len(shown) + 7) // 8)
    try:
        page = max(1, min(pages, int(query.get("page", ["1"])[0])))
    except ValueError:
        page = 1
    body += (
        '<form method="get" class="work-toolbar"><input type="hidden" name="tab" value="records"><label for="entry-search">搜索报工记录</label><input id="entry-search" name="entry_q" placeholder="报工内容或备注" value="'
        + esc(needle)
        + '"><button>查询</button><a href="'
        + path
        + '?tab=records">全部记录</a></form>'
    )
    body += '<table class="work-records"><thead><tr><th>编号</th><th>报工日期</th><th>报工人</th><th>报工内容</th><th>时长（小时）</th><th>我的备注</th></tr></thead><tbody>'
    for row in shown[(page - 1) * 8 : page * 8]:
        body += (
            "<tr><td>TS-"
            + str(row["id"])
            + "</td><td>"
            + esc(row.get("work_date") or anchor[:10])
            + "</td><td>林澈</td><td>"
            + esc(row["content"])
            + "</td><td>"
            + hours(row["minutes"])
            + "</td><td>"
            + esc(row["notes"])
            + "</td></tr>"
        )
    body += "</tbody></table>"
    if not shown:
        body += '<p class="work-empty">没有匹配的报工记录。</p>'
    body += '<nav class="work-pager" aria-label="报工记录分页">'
    for step, title in [(-1, "上一页"), (1, "下一页")]:
        target = page + step
        if 1 <= target <= pages:
            body += (
                '<a href="'
                + path
                + "?"
                + esc(urlencode({"tab": "records", "page": target, "entry_q": needle}))
                + '">'
                + title
                + "</a>"
            )
    body += (
        "<span>第 "
        + str(page)
        + " / "
        + str(pages)
        + " 页 · 匹配 "
        + str(len(shown))
        + " 条</span></nav></section>"
    )
    body += (
        panel("activity")
        + '<h2>活动记录</h2><ul class="work-audit"><li><time>'
        + spec["start"]
        + "</time><p>周禾创建任务，登记所属版本与预计工时 "
        + spec["estimate"]
        + " 小时。</p></li><li><time>"
        + spec["start"]
        + "</time><p>任务流转至 "
        + spec["status"]
        + "，经办人 "
        + spec["owner"]
        + "。</p></li>"
    )
    for event in audit:
        body += (
            "<li><time>"
            + esc(event["created"])
            + "</time><p>新增报工 TS-"
            + str(event["entry_id"])
            + "，"
            + hours(event["minutes"])
            + " 小时："
            + esc(event["content"])
            + "</p></li>"
        )
    body += "</ul></section>"
    body += (
        '<dialog id="add-entry" aria-labelledby="entry-title"><h2 id="entry-title">添加报工</h2><p class="hint">'
        + spec["code"]
        + " · "
        + esc(spec["title"])
        + '</p><form method="post" action="'
        + path
        + '/entries">'
    )
    import secrets

    body += (
        '<input type="hidden" name="csrf" value="'
        + esc(current["csrf"])
        + '"><input type="hidden" name="nonce" value="'
        + secrets.token_urlsafe(20)
        + '">'
    )
    body += (
        '<div class="formgrid"><div><label for="work-date">报工日期 <span class="work-required">*</span></label><input type="date" id="work-date" name="work_date" value="'
        + anchor[:10]
        + '" required></div><div><label for="hours">报工时长（小时） <span class="work-required">*</span></label><input id="hours" name="hours" type="number" min="0.01" max="24" step="0.01" required></div></div>'
    )
    body += '<label for="content">报工内容 <span class="work-required">*</span></label><input id="content" name="content" maxlength="500" required><label for="notes">我的备注</label><textarea id="notes" name="notes" maxlength="1000"></textarea>'
    body += (
        '<p><button class="primary">确定</button> <button type="button" data-cancel-entry>取消</button></p></form></dialog><script src="'
        + base
        + '/ui.js"></script>'
    )
    return chrome(namespace, body)
