"""Outlook-inspired presentation of the isolated synthetic calendar."""

import calendar
import html
import json
import re
from datetime import date, datetime
from urllib.parse import urlencode

from ..portal.catalog import webmail_url

UI_VERSION = "calendar-outlook-v2"
CALENDARS = {
    "personal": ("我的日历", "blue"),
    "department": ("产品部共享日历", "green"),
    "rooms": ("会议室使用安排", "purple"),
}


def E(value):
    return html.escape(str(value), quote=True)


def icon(name):
    paths = {
        "search": "M21 21l-5-5M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16",
        "calendar": "M4 5h16v16H4zM7 2v6M17 2v6M4 11h16",
        "mail": "M3 5h18v14H3zM3 5l9 7 9-7",
        "people": "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8M18 7a3 3 0 0 1 0 6M22 21v-2a4 4 0 0 0-3-4",
        "grid": "M3 3h6v6H3zM15 3h6v6h-6zM3 15h6v6H3zM15 15h6v6h-6z",
        "chevron-left": "M15 5l-7 7 7 7",
        "chevron-right": "M9 5l7 7-7 7",
        "close": "M5 5l14 14M19 5L5 19",
    }
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="'
        + paths[name]
        + '"/></svg>'
    )


def events_on(rows, day):
    result = []
    for row in rows:
        start, end = datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"])
        if start.date() <= day <= end.date() and not (
            end.date() == day and end.hour == end.minute == 0
        ):
            result.append(row)
    return result


def render_calendar(ns, base, data, rows, selected, query):
    month = query.get("month", [data["anchor"][:7]])[0]
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        month = data["anchor"][:7]
    year, number = map(int, month.split("-"))
    weeks = calendar.Calendar(0).monthdatescalendar(year, number)
    view = query.get("view", ["month"])[0]
    if view not in ["month", "list"]:
        view = "month"
    search = query.get("q", [""])[0][:200]
    today = date.fromisoformat(data["anchor"][:10])
    chosen = query.get(
        "selected", [today.isoformat() if today.strftime("%Y-%m") == month else month + "-01"]
    )[0]
    try:
        chosen = date.fromisoformat(chosen)
    except ValueError:
        chosen = date(year, number, 1)

    def href(**updates):
        values = {
            "month": month,
            "view": view,
            "q": search,
            "calendar_filter": "1",
            "calendar": selected,
        }
        values.update(updates)
        return E(base + "?" + urlencode(values, doseq=True))

    def hidden_fields(include_search=True):
        values = {"month": month, "view": view, "calendar_filter": "1"}
        if include_search:
            values["q"] = search
        return "".join(
            '<input type="hidden" name="' + key + '" value="' + E(value) + '">'
            for key, value in values.items()
        )

    previous = (year - 1, 12) if number == 1 else (year, number - 1)
    following = (year + 1, 1) if number == 12 else (year, number + 1)
    visible = [
        r
        for r in rows
        if events_on([r], date(year, number, 1))
        or r["start"].startswith(month)
        or r["end"].startswith(month)
    ]
    mini = (
        '<table class="mini-calendar" aria-label="日期导航"><thead><tr>'
        + "".join("<th>" + d + "</th>" for d in ["一", "二", "三", "四", "五", "六", "日"])
        + "</tr></thead><tbody>"
    )
    for week in weeks:
        mini += "<tr>"
        for day in week:
            classes = (
                (" outside" if day.month != number else "")
                + (" today" if day == today else "")
                + (" selected" if day == chosen else "")
            )
            mini += (
                '<td><a class="mini-day'
                + classes
                + '" href="'
                + href(month=day.strftime("%Y-%m"), selected=day.isoformat())
                + '" aria-label="选择'
                + day.isoformat()
                + '">'
                + str(day.day)
                + "</a></td>"
            )
        mini += "</tr>"
    mini += "</tbody></table>"
    filters = '<form class="calendar-filter" method="get">' + hidden_fields()
    for key, (title, color) in CALENDARS.items():
        filters += (
            '<label class="calendar-choice"><input type="checkbox" name="calendar" value="'
            + key
            + '" data-autosubmit'
            + (" checked" if key in selected else "")
            + '><span class="calendar-dot '
            + color
            + '"></span><span>'
            + title
            + "</span></label>"
        )
    filters += "<noscript><button>应用日历筛选</button></noscript></form>"
    grid = (
        '<div class="month-board" role="grid" aria-label="'
        + month
        + ' 月视图" style="--week-count:'
        + str(len(weeks))
        + '"><div class="weekday-row" role="row">'
        + "".join(
            '<div role="columnheader">' + d + "</div>"
            for d in ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        )
        + "</div>"
    )
    for week in weeks:
        grid += '<div class="month-week" role="row">'
        for day in week:
            classes = (
                (" outside" if day.month != number else "")
                + (" is-today" if day == today else "")
                + (" is-selected" if day == chosen else "")
            )
            grid += (
                '<div class="month-cell'
                + classes
                + '" role="gridcell" data-day="'
                + day.isoformat()
                + '"><button class="day-number" type="button" data-select-day="'
                + day.isoformat()
                + '" aria-label="查看'
                + day.isoformat()
                + '的日程">'
                + str(day.day)
                + "</button>"
            )
            events = events_on(rows, day)
            for event in events[:3]:
                title, color = CALENDARS[event["calendar_id"]]
                start = event["start"][11:16]
                label = event["title"] + "，" + event["start"] + "，" + title + "，查看日程详情"
                grid += (
                    '<a class="event-chip '
                    + color
                    + '" href="'
                    + base
                    + "/events/"
                    + str(event["id"])
                    + '" aria-label="'
                    + E(label)
                    + '"><span>'
                    + E(start)
                    + "</span> "
                    + E(event["title"])
                    + "</a>"
                )
            if len(events) > 3:
                grid += (
                    '<button class="more-events" type="button" data-select-day="'
                    + day.isoformat()
                    + '">还有'
                    + str(len(events) - 3)
                    + "项</button>"
                )
            grid += "</div>"
        grid += "</div>"
    grid += "</div>"
    agenda = '<div class="agenda-list" aria-label="日程列表">'
    last_day = None
    for event in visible:
        day = date.fromisoformat(event["start"][:10])
        if day != last_day:
            agenda += (
                '<div class="agenda-date"><b>'
                + str(day.day)
                + "</b><span>"
                + str(day.month)
                + "月 · "
                + ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][
                    day.weekday()
                ]
                + "</span></div>"
            )
            last_day = day
        title, color = CALENDARS[event["calendar_id"]]
        agenda += (
            '<a class="agenda-item '
            + color
            + '" href="'
            + base
            + "/events/"
            + str(event["id"])
            + '"><span class="agenda-time">'
            + E(event["start"][11:16])
            + " – "
            + E(event["end"][11:16])
            + '</span><span class="agenda-subject"><strong>'
            + E(event["title"])
            + "</strong><small>"
            + E(title)
            + '</small></span><span class="agenda-open">查看详情 ›</span></a>'
        )
    if not visible:
        agenda += '<div class="calendar-empty">本月没有匹配的日程</div>'
    agenda += "</div>"
    # Sidebar day details use exactly the date range rendered by this view.
    public_rows = (
        visible
        if view == "list"
        else [
            r
            for r in rows
            if r["start"][:10] <= weeks[-1][-1].isoformat()
            and r["end"][:10] >= weeks[0][0].isoformat()
        ]
    )
    public = [
        {k: r[k] for k in ["id", "title", "start", "end", "calendar_id"]} for r in public_rows
    ]
    payload = json.dumps(
        {"base": base, "selected": chosen.isoformat(), "events": public}, ensure_ascii=False
    ).replace("<", "\\u003c")
    mail_url = E(webmail_url()) if data.get("case") == "office" else "/office-info/" + ns + "/mail"
    contacts_url = (
        "/office/" + ns + "/contacts"
        if data.get("case") == "office"
        else "/office-info/" + ns + "/contacts"
    )
    body = (
        f'<link rel="stylesheet" href="{base}/ui.css"><div class="outlook-shell" data-page-version="{UI_VERSION}">'
        '<nav class="app-rail" aria-label="办公应用">'
        f'<a href="/workbench/{ns}" title="返回工作台" aria-label="返回工作台">'
        + icon("grid")
        + "</a>"
        f'<a href="{mail_url}" title="办公邮箱" aria-label="办公邮箱">' + icon("mail") + "</a>"
        '<a class="rail-active" aria-current="page" href="'
        + href()
        + '" title="日历" aria-label="日历">'
        + icon("calendar")
        + "</a>"
        f'<a href="{contacts_url}" title="通讯录" aria-label="通讯录">'
        + icon("people")
        + "</a></nav>"
        '<aside class="calendar-sidebar"><div class="sidebar-heading"><h1>日历</h1><span class="user-avatar">林</span><button type="button" class="sidebar-close" id="close-calendars" aria-label="关闭日历导航">'
        + icon("close")
        + "</button></div>"
        '<div class="mini-title"><strong>'
        + str(year)
        + "年"
        + str(number)
        + '月</strong><div><a href="'
        + href(month="%04d-%02d" % previous)
        + '" aria-label="迷你日历上月">'
        + icon("chevron-left")
        + '</a><a href="'
        + href(month="%04d-%02d" % following)
        + '" aria-label="迷你日历下月">'
        + icon("chevron-right")
        + "</a></div></div>"
        + mini
        + '<h2 class="calendar-group-title">我的日历</h2>'
        + filters
        + '<p class="timezone-label">Asia/Shanghai</p></aside>'
        '<div class="calendar-workspace"><div class="calendar-commandbar" role="toolbar" aria-label="日历操作"><button type="button" class="calendar-nav-toggle" id="toggle-calendars" aria-label="切换日历导航" aria-expanded="false">☰</button>'
        '<button class="primary new-event" id="new-event" type="button"><span aria-hidden="true">＋</span> 新建日程</button>'
        '<a class="today-button" href="'
        + href(month=today.strftime("%Y-%m"), selected=today.isoformat())
        + '">今天</a>'
        '<a class="icon-button" href="'
        + href(month="%04d-%02d" % previous)
        + '" aria-label="上月">'
        + icon("chevron-left")
        + "</a>"
        '<a class="icon-button" href="'
        + href(month="%04d-%02d" % following)
        + '" aria-label="下月">'
        + icon("chevron-right")
        + "</a>"
        '<strong class="month-title">' + str(year) + "年" + str(number) + "月</strong>"
        '<div class="view-switch" role="group" aria-label="日历视图"><a href="'
        + href(view="month")
        + '"'
        + (' aria-current="page"' if view == "month" else "")
        + '>月视图</a><a href="'
        + href(view="list")
        + '"'
        + (' aria-current="page"' if view == "list" else "")
        + ">日程列表</a></div>"
        '<button class="my-day-button" id="show-day" type="button">我的一天</button>'
        '<details class="calendar-menu"><summary aria-label="更多日历操作">⋯</summary><div><button id="export" type="button">导出日历</button><button id="settings" type="button">日历设置</button></div></details></div>'
        '<div class="calendar-subbar"><form class="calendar-search" method="get">'
        + hidden_fields(False)
        + "".join('<input type="hidden" name="calendar" value="' + c + '">' for c in selected)
        + '<label for="q">'
        + icon("search")
        + '<span class="sr-only">搜索日程</span></label><input id="q" name="q" type="search" value="'
        + E(search)
        + '" placeholder="搜索日程"><button type="submit">搜索</button></form>'
        '<span class="clock-note">办公时间：<time>' + E(data["anchor"]) + "</time></span></div>"
        '<div class="calendar-canvas">'
        + (grid if view == "month" else agenda)
        + '<aside class="day-pane" id="day-pane" hidden><div class="day-pane-title"><h2 id="day-heading"></h2><button id="close-day" aria-label="关闭我的一天" type="button">'
        + icon("close")
        + '</button></div><div id="day-events"></div><button class="day-new" id="day-new" type="button">在所选日期新建日程</button></aside></div>'
        '<div class="calendar-status"><span>'
        + str(len(visible))
        + "项日程 · "
        + ("月视图" if view == "month" else "日程列表")
        + "</span><span>林澈 · 产品部</span></div></div>"
        '<script type="application/json" id="calendar-data">' + payload + "</script></div>"
    )
    return body
