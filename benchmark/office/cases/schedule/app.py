"""Calendar page adapter for the existing loopback office page layer.

Namespaces are isolated administrator-issued test capabilities. The workbench
bootstraps a synthetic SSO session; no user passwords or external SSO are used.
All event mutations require that session plus a CSRF token and persist in SQLite.
"""

import hashlib
import html
import json
import secrets
import time
from datetime import datetime, timedelta
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import resources
from .data import NAMESPACE, ZONE, connect
from .editor import calendar_return_url, datetime_field, with_query
from .store import occupancy, save_booking, saved_receipt


def escape(value):
    return html.escape(str(value), quote=True)


def layout(title, body):
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>" + escape(title) + "</title><style>"
        "body{font:16px system-ui;margin:0;background:#f3f6fb;color:#21314a}"
        "header{background:#183b61;color:white;padding:18px 28px;display:flex;justify-content:space-between}"
        "main{max-width:1150px;margin:24px auto;padding:0 20px}a{color:#15569b}"
        "section,article,form{background:white;border:1px solid #dce3ec;border-radius:8px;padding:20px;margin:16px 0}"
        "button,input,select,textarea{font:inherit;padding:9px;border:1px solid #b8c8da;border-radius:4px}"
        "button{cursor:pointer;background:#edf4fc}button.primary{background:#18599b;color:white}"
        "label{display:block;margin:12px 0 5px}nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #dce3ec;padding:10px;text-align:left}"
        ".muted{color:#64748b}.notice{background:#fff8dc;padding:12px}.error{color:#a21e28}"
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:16px}"
        "dialog{border:1px solid #c0ccdb;border-radius:10px;width:540px;max-width:90vw}"
        "</style><header><strong>办公平台 · 日程管理</strong><span>林澈 · 产品部</span>"
        "</header><main>" + body + "</main></html>"
    ).encode()


class CalendarApp:
    def __init__(self, registry):
        self.registry = Path(registry)

    def config(self, namespace):
        if not NAMESPACE.fullmatch(namespace):
            raise ValueError("Invalid namespace")
        data = json.loads((self.registry / (namespace + ".json")).read_text())
        if data.get("case") not in ("schedule", "office") or data.get("namespace") != namespace:
            raise ValueError("Unknown calendar")
        return data

    def cookie_name(self, ns):
        return "office_schedule_" + hashlib.sha256(ns.encode()).hexdigest()[:16]

    def session(self, handler, ns):
        try:
            jar = SimpleCookie()
            jar.load(handler.headers.get("Cookie", ""))
            token = jar[self.cookie_name(ns)].value
        except (KeyError, ValueError):
            return None
        db = connect(self.registry, ns)
        try:
            row = db.execute(
                "SELECT * FROM sessions WHERE token_hash=? AND created>?",
                (hashlib.sha256(token.encode()).hexdigest(), time.time() - 3600),
            ).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def bootstrap(self, handler, ns, content):
        """Existing office tests are pre-authenticated; give each case its own SSO cookie."""
        self.config(ns)
        if self.session(handler, ns):
            return handler.respond(200, content)
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        db = connect(self.registry, ns)
        try:
            with db:
                db.execute(
                    "INSERT INTO sessions VALUES(?,?,?)",
                    (hashlib.sha256(token.encode()).hexdigest(), csrf, time.time()),
                )
        finally:
            db.close()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header(
            "Set-Cookie",
            f"{self.cookie_name(ns)}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=3600",
        )
        handler.send_header("Content-Length", str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)

    def redirect(self, h, target):
        h.send_response(303)
        h.send_header("Location", target)
        h.send_header("Content-Length", "0")
        h.end_headers()

    def handle(self, h, method, path):
        pieces = path.strip("/").split("/")
        if not pieces or pieces[0] != "calendar":
            return False
        if len(pieces) < 2:
            return h.respond(404, b"Not found") or True
        ns = pieces[1]
        try:
            data = self.config(ns)
        except (ValueError, OSError, KeyError):
            return h.respond(404, b"Not found") or True
        base = "/calendar/" + ns
        current = self.session(h, ns)
        if not current and method == "POST":
            h.respond(403, b"Calendar session required")
            return True
        if not current:
            self.redirect(h, "/workbench/" + ns)
            return True
        if method == "POST":
            if path != base + "/events":
                return h.respond(404, b"Not found") or True
            return self.create(h, ns, current) or True
        if method != "GET":
            return h.respond(405, b"Method not allowed") or True
        if path == base + "/room-options" and data.get("resource_booking"):
            try:
                params = {k: v[0] for k, v in parse_qs(urlsplit(h.path).query).items()}
                payload = resources.query(self.registry, ns, params)
            except (ValueError, KeyError, OverflowError):
                return h.respond(422, b"Invalid resource query") or True
            return (
                h.respond(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json")
                or True
            )
        if path == base + "/availability":
            query = parse_qs(urlsplit(h.path).query)
            calendar = query.get("calendar", [""])[0]
            try:
                day = datetime.strptime(query.get("date", [""])[0], "%Y-%m-%d").replace(tzinfo=ZONE)
                if calendar not in ("personal", "department"):
                    raise ValueError()
            except ValueError:
                return h.respond(422, b"Valid date and calendar required") or True
            db = connect(self.registry, ns)
            try:
                rows = occupancy(
                    db,
                    calendar,
                    day.isoformat(timespec="minutes"),
                    (day + timedelta(days=1)).isoformat(timespec="minutes"),
                )
            finally:
                db.close()
            payload = {
                "date": day.date().isoformat(),
                "calendar_label": "我的日历" if calendar == "personal" else "产品部共享日历",
                "events": rows,
            }
            if "application/json" in h.headers.get("Accept", ""):
                h.respond(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json")
            else:
                content = (
                    "<h1>日程占用</h1><p>"
                    + escape(payload["date"] + " · " + payload["calendar_label"])
                    + "</p>"
                )
                content += (
                    "".join(
                        "<p>" + escape(r["start"] + " — " + r["end"] + " · " + r["title"]) + "</p>"
                        for r in rows
                    )
                    if rows
                    else "<p>当天无占用。</p>"
                )
                h.respond(200, layout("日程占用", content))
            return True
        if path in (base, base + "/"):
            from .views import render_calendar

            query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
            requested = query.get("calendar", [] if "calendar_filter" in query else ["personal"])
            selected = [key for key in ("personal", "department", "rooms") if key in requested]
            search = query.get("q", [""])[0][:200]
            db = connect(self.registry, ns)
            try:
                placeholders = ",".join("?" for _ in selected)
                rows = (
                    [
                        dict(r)
                        for r in db.execute(
                            "SELECT * FROM events WHERE calendar_id IN ("
                            + placeholders
                            + ") AND title LIKE ? ORDER BY start,id",
                            (*selected, "%" + search + "%"),
                        )
                    ]
                    if selected
                    else []
                )
            finally:
                db.close()
            notice = ""
            token = query.get("saved", [""])[0]
            if token:
                saved = saved_receipt(self.registry, ns, current, token)
                if saved:
                    target = with_query(
                        base,
                        month=saved["start"][:7],
                        selected=saved["start"][:10],
                        view=query.get("view", ["month"])[0],
                        q="",
                        calendar_filter="1",
                        calendar=list(dict.fromkeys([*selected, saved["calendar_id"]])),
                    )
                    notice = (
                        '<div class="saved-notice" role="status">日程已保存，请在日历中核验本次记录。 <a href="'
                        + escape(target)
                        + '">查看目标日期</a></div>'
                    )
            body = (
                notice
                + render_calendar(ns, base, data, rows, selected, query)
                + self.form(base, data, current, calendar_return_url(base, h.path))
            )
            body += '<dialog id="info"><h2 id="info-title"></h2><p id="info-body"></p><button onclick="this.closest(\'dialog\').close()">关闭</button></dialog>'
            body += f'<script src="{base}/ui.js"></script>'
            if data.get("resource_booking"):
                body += f'<script src="{base}/resource_ui.js"></script>'
            h.respond(200, layout("日程管理", body))
            return True
        if path in (base + "/ui.js", base + "/ui.css", base + "/resource_ui.js"):
            name = path.rsplit("/", 1)[-1]
            content_type = (
                "application/javascript; charset=utf-8"
                if name.endswith(".js")
                else "text/css; charset=utf-8"
            )
            h.respond(200, Path(__file__).with_name(name).read_bytes(), content_type)
            return True
        tail = pieces[2:]
        if len(tail) == 2 and tail[0] == "events" and tail[1].isdigit():
            db = connect(self.registry, ns)
            try:
                event = db.execute("SELECT * FROM events WHERE id=?", (int(tail[1]),)).fetchone()
                if event:
                    with db:
                        db.execute(
                            "INSERT INTO audit(action,event_id) VALUES(?,?)",
                            ("detail_read", event["id"]),
                        )
            finally:
                db.close()
            if not event:
                return h.respond(404, b"Event not found") or True
            minutes = int(
                (
                    datetime.fromisoformat(event["end"]) - datetime.fromisoformat(event["start"])
                ).total_seconds()
                / 60
            )
            fields = [
                ("日程编号", "CAL-" + str(event["id"])),
                ("标题", event["title"]),
                (
                    "归属账号",
                    "林澈（产品部）" if event["owner"] == "linche-product" else "产品部团队",
                ),
                (
                    "日历",
                    {"personal": "我的日历", "department": "部门共享日历", "rooms": "会议室安排"}[
                        event["calendar_id"]
                    ],
                ),
                ("开始时间", event["start"]),
                ("结束时间", event["end"]),
                ("持续时间", str(minutes) + "分钟"),
                ("时区", "Asia/Shanghai"),
                ("状态", "已保存"),
                ("备注", event["notes"]),
            ]
            if data.get("resource_booking"):
                db = connect(self.registry, ns)
                try:
                    reservation = db.execute(
                        "SELECT r.*,m.name FROM room_reservations r JOIN meeting_rooms m ON m.id=r.room_id WHERE r.event_id=?",
                        (event["id"],),
                    ).fetchone()
                finally:
                    db.close()
                if reservation:
                    fields.extend(
                        [
                            ("会议室", reservation["name"]),
                            ("参会人数", str(reservation["attendees"])),
                            ("会议类型", reservation["kind"]),
                            ("会议室预约", "已生效"),
                        ]
                    )
            body = (
                "<h1>日程详情</h1><section>"
                + "".join(f"<p><strong>{escape(k)}：</strong>{escape(v)}</p>" for k, v in fields)
                + "</section>"
                + f'<a href="{base}">返回日程列表</a>'
            )
            h.respond(200, layout("日程详情", body))
            return True
        h.respond(404, b"Not found")
        return True

    def form(self, base, data, current, return_to=None):
        day = datetime.fromisoformat(data["anchor"]).date().isoformat()
        nonce = secrets.token_urlsafe(20)
        return (
            f'<dialog id="editor"><h2>新建日程</h2><form method="post" action="{base}/events">'
            f'<input type="hidden" name="csrf" value="{escape(current["csrf"])}">'
            f'<input type="hidden" name="nonce" value="{nonce}">'
            f'<input type="hidden" name="return_to" value="{escape(return_to or base)}">'
            '<label for="title">日程标题</label><input id="title" name="title" required maxlength="160">'
            '<label for="destination">保存到日历</label><select id="destination" name="calendar_id">'
            '<option value="personal">我的日历 · 林澈（产品部）</option><option value="department">产品部共享日历</option></select>'
            '<div class="grid"><div>'
            + datetime_field("start_date", "开始日期", day, "date")
            + datetime_field("start_time", "开始时间", "", "time")
            + "</div><div>"
            + datetime_field("end_date", "结束日期", day, "date")
            + datetime_field("end_time", "结束时间", "", "time")
            + "</div></div>"
            '<p><button type="button" id="check-availability">查询整日占用</button></p>'
            '<section id="availability-panel" aria-label="日程占用" hidden></section>'
            + (resources.fields() if data.get("resource_booking") else "")
            + '<label for="notes">备注</label><textarea id="notes" name="notes" maxlength="1000"></textarea>'
            "<p>直接填写起止日期和时间，格式为 YYYY-MM-DD、HH:mm。</p>"
            '<p id="editor-status" role="status" aria-live="polite">时区：Asia/Shanghai。</p>'
            '<p><button class="primary">保存日程</button> <button type="button" id="cancel-event">取消</button></p></form></dialog>'
        )

    def create(self, h, ns, current):
        origin = h.headers.get("Origin")
        if origin and urlsplit(origin).netloc != h.headers.get("Host"):
            return h.respond(403, b"Cross-origin mutation denied")
        length = int(h.headers.get("Content-Length", "0"))
        if not 0 < length <= 12000:
            return h.respond(413, b"Invalid form size")
        fields = {
            k: v[0]
            for k, v in parse_qs(h.rfile.read(length).decode(), keep_blank_values=True).items()
        }
        if not secrets.compare_digest(fields.get("csrf", ""), current["csrf"]):
            return h.respond(403, b"Invalid CSRF token")
        try:
            title = fields["title"].strip()
            calendar = fields["calendar_id"]
            nonce = fields["nonce"]
            if (
                not title
                or len(title) > 160
                or calendar not in ("personal", "department")
                or not 10 <= len(nonce) <= 100
            ):
                raise ValueError()
            start = datetime.strptime(
                fields["start_date"] + " " + fields["start_time"], "%Y-%m-%d %H:%M"
            ).replace(tzinfo=ZONE)
            end = datetime.strptime(
                fields["end_date"] + " " + fields["end_time"], "%Y-%m-%d %H:%M"
            ).replace(tzinfo=ZONE)
            duration = (end - start).total_seconds() / 60
            if not 0 < duration <= 1440:
                raise ValueError()
        except (KeyError, ValueError):
            return h.respond(
                422,
                layout(
                    "日程校验未通过",
                    '<h1 class="error">日程未保存</h1><p>请填写有效标题、日历与起止日期时间，结束须晚于开始且不超过24小时。</p><button onclick="history.back()">返回修改</button>',
                ),
            )
        policy = self.config(ns).get("calendar_policy", {})
        room_values = {}
        if self.config(ns).get("resource_booking"):
            try:
                room_values = {
                    "room_id": fields["room_id"],
                    "kind": fields["kind"],
                    "attendees": int(fields["attendees"]),
                }
                if (
                    room_values["kind"] not in resources.KINDS
                    or not 1 <= room_values["attendees"] <= 100
                ):
                    raise ValueError()
            except (KeyError, ValueError):
                return h.respond(422, b"Valid meeting kind, attendees and room required")
        result = save_booking(
            self.registry,
            ns,
            current,
            {
                "owner": "linche-product" if calendar == "personal" else "department-team",
                "calendar_id": calendar,
                "title": title,
                "start": start.isoformat(timespec="minutes"),
                "end": end.isoformat(timespec="minutes"),
                "notes": fields.get("notes", "")[:1000],
                "nonce": nonce,
                **room_values,
            },
            policy.get("reject_overlaps", False),
        )
        if "resource_error" in result:
            payload = {
                "error": "resource_unavailable",
                "resource": True,
                "message": result["resource_error"],
            }
            return h.respond(
                409, json.dumps(payload, ensure_ascii=False).encode(), "application/json"
            )
        if "conflicts" in result:
            payload = {
                "error": "calendar_conflict",
                "message": "所选时间与目标日历已有安排重叠，草稿已保留。请调整时段或查询整日占用。",
                "date": start.date().isoformat(),
                "calendar_label": "我的日历" if calendar == "personal" else "产品部共享日历",
                "events": result["conflicts"],
            }
            if "application/json" in h.headers.get("Accept", ""):
                h.respond(409, json.dumps(payload, ensure_ascii=False).encode(), "application/json")
            else:
                h.respond(
                    409,
                    layout(
                        "日程时间冲突",
                        "<h1>日程未保存</h1><p>"
                        + escape(payload["message"])
                        + '</p><button onclick="history.back()">返回修改</button>',
                    ),
                )
            return
        base = "/calendar/" + ns
        destination = with_query(
            calendar_return_url(base, fields.get("return_to", base)), saved=result["receipt"]
        )
        if "application/json" in h.headers.get("Accept", ""):
            h.respond(
                200,
                json.dumps({"saved": True, "return_url": destination}).encode(),
                "application/json",
            )
        else:
            self.redirect(h, destination)
