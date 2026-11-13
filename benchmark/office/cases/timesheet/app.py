"""Scoped task-board routing and transactional work-log mutations."""

import secrets
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from benchmark.office.datasets.timesheet.catalog import card_specs

from ..portal.app import layout
from ..portal.store import connect
from .views import board, detail


def handle(h, portal, ns, current, method, path):
    base = "/timesheet/" + ns
    tail = path[len(base) :].strip("/").split("/")
    config = portal.config(ns)
    cards = {s["key"]: s for s in card_specs(config["anchor"])}
    if method == "POST":
        if len(tail) != 3 or tail[0] != "cards" or tail[1] not in cards or tail[2] != "entries":
            h.respond(404, b"Unknown timesheet route")
            return True
        card = tail[1]
        try:
            fields = portal.auth.form(h)
            if not secrets.compare_digest(fields.get("csrf", ""), current["csrf"]):
                h.respond(403, b"Invalid CSRF token")
                return True
            value = Decimal(fields.get("hours", ""))
            if not value.is_finite() or not Decimal("0") < value <= 24:
                raise ValueError()
            minutes = value * 60
            if minutes != minutes.to_integral_value():
                raise ValueError()
            content = fields.get("content", "").strip()
            notes = fields.get("notes", "").strip()
            nonce = fields.get("nonce", "")
            work_date = fields.get("work_date", config["anchor"][:10])
            if date.fromisoformat(work_date).isoformat() != work_date:
                raise ValueError()
            if (
                not content
                or len(content) > 500
                or len(notes) > 1000
                or not 10 <= len(nonce) <= 100
            ):
                raise ValueError()
        except (ValueError, InvalidOperation):
            h.respond(
                422,
                layout(
                    "报工校验",
                    '<h1>报工未保存</h1><p>请核对日期、时长、内容和备注。</p><button onclick="history.back()">返回修改</button>',
                ),
            )
            return True
        db = connect(portal.registry, ns)
        try:
            with db:
                result = db.execute(
                    "INSERT INTO entries(card,minutes,content,notes,source,nonce,work_date) VALUES(?,?,?,?,?,?,?) ON CONFLICT(nonce) DO NOTHING",
                    (card, int(minutes), content, notes, "agent", nonce, work_date),
                )
                row = db.execute("SELECT * FROM entries WHERE nonce=?", (nonce,)).fetchone()
                if (
                    row["card"],
                    row["minutes"],
                    row["content"],
                    row["notes"],
                    row["work_date"],
                ) != (card, int(minutes), content, notes, work_date):
                    h.respond(409, b"Form was already submitted with different values")
                    return True
                if result.rowcount:
                    db.execute(
                        "INSERT INTO audit(action,entry_id) VALUES(?,?)",
                        ("timesheet_created", row["id"]),
                    )
        finally:
            db.close()
        portal.redirect(h, base + "/cards/" + card + "?saved=1&tab=records")
        return True
    if method != "GET":
        h.respond(405, b"Method not allowed")
        return True
    if path in (base + "/ui.js", base + "/ui.css"):
        filename = "ui." + path.rsplit(".", 1)[1]
        mime = (
            "application/javascript; charset=utf-8"
            if filename.endswith(".js")
            else "text/css; charset=utf-8"
        )
        h.respond(200, Path(__file__).with_name(filename).read_bytes(), mime)
        return True
    if path not in (base, base + "/") and not (
        len(tail) == 2 and tail[0] == "cards" and tail[1] in cards
    ):
        h.respond(404, b"Work card not found")
        return True
    query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
    db = connect(portal.registry, ns)
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM entries ORDER BY id DESC")]
        if path in (base, base + "/"):
            content = board(ns, config["anchor"], rows, query)
        else:
            key = tail[1]
            audit = [
                dict(r)
                for r in db.execute(
                    "SELECT a.created,a.entry_id,e.minutes,e.content FROM audit a JOIN entries e ON a.entry_id=e.id WHERE e.card=? ORDER BY a.id DESC LIMIT 30",
                    (key,),
                )
            ]
            content = detail(
                ns,
                config["anchor"],
                cards[key],
                [r for r in rows if r["card"] == key],
                audit,
                current,
                query,
            )
    finally:
        db.close()
    h.respond(200, content)
    return True
