"""Synthetic meeting resources: public filters and transactional reservations."""

from datetime import datetime, timedelta
from html import escape

from benchmark.office.datasets.schedule.data import MEETING_KINDS as KINDS

from .data import connect


def initialize(db):
    db.executescript("""
    CREATE TABLE meeting_rooms(id TEXT PRIMARY KEY,name TEXT NOT NULL,capacity INTEGER NOT NULL,
      projector INTEGER NOT NULL,video INTEGER NOT NULL,department TEXT NOT NULL,
      active INTEGER NOT NULL,priority INTEGER NOT NULL);
    CREATE TABLE room_busy(id INTEGER PRIMARY KEY,room_id TEXT NOT NULL REFERENCES meeting_rooms(id),
      title TEXT NOT NULL,start TEXT NOT NULL,end TEXT NOT NULL);
    CREATE TABLE room_reservations(event_id INTEGER PRIMARY KEY REFERENCES events(id),
      room_id TEXT NOT NULL REFERENCES meeting_rooms(id),attendees INTEGER NOT NULL,kind TEXT NOT NULL);
    """)


def snapshot(db):
    return {
        name: [dict(r) for r in db.execute(f"SELECT * FROM {name} ORDER BY 1")]
        for name in ("meeting_rooms", "room_busy", "room_reservations")
    }


def eligible(room, kind, attendees):
    return bool(
        room["active"]
        and room["department"] in ("public", "product")
        and room["capacity"] >= attendees
        and kind in KINDS
        and (room["department"] == "product" if KINDS[kind] == "department" else room[KINDS[kind]])
    )


def busy_rows(db, room_id, start, end):
    return [
        dict(r)
        for r in db.execute(
            """
      SELECT title,start,end FROM room_busy WHERE room_id=? AND start<? AND end>?
      UNION ALL
      SELECT e.title,e.start,e.end FROM room_reservations r JOIN events e ON e.id=r.event_id
      WHERE r.room_id=? AND e.start<? AND e.end>? ORDER BY start,end
      """,
            (room_id, end, start, room_id, end, start),
        )
    ]


def validate(db, values):
    room = db.execute("SELECT * FROM meeting_rooms WHERE id=?", (values["room_id"],)).fetchone()
    if not room or not eligible(dict(room), values["kind"], values["attendees"]):
        return "会议室的容量、设备、归属或启用状态不符合本次会议要求。"
    if busy_rows(db, values["room_id"], values["start"], values["end"]):
        return "该会议室在本次完整时段内已有占用，请重新查询并选择资源或调整时间。"
    return None


def query(registry, ns, params):
    start = datetime.fromisoformat(params["start"])
    end = datetime.fromisoformat(params["end"])
    if start.utcoffset() != timedelta(hours=8) or end.utcoffset() != timedelta(hours=8):
        raise ValueError("Use Beijing time")
    minimum = int(params.get("minimum", "0"))
    equipment = params.get("equipment", "any")
    scope = params.get("scope", "any")
    calendar = params["calendar"]
    if not 0 < (end - start).total_seconds() <= 86400 or not 0 <= minimum <= 100:
        raise ValueError("Invalid interval or capacity")
    if (
        equipment not in ("any", "projector", "video")
        or scope not in ("any", "public", "product")
        or calendar not in ("personal", "department")
    ):
        raise ValueError("Invalid filter")
    day_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    db = connect(registry, ns)
    try:
        rows = [dict(r) for r in db.execute("SELECT * FROM meeting_rooms ORDER BY priority,id")]
        rows = [
            r
            for r in rows
            if r["capacity"] >= minimum
            and (equipment == "any" or r[equipment])
            and (scope == "any" or r["department"] == scope)
        ]
        for r in rows:
            r["busy"] = busy_rows(
                db,
                r["id"],
                day_start.isoformat(timespec="minutes"),
                day_end.isoformat(timespec="minutes"),
            )
            r["available"] = bool(
                r["active"]
                and r["department"] in ("public", "product")
                and not any(
                    x["start"] < end.isoformat(timespec="minutes")
                    and x["end"] > start.isoformat(timespec="minutes")
                    for x in r["busy"]
                )
            )
        calendars = [
            dict(r)
            for r in db.execute(
                "SELECT title,start,end FROM events WHERE calendar_id=? AND start<? AND end>? ORDER BY start,id",
                (
                    calendar,
                    day_end.isoformat(timespec="minutes"),
                    day_start.isoformat(timespec="minutes"),
                ),
            )
        ]
        return {
            "date": start.date().isoformat(),
            "start": start.isoformat(timespec="minutes"),
            "end": end.isoformat(timespec="minutes"),
            "rooms": rows,
            "calendar_busy": calendars,
        }
    finally:
        db.close()


def fields():
    options = "".join(f'<option value="{escape(k)}">{escape(k)}</option>' for k in KINDS)
    return (
        '<fieldset id="resource-booking"><legend>会议室预约</legend>'
        '<label for="meeting-kind">会议类型</label><select id="meeting-kind" name="kind" required><option value="">请选择</option>'
        + options
        + "</select>"
        '<label for="attendees">参会人数</label><input type="number" min="1" max="100" id="attendees" name="attendees" required>'
        '<div class="grid"><div><label for="room-minimum">最小容量</label><input id="room-minimum" type="number" min="0" max="100" value="0"></div>'
        '<div><label for="room-equipment">设备筛选</label><select id="room-equipment"><option value="any">不限</option><option value="projector">投屏</option><option value="video">视频会议</option></select></div></div>'
        '<label for="room-scope">资源归属</label><select id="room-scope"><option value="any">全部归属</option><option value="public">公共会议室</option><option value="product">产品部专属</option></select>'
        '<button type="button" id="query-rooms">查询会议室与日历占用</button>'
        '<p id="room-status" role="status">填写起止日期时间后查询，选择整个时段可用且符合规则的会议室。</p><div id="room-results"></div></fieldset>'
    )
