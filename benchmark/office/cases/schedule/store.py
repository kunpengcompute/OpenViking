"""Transactional bookings, target-calendar occupancy and scoped save receipts."""

import hashlib
import secrets
import time

from . import resources
from .data import connect

RECEIPT_TTL = 600


def occupancy(db, calendar, start, end):
    return [
        dict(r)
        for r in db.execute(
            "SELECT title,start,end FROM events WHERE calendar_id=? AND start<? AND end>? ORDER BY start,end,id",
            (calendar, end, start),
        )
    ]


def save_booking(registry, ns, current, values, reject_overlaps):
    db = connect(registry, ns)
    try:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT id FROM events WHERE nonce=?", (values["nonce"],)).fetchone()
        if old:
            event_id = old["id"]
        else:
            conflicts = (
                occupancy(db, values["calendar_id"], values["start"], values["end"])
                if reject_overlaps
                else []
            )
            if conflicts:
                db.execute("INSERT INTO audit(action,event_id) VALUES('conflict_rejected',NULL)")
                db.commit()
                return {"conflicts": conflicts}
            if "room_id" in values:
                error = resources.validate(db, values)
                if error:
                    db.execute(
                        "INSERT INTO audit(action,event_id) VALUES('resource_rejected',NULL)"
                    )
                    db.commit()
                    return {"resource_error": error}
            row = db.execute(
                "INSERT INTO events(owner,calendar_id,title,start,end,notes,source,nonce) VALUES(:owner,:calendar_id,:title,:start,:end,:notes,'agent',:nonce)",
                values,
            )
            event_id = row.lastrowid
            if "room_id" in values:
                db.execute(
                    "INSERT INTO room_reservations VALUES(?,?,?,?)",
                    (event_id, values["room_id"], values["attendees"], values["kind"]),
                )
            db.execute("INSERT INTO audit(action,event_id) VALUES('event_created',?)", (event_id,))
        db.execute(
            "CREATE TABLE IF NOT EXISTS save_receipts(token_hash TEXT PRIMARY KEY,event_id INTEGER NOT NULL,session_hash TEXT NOT NULL,created REAL NOT NULL)"
        )
        db.execute("DELETE FROM save_receipts WHERE created<?", (time.time() - RECEIPT_TTL,))
        token = "s-" + secrets.token_urlsafe(24)
        db.execute(
            "INSERT INTO save_receipts VALUES(?,?,?,?)",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                event_id,
                hashlib.sha256(current["csrf"].encode()).hexdigest(),
                time.time(),
            ),
        )
        db.commit()
        return {"receipt": token}
    finally:
        db.close()


def saved_receipt(registry, ns, current, token):
    if not token.startswith("s-") or len(token) > 100:
        return None
    db = connect(registry, ns)
    try:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='save_receipts'"
        ).fetchone():
            return None
        row = db.execute(
            "SELECT e.* FROM events e JOIN save_receipts r ON e.id=r.event_id WHERE r.token_hash=? AND r.session_hash=? AND r.created>=?",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                hashlib.sha256(current["csrf"].encode()).hexdigest(),
                time.time() - RECEIPT_TTL,
            ),
        ).fetchone()
        return dict(row) if row else None
    finally:
        db.close()
