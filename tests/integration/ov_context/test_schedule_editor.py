"""Calendar saves retain view state and expose a verifiable persisted result."""

import io
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from benchmark.office.cases.schedule.app import (
    CalendarApp,
)
from benchmark.office.cases.schedule.data import (
    connect,
    provision,
)
from benchmark.office.cases.schedule.editor import (
    calendar_return_url,
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    provision(tmp_path, "editor-test", "2026-10-31T09:00+08:00")
    instance = CalendarApp(tmp_path)
    monkeypatch.setattr(instance, "session", lambda *_: {"csrf": "test-csrf"})
    return instance


def handler(fields=None, *, accept="application/json", path="/calendar/editor-test"):
    body = urlencode(fields or {}).encode()
    responses = []
    headers = {"Content-Length": str(len(body)), "Accept": accept}
    return SimpleNamespace(
        headers=headers,
        path=path,
        rfile=io.BytesIO(body),
        respond=lambda *args: responses.append(args),
        send_response=lambda code: responses.append((code,)),
        send_header=lambda key, value: responses.append((key, value)),
        end_headers=lambda: None,
        responses=responses,
    )


def fields(**updates):
    return {
        "csrf": "test-csrf",
        "nonce": "unique-calendar-save",
        "title": "验收会议",
        "calendar_id": "department",
        "start_date": "2026-11-16",
        "end_date": "2026-11-16",
        "start_time": "15:00",
        "end_time": "15:20",
        "notes": "项目：OV集成验收",
        "return_to": "/calendar/editor-test?month=2026-10&view=list&calendar_filter=1&calendar=personal",
        **updates,
    }


def test_json_save_and_retry_reuse_record_and_retain_calendar_view(app):
    receipts = []
    for _ in range(2):
        h = handler(fields())
        app.handle(h, "POST", "/calendar/editor-test/events")
        assert h.responses[0][0] == 200
        receipts.append(json.loads(h.responses[0][1]))
    assert all(r["saved"] is True and "event_id" not in r for r in receipts)
    receipt = receipts[0]
    assert urlsplit(receipt["return_url"]).path == "/calendar/editor-test"
    assert parse_qs(urlsplit(receipt["return_url"]).query) == {
        "month": ["2026-10"],
        "view": ["list"],
        "calendar_filter": ["1"],
        "calendar": ["personal"],
        "saved": [parse_qs(urlsplit(receipt["return_url"]).query)["saved"][0]],
    }
    assert parse_qs(urlsplit(receipt["return_url"]).query)["saved"][0].startswith("s-")
    db = connect(app.registry, "editor-test")
    try:
        rows = db.execute("SELECT * FROM events WHERE source='agent'").fetchall()
        assert len(rows) == 1
        assert rows[0]["end"] == "2026-11-16T15:20+08:00"
        assert rows[0]["calendar_id"] == "department"
    finally:
        db.close()

    page = handler(path=receipt["return_url"])
    app.handle(page, "GET", "/calendar/editor-test")
    html = page.responses[0][1].decode()
    assert "日程已保存" in html and "查看目标日期" in html
    assert "CAL-" not in html
    assert "month=2026-11" in html and "selected=2026-11-16" in html
    assert "calendar=department" in html
    # Calendar display is a separate observation from loading the detail page.
    db = connect(app.registry, "editor-test")
    try:
        assert not db.execute("SELECT 1 FROM audit WHERE action='detail_read'").fetchall()
    finally:
        db.close()


def test_normal_form_submission_returns_to_calendar(app):
    h = handler(fields(), accept="text/html")
    app.handle(h, "POST", "/calendar/editor-test/events")
    assert h.responses[0] == (303,)
    location = next(value for key, value in h.responses[1:] if key == "Location")
    assert location.startswith("/calendar/editor-test?") and "saved=s-" in location


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/calendar/editor-test",
        "//example.com/calendar/editor-test",
        "/calendar/another-workspace?view=list",
        "/calendar/editor-test/events/1",
    ],
)
def test_return_location_stays_within_same_calendar(target):
    assert calendar_return_url("/calendar/editor-test", target) == "/calendar/editor-test"


def test_return_location_preserves_multiple_filters_and_drops_receipt():
    url = calendar_return_url(
        "/calendar/editor-test",
        "/calendar/editor-test?calendar=personal&calendar=department&created=5&unknown=x",
    )
    assert parse_qs(urlsplit(url).query) == {"calendar": ["personal", "department"]}


@pytest.mark.parametrize("change", [{"csrf": "wrong"}, {"end_time": "14:00"}])
def test_failed_save_never_creates_calendar_row(app, change):
    h = handler(fields(**change))
    app.handle(h, "POST", "/calendar/editor-test/events")
    assert h.responses[0][0] in (403, 422)
    db = connect(app.registry, "editor-test")
    try:
        assert db.execute("SELECT count(*) FROM events WHERE source='agent'").fetchone()[0] == 0
    finally:
        db.close()


def reject_overlaps(app):
    path = app.registry / "editor-test.json"
    config = json.loads(path.read_text())
    config["calendar_policy"] = {"reject_overlaps": True}
    path.write_text(json.dumps(config))


def test_conflict_rejected_then_same_draft_nonce_can_save_at_touching_boundary(app):
    reject_overlaps(app)
    first = handler(fields())
    app.handle(first, "POST", "/calendar/editor-test/events")
    second_fields = fields(nonce="second-calendar-save", start_time="15:10", end_time="15:30")
    blocked = handler(second_fields)
    app.handle(blocked, "POST", "/calendar/editor-test/events")
    assert blocked.responses[0][0] == 409
    feedback = json.loads(blocked.responses[0][1])
    assert feedback["error"] == "calendar_conflict" and len(feedback["events"]) == 1
    assert set(feedback["events"][0]) == {"title", "start", "end"}
    assert "草稿已保留" in feedback["message"]
    retry = handler({**second_fields, "start_time": "15:20", "end_time": "15:50"})
    app.handle(retry, "POST", "/calendar/editor-test/events")
    assert retry.responses[0][0] == 200
    other = handler(fields(nonce="other-calendar-save", calendar_id="personal"))
    app.handle(other, "POST", "/calendar/editor-test/events")
    assert other.responses[0][0] == 200
    db = connect(app.registry, "editor-test")
    try:
        assert db.execute("SELECT count(*) FROM events WHERE source='agent'").fetchone()[0] == 3
        assert (
            db.execute("SELECT count(*) FROM audit WHERE action='conflict_rejected'").fetchone()[0]
            == 1
        )
    finally:
        db.close()


def test_concurrent_occupied_slot_has_only_one_winner(app):
    from concurrent.futures import ThreadPoolExecutor

    reject_overlaps(app)

    def save(n):
        h = handler(fields(nonce=f"concurrent-calendar-{n}"))
        app.handle(h, "POST", "/calendar/editor-test/events")
        return h.responses[0][0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, range(2))) == [200, 409]


def test_occupancy_query_selects_requested_calendar_and_date(app):
    h = handler(fields())
    app.handle(h, "POST", "/calendar/editor-test/events")
    for cal, count in [("department", 1), ("personal", 0)]:
        query = handler(path=f"/calendar/editor-test/availability?date=2026-11-16&calendar={cal}")
        app.handle(query, "GET", "/calendar/editor-test/availability")
        payload = json.loads(query.responses[0][1])
        assert payload["date"] == "2026-11-16" and len(payload["events"]) == count
    invalid = handler(path="/calendar/editor-test/availability?date=wrong&calendar=department")
    app.handle(invalid, "GET", "/calendar/editor-test/availability")
    assert invalid.responses[0][0] == 422


def test_save_receipt_is_bound_to_session_and_expires(app, monkeypatch):
    h = handler(fields())
    app.handle(h, "POST", "/calendar/editor-test/events")
    url = json.loads(h.responses[0][1])["return_url"]
    monkeypatch.setattr(app, "session", lambda *_: {"csrf": "another-session"})
    page = handler(path=url)
    app.handle(page, "GET", "/calendar/editor-test")
    assert "日程已保存" not in page.responses[0][1].decode()
    monkeypatch.setattr(app, "session", lambda *_: {"csrf": "test-csrf"})
    db = connect(app.registry, "editor-test")
    with db:
        db.execute("UPDATE save_receipts SET created=0")
    db.close()
    page = handler(path=url)
    app.handle(page, "GET", "/calendar/editor-test")
    assert "日程已保存" not in page.responses[0][1].decode()


def test_calendar_payload_contains_only_the_displayed_date_range(app):
    import re

    h = handler(fields())
    app.handle(h, "POST", "/calendar/editor-test/events")
    for month, expected in [("2026-10", False), ("2026-11", True)]:
        page = handler(path=f"/calendar/editor-test?month={month}&calendar=department")
        app.handle(page, "GET", "/calendar/editor-test")
        html = page.responses[0][1].decode()
        payload = json.loads(
            re.search(
                r'<script type="application/json" id="calendar-data">(.*?)</script>', html
            ).group(1)
        )
        assert any(e["title"] == "验收会议" for e in payload["events"]) is expected
