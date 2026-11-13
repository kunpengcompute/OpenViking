"""Freeze eight business results; failure cases retain partial completion."""

from copy import deepcopy

import pytest

from benchmark.office.cases.schedule.weekly import (
    check,
    prepare,
)


def completed_state(reference):
    after = deepcopy(reference["before"])
    next_id = max(e["id"] for e in after["events"]) + 1
    for target in reference["targets"]:
        if target["existing_id"] is None:
            after["events"].append(
                {
                    **{k: v for k, v in target.items() if k != "existing_id"},
                    "id": next_id,
                    "source": "agent",
                    "nonce": f"agent-{next_id}",
                }
            )
            next_id += 1
    return after


def test_fixture_freezes_two_reuses_six_creates_and_conflicts(tmp_path):
    reference = prepare(tmp_path, "weekly-test", "http://127.0.0.1:19482")
    assert len(reference["targets"]) == 8
    assert sum(t["existing_id"] is not None for t in reference["targets"]) == 2
    assert reference["targets"][1]["start"][11:16] == "14:00"
    assert reference["targets"][5]["start"][11:16] == "15:00"
    assert check(reference, reference["before"])["completed"] == 2
    assert check(reference, completed_state(reference))["all_backend_requirements_met"]
    with pytest.raises(FileExistsError):
        prepare(tmp_path, "weekly-test", "http://127.0.0.1:19482")
    assert "weekly-test" in reference["task"]
    assert "existing_id" not in reference["task"]


@pytest.mark.parametrize(
    "mutation", ["wrong_calendar", "wrong_time", "missing_note", "duplicate", "remove_seed"]
)
def test_backend_errors_are_detected(tmp_path, mutation):
    reference = prepare(tmp_path, "weekly-test", "http://127.0.0.1:19482")
    after = completed_state(reference)
    row = next(e for e in after["events"] if e["source"] == "agent")
    if mutation == "wrong_calendar":
        row["calendar_id"] = "personal"
    elif mutation == "wrong_time":
        row["start"] = row["start"].replace("14:00", "10:30")
    elif mutation == "missing_note":
        row["notes"] = ""
    elif mutation == "duplicate":
        after["events"].append({**row, "id": 9999, "nonce": "duplicate"})
    else:
        after["events"].pop(0)
        after["background_sha256"] = "changed"
    result = check(reference, after)
    assert not result["all_backend_requirements_met"]
    if mutation != "remove_seed":
        assert result["completed"] == 7


def test_event_detail_exposes_saved_notes_as_escaped_page_text(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmark.office.cases.schedule.app import (
        CalendarApp,
    )
    from benchmark.office.cases.schedule.data import (
        connect,
    )

    reference = prepare(tmp_path, "notes-detail", "http://127.0.0.1:19482")
    event_id = reference["before"]["events"][0]["id"]
    notes = '官方浏览器验收；前置编号：CAL-42；<script>alert("x")</script>'
    db = connect(tmp_path, "notes-detail")
    with db:
        db.execute("UPDATE events SET notes=? WHERE id=?", (notes, event_id))
    db.close()
    app = CalendarApp(tmp_path)
    monkeypatch.setattr(app, "session", lambda *_: {"authenticated": True})
    replies = []
    handler = SimpleNamespace(respond=lambda status, content: replies.append((status, content)))
    assert app.handle(handler, "GET", f"/calendar/notes-detail/events/{event_id}")
    status, content = replies[0]
    assert status == 200
    page = content.decode()
    assert "备注：" in page and "官方浏览器验收；前置编号：CAL-42；" in page
    assert "&lt;script&gt;" in page and '<script>alert("x")</script>' not in page
    db = connect(tmp_path, "notes-detail")
    assert db.execute("SELECT notes FROM events WHERE id=?", (event_id,)).fetchone()[0] == notes
    db.close()
