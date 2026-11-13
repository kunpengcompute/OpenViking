"""Resource booking requires correct business attributes and atomic occupancy."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmark.office.cases.schedule import (
    resource_case,
    resources,
)
from benchmark.office.cases.schedule.app import (
    CalendarApp,
)
from tests.integration.ov_context.test_schedule_editor import handler


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    ref = resource_case.prepare(tmp_path, "resource-test")
    app = CalendarApp(tmp_path)
    monkeypatch.setattr(app, "session", lambda *_: {"csrf": "test-csrf"})
    return app, ref


def save(app, target, nonce="resource-save-nonce", **changes):
    fields = {
        "csrf": "test-csrf",
        "nonce": nonce,
        "title": target["title"],
        "calendar_id": target["calendar_id"],
        "start_date": target["start"][:10],
        "end_date": target["end"][:10],
        "start_time": target["start"][11:16],
        "end_time": target["end"][11:16],
        "notes": target["notes"],
        "room_id": target["room_id"],
        "kind": target["kind"],
        "attendees": str(target["attendees"]),
        **changes,
    }
    h = handler(fields, path="/calendar/resource-test/events")
    app.handle(h, "POST", "/calendar/resource-test/events")
    return h.responses[0]


def test_four_independent_goals_include_live_resource_constraints(fixture):
    app, ref = fixture
    assert [t["room_id"] for t in ref["targets"]] == ["R03", "R04", "R06", "R06"]
    assert ref["targets"][3]["start"] == "2026-11-19T15:15+08:00"
    assert not resource_case.check(ref, resource_case.inventory(app.registry, "resource-test"))[
        "completed"
    ]
    for i, target in enumerate(ref["targets"]):
        assert save(app, target, nonce="goal-resource-nonce-" + str(i))[0] == 200
    assert resource_case.check(ref, resource_case.inventory(app.registry, "resource-test"))[
        "all_backend_requirements_met"
    ]
    assert save(app, ref["targets"][0], nonce="goal-resource-nonce-0")[0] == 200
    assert (
        len(
            resource_case.inventory(app.registry, "resource-test")["resources"]["room_reservations"]
        )
        == 4
    )
    page = handler(path="/calendar/resource-test/events/77")
    app.handle(page, "GET", "/calendar/resource-test/events/77")
    assert "星河协作室" in page.responses[0][1].decode()


@pytest.mark.parametrize(
    "updates",
    [
        {"room_id": "R01"},
        {"room_id": "R07"},
        {"room_id": "R08"},
        {"room_id": "R02"},
        {"room_id": "missing"},
    ],
)
def test_inadequate_inaccessible_inactive_or_busy_room_never_creates_event(fixture, updates):
    app, ref = fixture
    assert save(app, ref["targets"][0], **updates)[0] == 409
    after = resource_case.inventory(app.registry, "resource-test")
    assert after["events"] == ref["before"]["events"]
    assert after["resources"]["room_reservations"] == []


def test_full_interval_not_only_start_is_checked(fixture):
    app, ref = fixture
    assert save(app, ref["targets"][1], room_id="R03")[0] == 409
    assert save(app, ref["targets"][1])[0] == 200


def test_query_exposes_filtered_public_data_not_answers(fixture):
    app, ref = fixture
    target = ref["targets"][1]
    result = resources.query(
        app.registry,
        "resource-test",
        {
            "start": target["start"],
            "end": target["end"],
            "calendar": "personal",
            "minimum": "4",
            "equipment": "video",
        },
    )
    rows = {r["id"]: r for r in result["rooms"]}
    assert rows["R03"]["available"] is False and rows["R04"]["available"] is True
    assert rows["R07"]["available"] is False and rows["R08"]["available"] is False
    assert "R01" not in rows and "expected" not in json.dumps(result)
    with pytest.raises(ValueError):
        resources.query(
            app.registry,
            "resource-test",
            {
                "start": target["start"],
                "end": target["end"],
                "calendar": "personal",
                "equipment": "sql",
            },
        )


def test_room_conflict_across_calendars_is_atomic(fixture):
    app, ref = fixture
    target = ref["targets"][0]

    def attempt(calendar):
        return save(app, target, nonce="concurrent-resource-" + calendar, calendar_id=calendar)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(attempt, ["personal", "department"]))
    assert sorted(statuses) == [200, 409]
    after = resource_case.inventory(app.registry, "resource-test")
    assert len(after["resources"]["room_reservations"]) == 1
    assert len(after["events"]) == len(ref["before"]["events"]) + 1


def test_room_rejection_can_retry_same_nonce_and_wrong_priority_fails_scoring(fixture):
    app, ref = fixture
    assert save(app, ref["targets"][0], room_id="R01")[0] == 409
    assert save(app, ref["targets"][0], room_id="R04")[0] == 200
    result = resource_case.check(ref, resource_case.inventory(app.registry, "resource-test"))
    assert result["items"][0]["resource_correct"] is False
    assert result["completed"] == 0


def test_required_fields_and_public_form_contract(fixture):
    app, ref = fixture
    for fields in [{"room_id": ""}, {"kind": "unknown"}, {"attendees": "0"}, {"attendees": "abc"}]:
        assert save(app, ref["targets"][0], **fields)[0] in (409, 422)
    h = handler(path="/calendar/resource-test")
    app.handle(h, "GET", "/calendar/resource-test")
    text = h.responses[0][1].decode()
    assert 'id="query-rooms"' in text and "/resource_ui.js" in text
