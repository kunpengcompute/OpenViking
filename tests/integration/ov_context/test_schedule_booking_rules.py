"""Same-name candidates must not count as the new booking or the earliest slot."""

from copy import deepcopy

import pytest

from benchmark.office.cases.schedule.booking_rules import (
    check,
    prepare,
)


def complete(reference):
    after = deepcopy(reference["before"])
    first = max(e["id"] for e in after["events"]) + 1
    after["events"] += [
        dict(t, id=first + i, source="agent", nonce=f"actual-{i}")
        for i, t in enumerate(reference["targets"])
    ]
    return after


def test_fixture_requires_new_composite_matches_and_earliest_continuous_slot(tmp_path):
    ref = prepare(tmp_path, "booking-rules-test")
    assert len(ref["before"]["events"]) == 76
    assert check(ref, ref["before"])["completed"] == 0
    assert all(
        sum(e["title"] == t["title"] for e in ref["before"]["events"]) == 2 for t in ref["targets"]
    )
    assert ref["targets"][3]["start"] == "2026-11-19T15:00+08:00"
    assert ref["targets"][3]["end"] == "2026-11-19T15:30+08:00"
    assert check(ref, complete(ref))["all_backend_requirements_met"]
    with pytest.raises(FileExistsError):
        prepare(tmp_path, "booking-rules-test")


@pytest.mark.parametrize(
    "kind",
    ["short_gap", "later_slot", "wrong_calendar", "wrong_note", "duplicate", "remove_background"],
)
def test_partial_or_duplicate_outcomes_are_retained_as_failures(tmp_path, kind):
    ref = prepare(tmp_path, "booking-rules-test")
    after = complete(ref)
    row = after["events"][-1]
    if kind == "short_gap":
        row.update(start="2026-11-19T14:30+08:00", end="2026-11-19T15:00+08:00")
    elif kind == "later_slot":
        row.update(start="2026-11-19T15:30+08:00", end="2026-11-19T16:00+08:00")
    elif kind == "wrong_calendar":
        row["calendar_id"] = "personal"
    elif kind == "wrong_note":
        row["notes"] = "wrong"
    elif kind == "duplicate":
        after["events"].append(dict(row, id=999, nonce="duplicate"))
    else:
        after["events"][0]["title"] = "changed"
    result = check(ref, after)
    assert not result["all_backend_requirements_met"]
    if kind != "remove_background":
        assert result["completed"] == 3
