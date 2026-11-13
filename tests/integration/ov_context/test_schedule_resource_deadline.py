"""Business completion at a frozen cutoff, including boundary and late writes."""

from copy import deepcopy

import pytest

from benchmark.office.cases.schedule import (
    booking_rules,
    resources,
)
from benchmark.office.cases.schedule import (
    resource_six_case as case,
)
from benchmark.office.suites.schedule import scoring


def successful_state(tmp_path):
    ref = case.prepare(tmp_path, "six-score")
    after = case.inventory(tmp_path, "six-score")
    first = max(row["id"] for row in after["events"]) + 1
    for i, target in enumerate(ref["targets"]):
        after["events"].append(
            {
                **{k: target[k] for k in booking_rules.KEYS},
                "id": first + i,
                "source": "agent",
                "nonce": str(i),
            }
        )
        after["resources"]["room_reservations"].append(
            {"event_id": first + i, **{k: target[k] for k in ["room_id", "kind", "attendees"]}}
        )
    return ref, after, first


def test_six_goals_and_new_resource_constraints(tmp_path):
    ref, after, _first = successful_state(tmp_path)
    assert len(ref["before"]["events"]) == 76
    assert len(ref["resources_before"]["room_busy"]) == 10
    assert [t["room_id"] for t in ref["targets"]] == ["R03", "R04", "R06", "R06", "R03", "R04"]
    assert "以下六场" in ref["task"] and "灾备切换方案" in ref["task"] and "通知链路" in ref["task"]
    for target in ref["targets"][4:]:
        query = resources.query(
            tmp_path,
            "six-score",
            {
                "start": target["start"],
                "end": target["end"],
                "calendar": target["calendar_id"],
                "minimum": str(target["attendees"]),
                "equipment": "video" if target["kind"] == "技术联调" else "projector",
            },
        )
        eligible = [r for r in query["rooms"] if r["available"]]
        assert min(eligible, key=lambda r: (r["priority"], r["id"]))["id"] == target["room_id"]
    assert scoring.check(ref, after)["all_backend_requirements_met"]
    bad = deepcopy(after)
    bad["resources"]["room_reservations"][-1]["room_id"] = "R03"
    assert scoring.check(ref, bad)["completed"] == 5


def test_deadline_inclusive_and_late_or_unstamped_goals_excluded(tmp_path):
    ref, after, first = successful_state(tmp_path)
    writes = [
        {"event_id": first + i, "created": 100 + seconds}
        for i, seconds in enumerate([10, 100, 599, 600, 600.001])
    ]
    result = scoring.check_at_deadline(ref, after, writes, started_at=100, budget_seconds=600)
    assert result["completed"] == 4 and result["total"] == 6
    assert result["completion_rate"] == 4 / 6
    assert len(after["events"]) == 82  # Scoring preserves the supplied observation.
    assert scoring.check(ref, after)["completed"] == 6


def test_pre_deadline_snapshot_cannot_gain_later_repairs(tmp_path):
    ref, after, first = successful_state(tmp_path)
    snapshot = deepcopy(after)
    snapshot["resources"]["room_reservations"].pop()
    writes = [{"event_id": first + i, "created": 101 + i} for i in range(6)]
    assert (
        scoring.check_at_deadline(ref, snapshot, writes, started_at=100, budget_seconds=600)[
            "completed"
        ]
        == 5
    )
    assert scoring.check(ref, after)["completed"] == 6
    with pytest.raises(ValueError):
        scoring.check_at_deadline(ref, snapshot, writes, started_at=100, budget_seconds=0)
