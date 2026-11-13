"""Freeze relative dates, candidate selection and real predecessor-ID scoring."""

from copy import deepcopy

import pytest

from benchmark.office.cases.schedule.fortnight import (
    check,
    prepare,
)


def completed_state(reference):
    after = deepcopy(reference["before"])
    resolved = {}
    next_id = max(e["id"] for e in after["events"]) + 1
    for target in reference["targets"]:
        if target["existing_id"] is not None:
            resolved[target["title"]] = target["existing_id"]
            continue
        row = {k: v for k, v in target.items() if k not in ["existing_id", "parent", "date_rule"]}
        row.update(id=next_id, source="agent", nonce=f"agent-{next_id}")
        if target["parent"]:
            row["notes"] += (
                f"；前置会议：{target['parent']}；前置编号：CAL-{resolved[target['parent']]}"
            )
        after["events"].append(row)
        resolved[target["title"]] = next_id
        next_id += 1
    return after


def test_twelve_results_have_four_reuses_and_valid_dynamic_references(tmp_path):
    reference = prepare(tmp_path, "fortnight-test", "http://127.0.0.1:19482")
    assert len(reference["targets"]) == 12
    assert sum(t["existing_id"] is not None for t in reference["targets"]) == 4
    assert check(reference, reference["before"])["completed"] == 4
    result = check(reference, completed_state(reference))
    assert result["completed"] == 12 and result["new_event_count"] == 8
    assert result["all_backend_requirements_met"]
    assert result["forced_two_phase_execution"] is False
    assert "前置编号：CAL-实际编号" in reference["task"]
    assert "先统一排期" not in reference["task"]
    assert reference["targets"][-1]["start"] == "2026-11-13T16:00+08:00"
    with pytest.raises(FileExistsError):
        prepare(tmp_path, "fortnight-test", "http://127.0.0.1:19482")


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_date",
        "wrong_duration",
        "wrong_calendar",
        "wrong_parent",
        "missing_parent",
        "duplicate",
        "changed_seed",
    ],
)
def test_every_added_constraint_is_checked(tmp_path, mutation):
    reference = prepare(tmp_path, "fortnight-test", "http://127.0.0.1:19482")
    after = completed_state(reference)
    row = next(e for e in after["events"] if e["title"] == "交付复盘" and e["source"] == "agent")
    if mutation == "wrong_date":
        row["start"] = row["start"].replace("11-13", "11-12")
        row["end"] = row["end"].replace("11-13", "11-12")
    elif mutation == "wrong_duration":
        row["end"] = row["end"].replace("16:45", "16:30")
    elif mutation == "wrong_calendar":
        row["calendar_id"] = "personal"
    elif mutation == "wrong_parent":
        row["notes"] = "项目：OV集成验收；前置会议：回归验收；前置编号：CAL-9999"
    elif mutation == "missing_parent":
        row["notes"] = "项目：OV集成验收"
    elif mutation == "duplicate":
        after["events"].append({**row, "id": 9999, "nonce": "duplicate"})
    else:
        after["background_sha256"] = "changed"
    result = check(reference, after)
    assert not result["all_backend_requirements_met"]
    if mutation != "changed_seed":
        assert result["completed"] == 11


def test_preexisting_duration_distractor_can_be_preserved_while_correct_slot_is_created(tmp_path):
    reference = prepare(tmp_path, "fortnight-test", "http://127.0.0.1:19482")
    result = check(reference, completed_state(reference))
    item = next(i for i in result["items"] if i["title"] == "回归联调")
    assert item["completed"] and item["expected"]["start"] == "2026-11-10T14:00+08:00"
    old = next(e for e in reference["before"]["events"] if e["nonce"] == "fortnight-duration-10")
    assert old["start"] == "2026-11-10T10:00+08:00" and old["end"] == "2026-11-10T10:30+08:00"
