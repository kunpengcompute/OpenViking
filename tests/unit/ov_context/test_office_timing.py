# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-task deadlines, independent timing and immutable evidence binding."""

import hashlib
import json
import subprocess
import sys

import pytest

from benchmark.office.suites.common.timing import (
    reviewed_timings,
    score_subtasks,
    scoring_policy,
)


def tasks(times, *, passed=True):
    return [
        {
            "id": f"task-{i}",
            "final_passed": passed,
            "completed_seconds": seconds,
            "evidence": f"verified save and readback: event-{i}",
        }
        for i, seconds in enumerate(times, 1)
    ]


@pytest.mark.parametrize(
    "case,budget",
    [
        ("schedule", 450),
        ("timesheet", 300),
        ("mail", 550),
        ("ppt_update", 120),
        ("reqdoc_generation", 550),
    ],
)
def test_all_case_budgets_are_inclusive_and_independent_of_run_deadline(case, budget):
    result = score_subtasks(
        case,
        "Q",
        {"seconds": 900, "status": "done"},
        tasks([1, budget, budget + 0.001, 4, 5, 6]),
    )
    assert result["success_count"] == 5
    assert result["success_rate"] == 5 / 6
    assert result["run_seconds"] == 900
    assert result["budget_seconds"] == budget
    assert scoring_policy(case)["repetitions"] == 3


def test_late_repair_does_not_make_a_later_task_late():
    result = score_subtasks(
        "ppt_update",
        "Q",
        {"seconds": 200, "status": "error"},
        tasks([46.820, 56.701, 59.911, 69.213, 137.377, 89.938]),
    )
    assert result["success_count"] == 5
    assert result["subtasks"][5]["within_budget_passed"]
    assert result["subtasks"][5]["prefix_cumulative_seconds"] == 137.377
    assert result["run_status"] == "error"  # Final delivery has its own outcome.


def test_three_runs_use_18_tasks_and_retain_late_run_time():
    times = [
        [94.318, 141.580, 144.938, 149.160, 175.755, 183.210],
        [65.415, 80.904, 83.611, 86.405, 90.374, 101.882],
        [46.820, 56.701, 59.911, 69.213, 137.377, 89.938],
    ]
    results = [score_subtasks("ppt_update", "Q", {"seconds": 200}, tasks(t)) for t in times]
    assert sum(r["success_count"] for r in results) == 12
    assert sum(r["total"] for r in results) == 18
    assert all(r["run_seconds"] == 200 for r in results)


def test_unknown_time_and_failed_task_do_not_erase_later_successes():
    rows = tasks([1, None, 3, 4, 5, 6])
    rows[2]["final_passed"] = False
    result = score_subtasks("mail", "Q", {}, rows)
    assert result["success_count"] == 4
    assert result["unknown_completion_times"] == 1
    assert result["subtasks"][5]["prefix_cumulative_seconds"] is None
    assert result["subtasks"][5]["within_budget_passed"]
    assert result["success_rate"] is None and result["unverified_subtasks"] == 1


def test_terminal_upper_bound_can_prove_success_but_not_lateness():
    rows = tasks([100, 200, 300, 400, 550, 600])
    for row in rows:
        row["time_is_upper_bound"] = True
    result = score_subtasks("mail", "Q", {"seconds": 600}, rows)
    assert result["success_count"] == 5 and result["success_rate"] is None
    assert result["subtasks"][-1]["within_budget_passed"] is None
    assert result["subtasks"][-1]["prefix_time_is_upper_bound"]


def test_h_is_not_included_in_q_budget_scoring():
    result = score_subtasks("mail", "H", {"seconds": 1000}, tasks([1] * 3))
    assert result["success_count"] is None and result["success_rate"] is None
    assert result["budget_seconds"] is None and not result["applies"]


@pytest.mark.parametrize("seconds", [True, -1, "10", float("nan"), float("inf"), 501])
def test_invalid_or_post_run_completion_time_is_rejected(seconds):
    with pytest.raises(ValueError):
        score_subtasks("schedule", "Q", {"seconds": 500}, tasks([seconds] * 6))


def test_evidence_and_exact_six_unique_tasks_are_required():
    rows = tasks([1] * 6)
    with pytest.raises(ValueError, match="six unique"):
        score_subtasks("schedule", "Q", {}, rows[:5])
    rows[0]["id"] = rows[1]["id"]
    with pytest.raises(ValueError, match="six unique"):
        score_subtasks("schedule", "Q", {}, rows)
    rows = tasks([1] * 6)
    rows[0].pop("evidence")
    with pytest.raises(ValueError, match="evidence"):
        score_subtasks("schedule", "Q", {}, rows)


def test_timing_review_cannot_override_content_grade_or_another_run(tmp_path):
    raw = b'{"run_id":"r","statistics":{"seconds":600}}'
    (tmp_path / "result.json").write_bytes(raw)
    review = {
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "subtasks": [{**row, "final_passed": True} for row in tasks([1] * 6)],
    }
    path = tmp_path / "completion-review.json"
    path.write_text(json.dumps(review))
    graded = reviewed_timings(tmp_path, tasks([None] * 6, passed=False))
    assert not any(row["final_passed"] for row in graded)
    assert score_subtasks("mail", "Q", {}, graded)["success_count"] == 0
    (tmp_path / "result.json").write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="different Run"):
        reviewed_timings(tmp_path, graded)


def test_scoring_cli_saves_the_actual_policy_and_complete_runtime(tmp_path):
    raw = b'{"run_id":"r","statistics":{"seconds":600,"status":"done"}}'
    result = tmp_path / "result.json"
    result.write_bytes(raw)
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            {
                "case": "timesheet",
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "subtasks": tasks([100, 200, 300, 301, 302, 303]),
            }
        )
    )
    output = tmp_path / "score.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.office.suites.common.timing",
            "--case",
            "timesheet",
            "--result",
            str(result),
            "--observations",
            str(observations),
            "--output",
            str(output),
        ],
        check=True,
    )
    score = json.loads(output.read_text())
    assert score["run_id"] == "r" and score["run_seconds"] == 600
    assert score["success_count"] == 3 and score["total"] == 6
    assert score["policy"]["budget_seconds"] == 300
