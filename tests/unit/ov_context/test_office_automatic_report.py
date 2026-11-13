# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Full-run means, independent step scores, and repeated-run cohort guards."""

import pytest

from benchmark.office.suites.common.report import summarize
from benchmark.office.suites.common.timing import score_subtasks


def record(index, seconds, times):
    subtasks = [
        {"id": f"t{i}", "final_passed": True, "completed_seconds": value, "evidence": "snapshot"}
        for i, value in enumerate(times)
    ]
    stats = {"seconds": seconds, "status": "done", "total_tokens": 100 + index}
    return {
        "run_id": f"r{index}",
        "case": "ppt_update",
        "mode": "baseline",
        "phase": "Q",
        "statistics": stats,
        "automatic": {"latency": score_subtasks("ppt_update", "Q", stats, subtasks)},
    }


def test_three_runs_keep_late_completions_and_step_scores_separate():
    runs = [
        record(1, 200, [10, 20, 30, 40, 130, 60]),
        record(2, 300, [10, 20, 30, 40, 50, 60]),
        record(3, 400, [10, 20, 30, 40, 50, 60]),
    ]
    result = summarize(runs)
    assert result["success_count"] == 17 and result["total"] == 18
    assert result["mean_run_seconds"] == 300
    assert result["mean_prefix_cumulative_seconds"][-1] == pytest.approx(250 / 3)
    assert result["mean_agent_tokens"] == 102


def test_cohort_rejects_duplicate_runs_and_mixed_budgets():
    one = record(1, 200, [10] * 6)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([one, one, one])
    two, three = record(2, 200, [10] * 6), record(3, 200, [10] * 6)
    three["automatic"]["latency"]["policy"]["budget_seconds"] = 250
    with pytest.raises(ValueError, match="policy"):
        summarize([one, two, three])


def test_incomplete_repetitions_do_not_claim_a_three_run_average():
    with pytest.raises(ValueError, match="3"):
        summarize([record(1, 200, [10] * 6)])
