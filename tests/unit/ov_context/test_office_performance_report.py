# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Performance aggregation never infers task success from terminal status."""

import json

import pytest

from benchmark.office.suites.common.policy import acceptance_policy
from benchmark.office.suites.common.report import compare, read_run, summarize


def record(index, seconds=300, mode="baseline"):
    return {
        "run_id": f"r{index}",
        "case": "ppt_update",
        "mode": mode,
        "phase": "Q",
        "acceptance_policy": acceptance_policy("ppt_update"),
        "statistics": {
            "seconds": seconds,
            "status": "error" if index == 3 else "done",
            "total_tokens": 100 + index,
        },
    }


def test_three_runs_include_failures_but_leave_business_results_for_review():
    result = summarize([record(1, 200), record(2, 300), record(3, 400)])
    assert result["mean_run_seconds"] == 300
    assert result["mean_agent_tokens"] == 102
    assert result["run_statuses"] == ["done", "done", "error"]
    assert result["review_status"] == "pending"
    assert "success_rate" not in result and "success_count" not in result


def test_cohort_rejects_duplicates_incomplete_repetitions_and_mixed_policy():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([record(1)] * 3)
    with pytest.raises(ValueError, match="3"):
        summarize([record(1)])
    runs = [record(i) for i in range(3)]
    runs[-1]["acceptance_policy"]["budget_seconds"] = 250
    with pytest.raises(ValueError, match="policy"):
        summarize(runs)


def test_missing_ov_usage_remains_unknown_and_comparison_has_no_success_delta():
    base = summarize([record(i) for i in range(3)])
    memory = summarize([record(i, mode="ordinary") for i in range(3)])
    assert memory["mean_total_llm_tokens"] is None
    result = compare(base, memory)
    assert result["total_llm_token_reduction_percent"] is None
    assert "success_improvement_pp" not in result


def test_reader_uses_raw_evidence_without_a_grade_file(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps(record(1)))
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "case": "ppt_update",
                "phase": "Q",
                "review_prompt": "prompt.md",
                "manifest": {"acceptance_policy": acceptance_policy("ppt_update")},
            }
        )
    )
    result = read_run(tmp_path)
    assert result["run_id"] == "r1" and result["phase"] == "Q"
    assert not (tmp_path / "grade.json").exists()
