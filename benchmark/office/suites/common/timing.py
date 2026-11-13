# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Score verified subtask completion times independently of the Run watchdog."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from benchmark.paths import OFFICE_ROOT


def scoring_policy(case: str) -> dict:
    policy = json.loads((OFFICE_ROOT / "cases/scoring.json").read_text())
    budget = policy["budget_seconds"][case]
    return {**policy, "case": case, "budget_seconds": budget}


def valid_seconds(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def score_subtasks(
    case: str, phase: str, statistics: dict, subtasks: list[dict], *, policy: dict | None = None
) -> dict:
    """Times require evidence of all task assertions, using the Run's clock.

    Missing evidence stays unverified. A late repair of one task cannot make
    another task late. Final delivery and Run status remain separate outcomes.
    """
    policy = scoring_policy(case) if policy is None else policy
    if policy.get("case") != case or policy.get("task_count") != 6:
        raise ValueError("Frozen scoring policy does not match this case")
    if not valid_seconds(policy["budget_seconds"]) or policy["budget_seconds"] <= 0:
        raise ValueError("Scoring budget must be positive")
    ids = [row["id"] for row in subtasks]
    expected_count = (
        policy.get("preparation_task_counts", {}).get(case, policy["task_count"])
        if phase == "H"
        else policy["task_count"]
    )
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        count_label = "six" if expected_count == 6 else str(expected_count)
        raise ValueError(f"Scoring needs {count_label} unique subtasks in business order")
    if phase not in {"H", "Q"}:
        raise ValueError("Scoring phase must be H or Q")
    budget = policy["budget_seconds"] if phase == "Q" else None
    elapsed = statistics.get("seconds")
    run_seconds = elapsed if valid_seconds(elapsed) else None
    rows, cumulative, prefix_upper_bound = [], 0, False
    for row in subtasks:
        passed = row.get("final_passed")
        if passed is not None and type(passed) is not bool:
            raise ValueError("final_passed must be a boolean or null")
        seconds = row.get("completed_seconds")
        if seconds is not None:
            if not valid_seconds(seconds) or not row.get("evidence"):
                raise ValueError("Completion time needs finite seconds and verification evidence")
            if run_seconds is not None and seconds > run_seconds:
                raise ValueError("Subtask completion cannot occur after the Run ended")
        known = passed is True and seconds is not None
        upper_bound = row.get("time_is_upper_bound", False)
        if type(upper_bound) is not bool:
            raise ValueError("time_is_upper_bound must be a boolean")
        cumulative = max(cumulative, seconds) if cumulative is not None and known else None
        prefix_upper_bound = prefix_upper_bound or upper_bound
        within = None
        if budget is not None:
            if passed is False:
                within = False
            elif known:
                within = seconds <= budget
                if upper_bound and not within:
                    within = None  # A late upper bound does not prove a late task.
        rows.append(
            {
                **row,
                "prefix_cumulative_seconds": cumulative,
                "prefix_time_is_upper_bound": prefix_upper_bound
                if cumulative is not None
                else None,
                "within_budget_passed": within,
            }
        )
    success = sum(row["within_budget_passed"] is True for row in rows)
    unverified = sum(row["within_budget_passed"] is None for row in rows) if budget else 0
    return {
        "policy": policy,
        "applies": phase == "Q",
        "budget_seconds": budget,
        "run_seconds": run_seconds,
        "run_status": statistics.get("status"),
        "success_count": success if budget is not None else None,
        "total": len(rows),
        "success_rate": success / len(rows) if budget is not None and not unverified else None,
        "unverified_subtasks": unverified,
        "unknown_completion_times": sum(
            row.get("final_passed") is True and row.get("completed_seconds") is None for row in rows
        ),
        "subtasks": rows,
    }


def reviewed_timings(folder: Path, subtasks: list[dict]) -> list[dict]:
    """Apply an optional timing review bound to this exact result.json.

    Content acceptance comes from the case grader; the timing review only
    supplies verified completion times and trace references for those tasks.
    """
    path = folder / "completion-review.json"
    if not path.exists():
        return subtasks
    review = json.loads(path.read_text())
    digest = hashlib.sha256((folder / "result.json").read_bytes()).hexdigest()
    if review.get("result_sha256") != digest:
        raise ValueError("Completion review belongs to a different Run result")
    timings = {}
    ids = {row["id"] for row in subtasks}
    for row in review["subtasks"]:
        key = row["id"]
        if key not in ids or key in timings:
            raise ValueError("Unknown or duplicate subtask timing")
        timings[key] = {
            name: row[name]
            for name in ("completed_seconds", "evidence", "time_is_upper_bound")
            if name in row
        }
    return [{**row, **timings.get(row["id"], {})} for row in subtasks]


def main():
    """Score reviewed business/trace observations for any office case."""
    import argparse

    from .files_config import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=json.loads((OFFICE_ROOT / "cases/scoring.json").read_text())["budget_seconds"],
    )
    parser.add_argument("--phase", choices=["H", "Q"], default="Q")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    observations = json.loads(args.observations.read_text())
    if observations.get("result_sha256") != digest or observations.get("case") != args.case:
        raise ValueError("Observations must match this case and exact Run result")
    result = json.loads(raw)
    value = score_subtasks(args.case, args.phase, result["statistics"], observations["subtasks"])
    write_json(args.output, {**value, "run_id": result.get("run_id"), "result_sha256": digest})


if __name__ == "__main__":
    main()
