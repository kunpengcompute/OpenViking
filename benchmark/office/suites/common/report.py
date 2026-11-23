# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Aggregate terminal runtime and usage; task success is reviewed from evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .files_config import write_json
from .policy import valid_seconds


def complete_mean(values):
    return mean(values) if values and all(valid_seconds(v) for v in values) else None


def summarize(runs: list[dict]) -> dict:
    ids = [r["run_id"] for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot aggregate duplicate Runs")
    if len(runs) != 3:
        raise ValueError("A frozen office comparison needs exactly 3 terminal Runs per group")
    policy = runs[0]["acceptance_policy"]
    if any(r["acceptance_policy"] != policy or r["phase"] != "Q" for r in runs):
        raise ValueError("Mixed acceptance policy or preparation Runs")
    if len({(r["case"], r["mode"]) for r in runs}) != 1:
        raise ValueError("Mixed cases or experiment groups")
    signatures = [r.get("conditions", {}) for r in runs]
    if any(s != signatures[0] for s in signatures):
        raise ValueError("Model/input conditions differ within this group")
    agent = complete_mean([r["statistics"].get("total_tokens") for r in runs])
    ov = complete_mean(
        [r.get("ov_llm_tokens", 0 if r["mode"] == "baseline" else None) for r in runs]
    )
    return {
        "case": runs[0]["case"],
        "mode": runs[0]["mode"],
        "runs": ids,
        "policy": policy,
        "conditions": signatures[0],
        "review_status": "pending",
        "mean_run_seconds": complete_mean([r["statistics"].get("seconds") for r in runs]),
        "mean_agent_tokens": agent,
        "mean_ov_llm_tokens": ov,
        "mean_total_llm_tokens": agent + ov if agent is not None and ov is not None else None,
        "run_statuses": [r["statistics"].get("status") for r in runs],
    }


def compare(baseline: dict, memory: dict) -> dict:
    if baseline["policy"] != memory["policy"] or baseline["conditions"] != memory["conditions"]:
        raise ValueError("Comparison conditions or acceptance policy differ")

    def reduction(key):
        a, b = baseline[key], memory[key]
        return (1 - b / a) * 100 if a is not None and a > 0 and b is not None else None

    return {
        "latency_reduction_percent": reduction("mean_run_seconds"),
        "total_llm_token_reduction_percent": reduction("mean_total_llm_tokens"),
    }


def read_run(folder: Path) -> dict:
    result = json.loads((folder / "result.json").read_text())
    evidence = json.loads((folder / "evidence.json").read_text())
    condition_path = folder / "condition.json"
    condition = json.loads(condition_path.read_text()) if condition_path.exists() else {}
    execution = condition.get("execution", {})
    metadata = condition.get("metadata", {})
    mode = result.get("mode", result.get("group", condition.get("mode")))
    return {
        **result,
        "case": evidence["case"],
        "phase": evidence["phase"],
        "acceptance_policy": evidence["manifest"]["acceptance_policy"],
        "mode": mode,
        "review_prompt": evidence["review_prompt"],
        "conditions": {
            "model": metadata.get("model"),
            "inputs": condition.get("inputs", condition.get("input_sha256")),
            "case_version": condition.get("version", condition.get("task_version")),
            "reader": metadata.get("reader_defaults"),
            "profile": {
                k: execution.get(k)
                for k in (
                    "context_window",
                    "max_tokens",
                    "thinking",
                    "model_call_timeout_seconds",
                    "timeout_seconds",
                )
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for folder in args.runs:
        run = read_run(folder)
        if run["phase"] == "Q":
            groups[run["case"], run["mode"]].append(run)
    summaries = {f"{case}/{mode}": summarize(rows) for (case, mode), rows in groups.items()}
    comparisons = {}
    for (case, mode), _rows in groups.items():
        if mode != "baseline" and f"{case}/baseline" in summaries:
            comparisons[f"{case}/{mode}"] = compare(
                summaries[f"{case}/baseline"], summaries[f"{case}/{mode}"]
            )
    write_json(args.output, {"groups": summaries, "comparisons": comparisons})


if __name__ == "__main__":
    main()
