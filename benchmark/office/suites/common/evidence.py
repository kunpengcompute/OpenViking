# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Expose captured benchmark evidence without invoking a task judge."""

import json
from pathlib import Path


def export_evidence(folder: Path, statistics: dict) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    snapshots = [json.loads(line) for line in (folder / "snapshots.jsonl").read_text().splitlines()]
    return {
        "case": manifest["case"],
        "phase": manifest["phase"],
        "review_status": "pending",
        "statistics": statistics,
        "manifest": manifest,
        "snapshots": snapshots,
        "evidence_directory": str(folder),
        "review_prompt": "benchmark/office/evaluation/AI_REVIEW_PROMPT.md",
    }
