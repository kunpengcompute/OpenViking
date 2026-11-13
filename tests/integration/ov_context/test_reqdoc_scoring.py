# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Current document rubric: six chapter weights, 27 checkpoints and quoted evidence."""

import hashlib
import json

import pytest

from benchmark.office.suites.reqdoc_generation.scoring import (
    CHECK_CHAPTERS,
    CHECKPOINTS,
    LAYOUT,
    RUBRIC_VERSION,
    TITLES,
    grade,
)


def fixture(tmp_path):
    document = tmp_path / "artifacts/requirements.md"
    document.parent.mkdir()
    sections = []
    for i, title in enumerate(TITLES, 1):
        body = f"章节{i}有效证据"
        sections.append(f"## {i} {title}\n{body}\n")
    document.write_text("\n".join(sections))
    checks = []
    for task, criteria in CHECKPOINTS.items():
        _, chapters, section = LAYOUT[task]
        for index, _ in enumerate(criteria):
            chapter = CHECK_CHAPTERS.get(task, [chapters] * len(criteria))[index][0]
            quote = f"小节{section[-1]}独立证据" if section else f"章节{chapter}有效证据"
            checks.append(
                {
                    "id": f"{task}.{index + 1}",
                    "status": "passed",
                    "evidence": "分析员已核对该子项原文和相关证据",
                    "quotes": [quote],
                    "review_kind": "analyst",
                }
            )
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "document_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                "checks": checks,
            }
        )
    )
    return document, reviews


def test_27_checks_keep_six_equal_weight_tasks(tmp_path):
    _, reviews = fixture(tmp_path)
    result = grade(tmp_path, reviews)
    assert result["points"] == pytest.approx(6)
    assert sum(len(task["checks"]) for task in result["subtasks"]) == 27
    assert all(task["points"] == pytest.approx(1) for task in result["subtasks"])
    assert result["subtasks"][5]["title"] == "首次终态安装成功率分析"


@pytest.mark.parametrize("key,denominator", [("RD01.4", 4), ("RD03.5", 5), ("RD06.5", 5)])
def test_partial_credit_uses_its_task_checkpoint_count(tmp_path, key, denominator):
    _, reviews = fixture(tmp_path)
    data = json.loads(reviews.read_text())
    next(c for c in data["checks"] if c["id"] == key)["status"] = "failed"
    reviews.write_text(json.dumps(data))
    assert grade(tmp_path, reviews)["points"] == pytest.approx(6 - 1 / denominator)


@pytest.mark.parametrize("key", ["RD01.4", "RD02.4", "RD03.5", "RD04.4", "RD05.5"])
def test_restored_rules_must_quote_their_assigned_chapter(tmp_path, key):
    _, reviews = fixture(tmp_path)
    data = json.loads(reviews.read_text())
    next(c for c in data["checks"] if c["id"] == key)["quotes"] = ["章节6有效证据"]
    reviews.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="task scope"):
        grade(tmp_path, reviews)


def test_closing_evidence_cannot_replace_missing_acceptance_case(tmp_path):
    document, reviews = fixture(tmp_path)
    document.write_text(document.read_text() + "\n## 收尾核验记录\n仅在收尾出现的检查证据\n")
    data = json.loads(reviews.read_text())
    data["document_sha256"] = hashlib.sha256(document.read_bytes()).hexdigest()
    next(c for c in data["checks"] if c["id"] == "RD06.1")["quotes"] = ["仅在收尾出现的检查证据"]
    reviews.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="task scope"):
        grade(tmp_path, reviews)


def test_missing_delivery_does_not_change_content_score(tmp_path):
    _, reviews = fixture(tmp_path)
    result = grade(tmp_path, reviews)
    assert result["points"] == pytest.approx(6)
    assert result["delivery"]["final_download_link"] is False
    assert result["delivery"]["execution_done"] is False


def test_budget_success_requires_all_checks_but_keeps_delivery_separate(tmp_path):
    _, reviews = fixture(tmp_path)
    checks = json.loads(reviews.read_text())
    checks["checks"][0]["status"] = "failed"
    reviews.write_text(json.dumps(checks))
    raw = b'{"statistics":{"seconds":600,"status":"done"}}'
    (tmp_path / "result.json").write_bytes(raw)
    (tmp_path / "completion-review.json").write_text(
        json.dumps(
            {
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "subtasks": [
                    {"id": f"RD{i:02}", "completed_seconds": seconds, "evidence": f"call-{i}"}
                    for i, seconds in enumerate([100, 200, 300, 400, 550, 551], 1)
                ],
            }
        )
    )
    result = grade(tmp_path, reviews)
    assert result["latency"]["success_count"] == 4
    assert result["latency"]["budget_seconds"] == 550
    assert result["latency"]["run_seconds"] == 600
    assert not result["delivery"]["final_download_link"]
