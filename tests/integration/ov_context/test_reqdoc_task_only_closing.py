"""Task prompts describe deliverables; evaluation owns timing and presentation checks."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.office.suites.reqdoc_generation.run import seed_contract, seed_fingerprint
from benchmark.office.suites.reqdoc_generation.scoring import (
    CHECKPOINTS,
    RUBRIC_VERSION,
    grade,
)

SPEC = Path("benchmark/office/cases/reqdoc_generation/spec.json")


def test_prompt_hashes_and_functional_tasks_preserved():
    new = json.loads(SPEC.read_text())
    for phase in ["H", "Q"]:
        path = Path(new["phases"][phase]["prompt"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == new["phases"][phase]["sha256"]
    q = Path(new["phases"]["Q"]["prompt"]).read_text()
    assert "【公共要求】" not in q
    assert "评分" not in q and "耗时" not in q
    closing = q.split("【最后收尾】", 1)[1]
    for value in ["session_files", 'action="list"', "Markdown", "修正", "修复"]:
        assert value not in closing
    assert "保持正文及计算证据原样" in closing and "下载链接" in closing
    h = Path(new["phases"]["H"]["prompt"]).read_text()
    assert h.startswith(q)
    closing_guide = h.split("4. 六章完成后", 1)[1]
    assert all(
        value not in closing_guide for value in ["session_files", "Markdown", "修正", "修复"]
    )


def test_evaluator_keeps_closing_scope_and_requires_fresh_seed():
    spec = json.loads(SPEC.read_text())
    assert spec["closing"]["part_of_subtasks"] is False
    assert spec["closing"]["latency_bucket"] == "closing"
    assert spec["closing"]["repair_document"] is False
    assert spec["closing"]["report_issues"] is True
    assert sum(spec["task_checkpoint_counts"].values()) == 27
    with pytest.raises(ValueError, match="different case"):
        seed_fingerprint(
            {**seed_contract(spec), "case_version": "another-case", "fingerprint": {"uri": "sha"}},
            spec,
        )


@pytest.mark.parametrize(
    "form,valid",
    [
        ("plain", True),
        ("markdown", True),
        ("later_link", True),
        ("wrong_host", False),
        ("missing", False),
    ],
)
def test_delivery_checks_real_artifact_without_tool_or_markdown_requirements(tmp_path, form, valid):
    document = tmp_path / "artifacts/requirements.md"
    document.parent.mkdir()
    document.write_text("## 1 需求概述与入口识别\n内容\n")
    url = "/api/sessions/s/artifacts/r/requirements.md"
    (tmp_path / "artifact-downloads.json").write_text(
        json.dumps(
            [
                {
                    "filename": "requirements.md",
                    "url": url,
                    "status": 200,
                    "sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                }
            ]
        )
    )
    (tmp_path / "condition.json").write_text(json.dumps({"chat_url": "http://localhost:20004"}))
    (tmp_path / "result.json").write_text(json.dumps({"statistics": {"status": "done"}}))
    answers = {
        "plain": f"文档：http://localhost:20004{url}",
        "markdown": f"[文档]({url})",
        "later_link": f"资料来源：https://source.example/info\n文档：http://localhost:20004{url}",
        "wrong_host": f"https://invented.invalid{url}",
        "missing": "文档已完成。",
    }
    (tmp_path / "answer.md").write_text(answers[form])
    result = grade(tmp_path, rubric_version=RUBRIC_VERSION)
    delivery = result["delivery"]
    assert delivery["final_download_link"] is valid
    assert delivery["artifact_link_correct"] is valid
    assert "primary_url_correct" not in delivery and "clickable_download_link" not in delivery
    assert delivery["download_hash"] is True
    assert sum(len(task["checks"]) for task in result["subtasks"]) == 27
    assert [len(t["checks"]) for t in result["subtasks"]] == [
        len(values) for values in CHECKPOINTS.values()
    ]
