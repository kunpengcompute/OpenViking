"""Task prompts describe deliverables; evaluation owns timing and presentation checks."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.office.suites.reqdoc_generation.run import seed_contract, seed_fingerprint

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
