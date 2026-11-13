# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Frozen data tasks preserve their own script, JSON evidence and document section."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.office.suites.reqdoc_generation.run import seed_contract, seed_fingerprint

SPEC = Path("benchmark/office/cases/reqdoc_generation/spec.json")


def task_body(text, number):
    body = text.split(f"【RD0{number}：", 1)[1]
    return body.split(f"【RD0{number + 1}：", 1)[0] if number < 6 else body.split("【最后收尾】")[0]


def test_each_data_task_has_its_own_script_evidence_and_section():
    spec = json.loads(SPEC.read_text())
    q = Path(spec["phases"]["Q"]["prompt"]).read_text()
    assert "每章完成后保存正文及本章必要证据" in q
    for number in [5, 6]:
        task = f"RD0{number}"
        contract = spec["data_task_artifacts"][task]
        body = task_body(q, number)
        assert contract["script"] in body
        assert Path(contract["json"]).name in body
        assert contract["section"] in body
        assert "保存" in body and "独立复算" in body
        assert all(
            other["script"] not in body
            for key, other in spec["data_task_artifacts"].items()
            if key != task
        )


def test_prompts_and_hashes_match_without_numeric_answers():
    spec = json.loads(SPEC.read_text())
    prompts = {}
    for phase in ["H", "Q"]:
        path = Path(spec["phases"][phase]["prompt"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["phases"][phase]["sha256"]
        prompts[phase] = path.read_text()
    assert prompts["H"].startswith(prompts["Q"])
    for answer in ["24/30", "12/16", "80.0%", "75.0%", "I001", "/home/lhx"]:
        assert answer not in prompts["H"]


@pytest.mark.parametrize(
    "field",
    ["case_version", "rubric_version", "H_prompt_sha256", "Q_prompt_sha256", "input_fingerprints"],
)
def test_different_seed_contract_is_rejected(field):
    spec = json.loads(SPEC.read_text())
    seed = {**seed_contract(spec), "fingerprint": {"uri": "sha"}}
    seed[field] = {} if field == "input_fingerprints" else "another-contract"
    with pytest.raises(ValueError, match="different case"):
        seed_fingerprint(seed, spec)
