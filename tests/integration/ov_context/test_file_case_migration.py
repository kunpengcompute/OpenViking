# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Only frozen task specs choose inputs; oracle files never enter Agent uploads."""

import hashlib
import json

import pytest

from benchmark.office.cases.ppt_update.files import load_case as load_ppt
from benchmark.office.cases.reqdoc_generation.files import load_case as load_document
from benchmark.paths import OFFICE_ROOT


async def test_upload_size_active_run_and_session_isolation(chat, skill):
    from dataclasses import replace

    client, runtime = chat
    runtime.settings = replace(runtime.settings, pptx_skill=skill)
    first = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    second = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    source = b"fixed input\n"
    endpoint = f"/api/sessions/{first}/inputs/source.txt"
    runtime.active[first] = object()
    assert (await client.post(endpoint, content=source)).status_code == 409
    runtime.active.pop(first)
    assert (await client.post(endpoint, content=b"x" * (10 * 1024 * 1024 + 1))).status_code == 413
    response = await client.post(endpoint, content=source)
    assert response.status_code == 200
    assert response.json()["sha256"] == hashlib.sha256(source).hexdigest()
    assert not (runtime.settings.data_dir / "inputs" / second / "source.txt").exists()


async def test_shared_python_helper_upload_and_run_workspace_preserve_bytes(chat, skill, tmp_path):
    from dataclasses import replace

    from examples.deepagents_ov_context.chat.artifacts import valid_artifact_name, valid_input_name
    from examples.deepagents_ov_context.chat.office_skills import OfficeSkillRun

    client, runtime = chat
    runtime.settings = replace(runtime.settings, pptx_skill=skill)
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    _, sources = load_document()
    source = next(p for p in sources if p.name == "record_helpers.py")
    response = await client.post(
        f"/api/sessions/{sid}/inputs/{source.name}", content=source.read_bytes()
    )
    assert response.status_code == 200
    assert response.json()["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    run = OfficeSkillRun(
        {"pptx": skill},
        None,
        tmp_path / "artifacts",
        sid,
        "helper-test-run",
        input_dir=runtime.settings.data_dir / "inputs" / sid,
    )
    assert (run.folder / "inputs" / source.name).read_bytes() == source.read_bytes()
    assert valid_input_name(source.name)
    assert not valid_artifact_name(source.name)
    assert not valid_input_name("../record_helpers.py")
    assert not valid_input_name("record_helpers.sh")


@pytest.mark.parametrize("task", ["ppt_update", "reqdoc_generation"])
def test_frozen_assets_match_their_spec(task):
    spec_path = OFFICE_ROOT / "cases" / task / "spec.json"
    spec, value = (load_ppt if task == "ppt_update" else load_document)(spec_path)
    inputs = [value] if task == "ppt_update" else value
    assert len(inputs) == (1 if task == "ppt_update" else 7)
    assert not {"oracle.json", "spec.json", "case.json"}.intersection(p.name for p in inputs)
    assert set(spec["phases"]) == {"H", "Q"}


@pytest.mark.parametrize("task", ["ppt_update", "reqdoc_generation"])
def test_corrupted_input_fingerprint_is_rejected(tmp_path, task):
    spec = json.loads((OFFICE_ROOT / "cases" / task / "spec.json").read_text())
    if task == "ppt_update":
        spec["input_sha256"] = "0" * 64
    else:
        spec["inputs"][0]["sha256"] = "0" * 64
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="SHA256|hash mismatch"):
        (load_ppt if task == "ppt_update" else load_document)(path)
