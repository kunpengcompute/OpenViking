# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Capture original revisions and export evidence without synthesizing grades."""

import hashlib

from benchmark.office.suites.common.audit import RunAudit
from benchmark.office.suites.common.evidence import export_evidence


def test_document_revisions_remain_immutable_and_export_does_not_grade(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "output/requirements.md"
    output.parent.mkdir(parents=True)
    folder = tmp_path / "audit"
    audit = RunAudit(workspace, folder, "reqdoc_generation", "Q")
    output.write_text("first revision")
    audit.capture(10, "tool.end", tool_call_id="call-1")
    output.write_text("second revision")
    audit.capture(20, "run.end")
    result = export_evidence(folder, {"status": "done", "seconds": 20})
    snapshots = result["snapshots"]
    for expected, snap in zip([b"first revision", b"second revision"], snapshots, strict=True):
        digest = snap["files"]["output/requirements.md"]
        assert digest == hashlib.sha256(expected).hexdigest()
        assert (folder / "blobs" / digest).read_bytes() == expected
    assert result["review_status"] == "pending"
    assert snapshots[0]["tool_call_id"] == "call-1"
    assert "latency" not in result and "success_rate" not in result
    assert not (folder / "grade.json").exists()
    assert export_evidence(folder, result["statistics"]) == result


def test_capture_errors_are_retained_for_the_reviewer(tmp_path):
    audit = RunAudit(tmp_path / "workspace", tmp_path / "audit", "reqdoc_generation", "Q")
    (audit.workspace / "output").mkdir(parents=True)
    target = tmp_path / "outside.md"
    target.write_text("external contents")
    (audit.workspace / "output/requirements.md").symlink_to(target)
    audit.capture(1, "run.end")
    # Capture errors remain visible instead of becoming business failures.
    evidence = export_evidence(audit.folder, {"seconds": 1})
    assert evidence["snapshots"][0]["capture_error"]["type"] == "ValueError"
    assert "symlink" in evidence["snapshots"][0]["capture_error"]["message"]
    assert evidence["review_status"] == "pending"


async def test_chat_exports_evidence_and_has_no_grading_endpoint(chat):
    from .chat_model_support import NoToolModel
    from .test_chat_api import parse_events

    client, runtime = chat
    runtime.model = NoToolModel()
    session = await client.post(
        "/api/sessions",
        json={"experiment_id": "office-auto--ppt_update--Q", "execution": {"mode": "baseline"}},
    )
    sid = session.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "done"})
    events = parse_events(response.text)
    rid = next(data["run_id"] for event, data in events if event == "start")
    url = f"/api/sessions/{sid}/runs/{rid}/benchmark"
    result = await client.get(url + "/evidence")
    assert result.status_code == 200
    assert result.json()["review_status"] == "pending"
    assert len(result.json()["snapshots"]) >= 2
    assert "success_rate" not in result.json()
    assert (await client.post(url)).status_code == 404
    assert (await client.get(url + "/evidence")).json() == result.json()
    plain = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    plain_id = plain.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": plain_id, "message": "done"})
    plain_rid = next(d["run_id"] for e, d in parse_events(response.text) if e == "start")
    assert (
        await client.get(f"/api/sessions/{plain_id}/runs/{plain_rid}/benchmark/evidence")
    ).status_code == 404
