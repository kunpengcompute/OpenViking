# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Collect a self-contained review directory from runner and Chat evidence."""

import hashlib
import json
from pathlib import Path

import pytest


def put(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def sample(tmp_path):
    folder = tmp_path / "Q-mail"
    chat = tmp_path / "chat"
    raw = "主题\n完整邮件正文".encode()
    digest = hashlib.sha256(raw).hexdigest()
    audit = chat / "benchmark/s-1/r-1"
    (audit / "blobs").mkdir(parents=True)
    (audit / "blobs" / digest).write_bytes(raw)
    manifest = {"case": "mail", "phase": "Q", "reference": {"targets": [{"body": "完整邮件正文"}]}}
    put(audit / "manifest.json", manifest)
    row = {
        "elapsed_seconds": 12.5,
        "event": "run.end",
        "files": {},
        "published": {"mail_01.txt": digest},
        "business": None,
    }
    (audit / "snapshots.jsonl").write_text(json.dumps(row) + "\n")
    put(
        folder / "result.json",
        {
            "case": "mail",
            "phase": "Q",
            "mode": "baseline",
            "session_id": "s-1",
            "run_id": "r-1",
            "statistics": {"status": "done", "seconds": 12.5},
        },
    )
    put(
        folder / "condition.json",
        {
            "case": "mail",
            "phase": "Q",
            "mode": "baseline",
            "session_id": "s-1",
            "query": "查询邮件",
            "query_sha256": hashlib.sha256("查询邮件".encode()).hexdigest(),
        },
    )
    put(folder / "trace.json", {"run_id": "r-1", "events": []})
    put(
        folder / "evidence.json",
        {
            "case": "mail",
            "phase": "Q",
            "manifest": manifest,
            "snapshots": [row],
            "evidence_directory": "/server/path/not-mounted",
        },
    )
    (folder / "service.key").write_text("excluded")
    (folder / ".env").write_text("excluded")
    return folder, chat, raw, digest


def test_collector_copies_original_versions_and_an_index_without_credentials(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, raw, digest = sample(tmp_path)
    output = tmp_path / "review"
    result = collect([{"directory": folder}], output, chat_data=chat)
    assert result["complete"]
    assert result["runs"][0]["run_id"] == "r-1"
    assert (output / "AI_REVIEW_PROMPT.md").is_file()
    assert (output / "README.md").is_file()
    entries = json.loads((output / "files.json").read_text())
    blob = next(x for x in entries if x["path"].endswith("/blobs/" + digest))
    assert (output / blob["path"]).read_bytes() == raw
    version = next(
        x for x in entries if "/versions/" in x["path"] and x["path"].endswith("mail_01.txt")
    )
    assert (output / version["path"]).read_bytes() == raw
    assert not list(output.rglob("service.key")) and not list(output.rglob(".env"))
    assert all(not Path(x["path"]).is_absolute() for x in entries)


def test_missing_or_corrupt_blob_is_reported_without_inventing_a_score(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, digest = sample(tmp_path)
    (chat / "benchmark/s-1/r-1/blobs" / digest).write_bytes(b"corrupt")
    result = collect([{"directory": folder}], tmp_path / "review", chat_data=chat)
    assert not result["complete"]
    assert any(x["code"] == "hash_mismatch" for x in result["issues"])
    assert "success_rate" not in result


def test_existing_output_and_duplicate_run_selection_are_rejected(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, _ = sample(tmp_path)
    output = tmp_path / "review"
    output.mkdir()
    with pytest.raises(FileExistsError):
        collect([{"directory": folder}], output, chat_data=chat)
    assert not list(output.iterdir())
    with pytest.raises(ValueError, match="Duplicate"):
        collect([{"directory": folder}, {"directory": folder}], tmp_path / "other", chat_data=chat)
    assert not (tmp_path / "other").exists()


def test_failed_run_with_missing_blobs_keeps_evidence_and_reports_the_gap(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, digest = sample(tmp_path)
    value = json.loads((folder / "result.json").read_text())
    value["statistics"]["status"] = "error"
    put(folder / "result.json", value)
    (chat / "benchmark/s-1/r-1/blobs" / digest).unlink()
    output = tmp_path / "review"
    result = collect([{"directory": folder}], output, chat_data=chat)
    assert len(result["runs"]) == 1
    assert result["runs"][0]["statistics"]["status"] == "error"
    assert not result["complete"]
    assert any(x["code"] == "missing_file" for x in result["issues"])
    assert (output / result["runs"][0]["directory"] / "execution/trace.json").is_file()


def test_cli_can_collect_explicit_archived_audit_paths_from_a_relative_run_list(tmp_path):
    import subprocess
    import sys

    folder, chat, _, _ = sample(tmp_path)
    listing = tmp_path / "selected.json"
    listing.write_text(
        json.dumps([{"directory": folder.name, "audit_directory": "chat/benchmark/s-1/r-1"}])
    )
    output = tmp_path / "review"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.office.evaluation.collect",
            "--run-list",
            str(listing),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["complete"]
    assert json.loads((output / "runs.json").read_text())[0]["run_id"] == "r-1"


def test_collector_uses_hash_checked_uploaded_inputs_and_ignores_links(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, _ = sample(tmp_path)
    raw = b"frozen input material"
    put(
        folder / "input-receipts.json",
        [{"filename": "notes.txt", "sha256": hashlib.sha256(raw).hexdigest()}],
    )
    inputs = chat / "inputs/s-1"
    inputs.mkdir(parents=True)
    (inputs / "notes.txt").write_bytes(raw)
    artifacts = folder / "artifacts"
    artifacts.mkdir()
    (artifacts / "outside.txt").symlink_to(folder / "service.key")
    output = tmp_path / "review"
    result = collect([{"directory": folder}], output, chat_data=chat)
    destination = output / result["runs"][0]["directory"]
    assert (destination / "inputs/notes.txt").read_bytes() == raw
    assert not (destination / "artifacts/outside.txt").exists()
    assert any(x["code"] == "symlink" for x in result["issues"])


def test_download_receipt_hash_is_checked_after_artifact_directory_copy(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, _ = sample(tmp_path)
    (folder / "artifacts").mkdir()
    (folder / "artifacts/final.txt").write_text("actual downloaded content")
    put(
        folder / "artifact-downloads.json",
        [{"filename": "final.txt", "status": 200, "sha256": "0" * 64}],
    )
    result = collect([{"directory": folder}], tmp_path / "review", chat_data=chat)
    assert not result["complete"]
    assert any(x["code"] == "hash_mismatch" for x in result["issues"])


def test_audit_manifest_freezes_the_mail_targets_at_capture_time(tmp_path):
    from benchmark.office.datasets.mail.data import goals
    from benchmark.office.suites.common.audit import RunAudit

    folder = tmp_path / "audit"
    RunAudit(tmp_path / "workspace", folder, "mail", "Q")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["reference"]["targets"] == goals("Q")
    assert len(manifest["reference"]["targets"]) == 6


def test_valid_json_with_wrong_shape_is_retained_as_an_explicit_issue(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, _ = sample(tmp_path)
    put(folder / "trace.json", ["not a trace object"])
    result = collect([{"directory": folder}], tmp_path / "review", chat_data=chat)
    assert not result["complete"]
    assert any(x["code"] == "invalid_json_shape" for x in result["issues"])


def test_output_cannot_be_nested_in_the_assets_it_would_collect(tmp_path):
    from benchmark.office.evaluation.collect import collect

    folder, chat, _, _ = sample(tmp_path)
    source = tmp_path / "source"
    destination = source / "benchmark/office/datasets/mail/review"
    with pytest.raises(ValueError, match="source assets"):
        collect([{"directory": folder}], destination, chat_data=chat, source_root=source)
    assert not destination.exists()
