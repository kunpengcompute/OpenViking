"""Actual PPT stage files distinguish sequential verification from batch edits."""

import json
from zipfile import ZipFile

from benchmark.office.cases.ppt_update.scoring import (
    expected_package,
    package_bytes,
)
from benchmark.office.suites.ppt_update.observer import RevisionAudit
from benchmark.office.suites.ppt_update.scoring import score
from benchmark.paths import PROJECT_ROOT

SPEC = json.loads((PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json").read_text())
SOURCE = PROJECT_ROOT / SPEC["source"]


def write_deck(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    package = expected_package(package_bytes(SOURCE), {"edits": tasks})
    with ZipFile(path, "w") as archive:
        for name, raw in package.items():
            archive.writestr(name, raw)


def result_files(folder, deck):
    (folder / "artifacts").mkdir(exist_ok=True)
    (folder / "artifacts/updated.pptx").write_bytes(deck.read_bytes())
    (folder / "answer.md").write_text("/artifact/updated.pptx")
    (folder / "result.json").write_text(
        json.dumps(
            {
                "statistics": {"status": "done", "seconds": 1},
                "artifacts": [
                    {"filename": "updated.pptx", "status": 200, "url": "/artifact/updated.pptx"}
                ],
            }
        )
    )


def test_six_separate_saved_revisions_and_readback_are_recoverable(tmp_path):
    workspace = tmp_path / "workspace"
    audit = RevisionAudit(workspace, tmp_path / "audit")
    tasks = SPEC["phases"]["Q"]["tasks"]
    deck = workspace / "output/updated.pptx"
    for i in range(6):
        audit.capture("start", f"edit-{i}", "execute")
        write_deck(deck, tasks[: 2 * (i + 1)])
        audit.capture("end", f"edit-{i}", "execute")
        audit.capture(
            "start", f"read-{i}", "execute", {"command": "python verify.py output/updated.pptx"}
        )
        audit.capture(
            "end", f"read-{i}", "execute", {"command": "python verify.py output/updated.pptx"}
        )
    result_files(tmp_path, deck)
    value = score(SPEC, "Q", tmp_path, tmp_path / "audit", SOURCE)
    assert value["final_correct"] == 6 and value["final_preservation_ok"]
    assert value["revision_count"] == 6 and value["delivery_pass"]
    assert all(x["stage_evidence"] for x in value["subtasks"])
    assert not value["sequence_auto_verified"]


def test_all_edits_in_one_save_do_not_certify_six_stages(tmp_path):
    workspace = tmp_path / "workspace"
    audit = RevisionAudit(workspace, tmp_path / "audit")
    deck = workspace / "output/updated.pptx"
    write_deck(deck, SPEC["phases"]["Q"]["tasks"])
    audit.capture("end", "all", "execute")
    audit.capture("end", "check", "execute")
    result_files(tmp_path, deck)
    value = score(SPEC, "Q", tmp_path, tmp_path / "audit", SOURCE)
    assert value["final_correct"] == 6
    assert all(not x["stage_evidence"] for x in value["subtasks"][:5])


def test_failed_first_item_does_not_erase_later_completed_items(tmp_path):
    workspace = tmp_path / "workspace"
    audit = RevisionAudit(workspace, tmp_path / "audit")
    deck = workspace / "output/updated.pptx"
    tasks = SPEC["phases"]["Q"]["tasks"]
    for i in range(1, 6):
        write_deck(deck, tasks[2 : 2 * (i + 1)])
        audit.capture("end", f"edit-{i}", "execute")
        audit.capture("end", f"read-{i}", "execute")
    result_files(tmp_path, deck)
    value = score(SPEC, "Q", tmp_path, tmp_path / "audit", SOURCE)
    assert value["final_correct"] == 5 and value["subtasks"][0]["status"] == "failed"
    assert all(x["stage_evidence"] for x in value["subtasks"][1:])
