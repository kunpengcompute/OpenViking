"""Preserve actual PPT revisions and raw tool events for AI review."""

import hashlib
import json
from zipfile import ZipFile

from benchmark.office.cases.ppt_update.package import package_bytes
from benchmark.office.suites.ppt_update.observer import RevisionAudit
from benchmark.paths import PROJECT_ROOT
from tests.ov_context_support.ppt_fixture import expected_package

SPEC = json.loads((PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json").read_text())
SOURCE = PROJECT_ROOT / SPEC["source"]


def write_deck(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    package = expected_package(package_bytes(SOURCE), {"edits": tasks})
    with ZipFile(path, "w") as archive:
        for name, raw in package.items():
            archive.writestr(name, raw)


def test_six_revisions_keep_tool_ids_arguments_and_original_bytes(tmp_path):
    workspace = tmp_path / "workspace"
    folder = tmp_path / "audit"
    audit = RevisionAudit(workspace, folder)
    tasks = SPEC["phases"]["Q"]["tasks"]
    deck = workspace / "output/updated.pptx"
    for i in range(6):
        write_deck(deck, tasks[: 2 * (i + 1)])
        audit.capture("end", f"edit-{i}", "execute", {"command": f"edit {i}"})
    records = [json.loads(line) for line in (folder / "revisions.jsonl").read_text().splitlines()]
    assert len(records) == 6
    for i, row in enumerate(records):
        raw = (folder / (row["sha256"] + ".pptx")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert row["tool_call_id"] == f"edit-{i}"
    assert not (folder / "grade.json").exists()
