# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read-only benchmark snapshots, isolated from the Agent's visible workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from .files_config import write_json
from .timing import scoring_policy

EXPERIMENT = re.compile(
    r"office-auto--(schedule|timesheet|mail|ppt_update|reqdoc_generation)--([HQ])"
)


class RunAudit:
    def __init__(
        self, workspace: Path, folder: Path, case: str, phase: str, *, reference=None, started=None
    ):
        self.workspace, self.folder = workspace, folder
        self.case, self.phase = case, phase
        self.started = started
        self.reference = reference or {}
        self.cache = {}
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        (folder / "blobs").mkdir(exist_ok=True, mode=0o700)
        if case == "ppt_update":
            from benchmark.office.cases.ppt_update.files import load_case

            spec, source = load_case()
            self.reference = {"spec": spec, "source": self.blob(source.read_bytes())}
        write_json(
            folder / "manifest.json",
            {
                "case": case,
                "phase": phase,
                "reference": self.reference,
                "scoring_policy": scoring_policy(case),
                "schema_version": 1,
            },
        )

    def blob(self, raw: bytes) -> str:
        digest = hashlib.sha256(raw).hexdigest()
        target = self.folder / "blobs" / digest
        if not target.exists():
            target.write_bytes(raw)
        return digest

    def business(self):
        if self.case not in {"schedule", "timesheet"}:
            return None
        from benchmark.office.cases.schedule.resource_case import inventory as calendar
        from benchmark.office.cases.timesheet.data import inventory as worklogs

        registry = Path(self.reference["registry"])
        ns = self.reference["workspace"]
        return calendar(registry, ns) if self.case == "schedule" else worklogs(registry, ns)

    def capture(self, elapsed_seconds: float, event: str, **fields):
        row = {"elapsed_seconds": elapsed_seconds, "event": event, **fields}
        try:
            files = {}
            suffixes = {
                "ppt_update": {".pptx"},
                "reqdoc_generation": {".md", ".json", ".py"},
                "mail": {".txt"},
                "timesheet": {".png", ".txt"},
                "schedule": set(),
            }[self.case]
            for directory in ("output", "tmp"):
                root = self.workspace / directory
                if root.is_symlink():
                    raise ValueError("Benchmark workspace directory is a symlink")
                for path in sorted(root.rglob("*")):
                    if path.suffix not in suffixes or not path.is_file():
                        continue
                    relative = path.relative_to(self.workspace)
                    if path.is_symlink() or any(
                        (self.workspace / p).is_symlink() for p in relative.parents
                    ):
                        raise ValueError("Benchmark artifact is a symlink")
                    stat = path.stat()
                    if stat.st_size > 32 * 1024 * 1024 or len(files) >= 512:
                        raise ValueError("Benchmark snapshot limit exceeded")
                    key = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    cached = self.cache.get(str(relative))
                    if cached is None or cached[0] != key:
                        raw = path.read_bytes()
                        after = path.stat()
                        if (
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        ) != key:
                            raise ValueError("Artifact changed while taking snapshot")
                        cached = (key, self.blob(raw))
                        self.cache[str(relative)] = cached
                    files[str(relative)] = cached[1]
            row.update(files=files, business=self.business())
            if self.case == "mail":
                from benchmark.office.datasets.mail.data import goals
                from examples.deepagents_ov_context.chat.artifacts import (
                    FileArtifact,
                    valid_artifact_name,
                )

                published = {}
                for target in goals(self.phase):
                    name = target["filename"]
                    receipt = self.workspace / (name + ".artifact.json")
                    path = self.workspace / name
                    if not receipt.exists():
                        continue
                    try:
                        if (
                            receipt.is_symlink()
                            or path.is_symlink()
                            or not valid_artifact_name(name)
                        ):
                            raise ValueError("Invalid published mail path")
                        artifact = FileArtifact.model_validate_json(receipt.read_text())
                        raw = path.read_bytes()
                        if artifact.filename != name or artifact.bytes != len(raw):
                            raise ValueError("Published mail differs from its receipt")
                        published[name] = self.blob(raw)
                    except (ValueError, OSError):
                        published[name] = None
                row["published"] = published
        except Exception as exc:
            row["capture_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if self.started is not None and event != "run.end":
            # Use the completed observation, never a time before the captured file existed.
            row["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        with (self.folder / "snapshots.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def create_audit(data_dir: Path, session, run) -> RunAudit | None:
    match = EXPERIMENT.fullmatch(session.experiment_id or "")
    if match is None:
        return None
    case, phase = match.groups()
    reference = {}
    if case in {"schedule", "timesheet", "mail"}:
        from benchmark.office.cases.portal.app import OfficePortal
        from benchmark.office.cases.schedule.resource_case import inventory as calendar
        from benchmark.office.cases.timesheet.data import inventory as worklogs

        registry = Path(os.environ.get("OV_BENCHMARK_REGISTRY", data_dir.parent / "cases/registry"))
        config = OfficePortal(registry).config(session.user_id)
        if config.get("task_case") != case or config.get("task_phase") != phase:
            raise ValueError("Benchmark Session and prepared workspace differ")
        reference = {"registry": str(registry), "workspace": session.user_id}
        if case == "timesheet":
            reference["before"] = worklogs(registry, session.user_id)
        if case == "schedule":
            from benchmark.office.cases.schedule.resource_six_case import prepare

            # Derive the frozen targets in a disposable fixture; never modify the live workspace.
            with tempfile.TemporaryDirectory(prefix="office-targets-") as temp:
                target = prepare(Path(temp), "reference")
            before = calendar(registry, session.user_id)
            target.update(before=before, resources_before=before["resources"])
            reference["calendar"] = target
    audit = RunAudit(
        data_dir / "artifacts" / session.session_id / run.run_id,
        data_dir / "benchmark" / session.session_id / run.run_id,
        case,
        phase,
        reference=reference,
        started=run.started,
    )
    return audit
