# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Replay immutable revisions through case validators; no manual timing map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .audit import RunAudit as RunAudit
from .files_config import write_json
from .timing import score_subtasks, valid_seconds


def blob(folder, digest):
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid snapshot blob hash")
    path = folder / "blobs" / digest
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Snapshot blob failed integrity check")
    return path


def ppt_states(folder, manifest, snapshot):
    from zipfile import ZipFile

    from benchmark.office.suites.ppt_update.scoring import stage_view

    reference = manifest["reference"]
    spec = reference["spec"]
    source = blob(folder, reference["source"])
    tasks = spec["phases"][manifest["phase"]]["tasks"]
    candidates = [
        blob(folder, sha)
        for path, sha in snapshot["files"].items()
        if path.startswith("output/") and path.endswith(".pptx")
    ]
    if not candidates:
        return [False] * 6
    if len(candidates) != 1:
        raise ValueError("Ambiguous PPT output candidates")
    with ZipFile(candidates[0]) as archive:
        infos = archive.infolist()
        if len(infos) > 4096 or sum(item.file_size for item in infos) > 128 * 1024 * 1024:
            raise ValueError("PPT package exceeds the validation limit")
    state = stage_view(source, candidates[0], tasks, spec["slides"])
    if not state["preservation_ok"] or state["slides"] != spec["slides"]:
        return [False] * 6
    return [index in state["applied"] for index in range(6)]


async def evaluate_audit(folder: Path, statistics: dict, *, judge=None) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    records = [json.loads(line) for line in (folder / "snapshots.jsonl").read_text().splitlines()]
    policy = manifest["scoring_policy"]
    count = (
        policy.get("preparation_task_counts", {}).get(manifest["case"], policy["task_count"])
        if manifest["phase"] == "H"
        else policy["task_count"]
    )
    times, final, evidence = [None] * count, [None] * count, [None] * count
    errors = []
    previous = -1
    for ordinal, snapshot in enumerate(records):
        seconds = snapshot["elapsed_seconds"]
        if not valid_seconds(seconds) or seconds < previous:
            raise ValueError("Audit times must use a monotonic Run clock")
        previous = seconds
        try:
            if "capture_error" in snapshot:
                raise ValueError(str(snapshot["capture_error"]))
            case = manifest["case"]
            if case == "ppt_update":
                states = ppt_states(folder, manifest, snapshot)
            else:
                states = await case_states(folder, manifest, snapshot, judge)
            errors.extend(
                {"snapshot": ordinal, **item} for item in snapshot.get("validation_errors", [])
            )
        except Exception as exc:
            errors.append({"snapshot": ordinal, "type": type(exc).__name__, "message": str(exc)})
            states = [None] * count
        if len(states) != count or any(type(s) is not bool and s is not None for s in states):
            raise ValueError(f"Validator must return {count} boolean or unknown task states")
        for index, passed in enumerate(states):
            if passed is not True:
                times[index] = evidence[index] = None
            elif final[index] is not True:
                times[index] = seconds
                evidence[index] = f"snapshots.jsonl:{ordinal + 1} {snapshot['event']}"
            final[index] = passed
    rows = [
        {
            "id": f"{manifest['case']}-{index + 1}",
            "final_passed": passed,
            "completed_seconds": times[index],
            "evidence": evidence[index],
        }
        for index, passed in enumerate(final)
    ]
    if not records or records[-1]["event"] != "run.end":
        errors.append({"type": "MissingTerminalSnapshot", "message": "Run evidence is incomplete"})
        rows = [{"id": row["id"], "final_passed": None, "completed_seconds": None} for row in rows]
    value = {
        "case": manifest["case"],
        "phase": manifest["phase"],
        "automatic": True,
        "snapshots": len(records),
        "errors": errors,
        "latency": score_subtasks(
            manifest["case"], manifest["phase"], statistics, rows, policy=manifest["scoring_policy"]
        ),
    }
    usages = [
        json.loads(path.read_text()).get("usage", {})
        for path in (folder / "judgments").glob("*.json")
    ]
    value["judge_usage"] = {
        name: sum(u.get(name, 0) or 0 for u in usages)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    value["judge_usage"]["scope"] = "Evaluation only; excluded from Agent and OV task usage"
    if manifest["case"] == "reqdoc_generation":
        value["document_judgments"] = {
            path.stem: json.loads(path.read_text())
            for path in (folder / "judgments").glob("*.json")
        }
        reviews = folder / "document-current-reviews.json"
        if reviews.exists():
            review = json.loads(reviews.read_text())
            if records and review["document_sha256"] == records[-1].get("files", {}).get(
                "output/requirements.md"
            ):
                value["document_review"] = review
    write_json(folder / "grade.json", value)
    return value


async def case_states(folder, manifest, snapshot, judge):
    case, phase = manifest["case"], manifest["phase"]
    if case == "schedule":
        from benchmark.office.suites.schedule.scoring import check

        state = snapshot["business"]
        result = check(manifest["reference"]["calendar"], state)
        reads = {row["event_id"] for row in state["audit"] if row["action"] == "detail_read"}
        return [
            bool(
                item["completed"]
                and set(item["actual_ids"]) <= reads
                and result["background_unchanged"]
                and result["resource_background_unchanged"]
            )
            for item in result["items"]
        ]
    if case == "timesheet":
        from PIL import Image

        from benchmark.office.suites.timesheet.scoring import check

        result = check(phase, manifest["reference"]["before"], snapshot["business"])
        states = []
        for index, item in enumerate(result["items"], 1):
            digest = snapshot["files"].get(f"output/timesheet_verification_{index:02}.png")
            valid = False
            if digest:
                try:
                    with Image.open(blob(folder, digest)) as image:
                        valid = image.format == "PNG"
                        image.verify()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    snapshot.setdefault("validation_errors", []).append(
                        {
                            "task": index,
                            "type": type(exc).__name__,
                            "message": "Invalid screenshot evidence",
                        }
                    )
                    states.append(None)
                    continue
            states.append(
                bool(
                    item["saved"]
                    and item["delta_correct"]
                    and result["background_unchanged"]
                    and valid
                )
            )
        return states
    if case == "mail":
        from benchmark.office.datasets.mail.data import goals

        def normalize(text):
            return "".join(text.split())

        states = []
        for target in goals(phase):
            name = target["filename"]
            if name in snapshot.get("published", {}) and snapshot["published"][name] is None:
                states.append(None)
                continue
            digest = snapshot.get("published", {}).get(name)
            states.append(
                bool(
                    digest
                    and normalize(blob(folder, digest).read_text())
                    == normalize(target["message"]["body"])
                )
            )
        return states
    if case == "reqdoc_generation":
        from .document_judge import document_states

        return await document_states(folder, manifest, snapshot, judge)
    raise ValueError(f"Unknown benchmark case: {case}")
