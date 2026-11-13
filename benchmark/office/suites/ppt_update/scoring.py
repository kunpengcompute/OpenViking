"""Separate content, sequence evidence and six independent scoring deadlines."""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

from benchmark.office.cases.ppt_update.scoring import (
    check_deck,
    expected_package,
    package_bytes,
    slide_parts,
    xml,
)
from benchmark.office.suites.common.timing import reviewed_timings, score_subtasks


def task_groups(tasks):
    groups = {}
    for i, task in enumerate(tasks):
        groups.setdefault(task.get("task_id", task["id"]), []).append(i)
    return groups


def stage_view(source: Path, output: Path, tasks, slides):
    detail = check_deck(source, output, {"edits": tasks, "slides": slides, "preserve": []})
    applied = [i for i, ok in enumerate(detail["edits"]) if ok]
    groups = task_groups(tasks)
    expected = {"edits": [tasks[i] for i in applied], "slides": slides, "preserve": []}
    detail2 = check_deck(source, output, expected)
    return {
        "applied": [i for i, indexes in enumerate(groups.values()) if set(indexes) <= set(applied)],
        "applied_edits": applied,
        "preservation_ok": not detail2["unexpected_parts"],
        "slides": detail["slide_count"],
    }


def latency_result(spec, phase, statistics, subtasks):
    value = score_subtasks("ppt_update", phase, statistics, subtasks)
    if spec["latency_budget_seconds"] != value["policy"]["budget_seconds"]:
        raise ValueError("PPT budget differs from the office scoring policy")
    return value


def _texts(package):
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    return [
        ["".join(p.itertext()) for p in xml(package[n]).findall(".//a:p", ns)]
        for n in slide_parts(package)
    ]


def _order(package):
    mapping = {
        e.get("Id"): e.get("Target") for e in xml(package["ppt/_rels/presentation.xml.rels"])
    }
    ids = xml(package["ppt/presentation.xml"]).findall(
        ".//{http://schemas.openxmlformats.org/presentationml/2006/main}sldId"
    )
    return [
        posixpath.normpath(
            posixpath.join(
                "ppt",
                mapping[
                    e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                ],
            )
        )
        for e in ids
    ]


def score(spec, phase, folder: Path, audit: Path, source: Path):
    tasks = spec["phases"][phase]["tasks"]
    groups = task_groups(tasks)
    original = package_bytes(source)
    expected = expected_package(original, {"edits": tasks})
    result = json.loads((folder / "result.json").read_text())
    answer = (folder / "answer.md").read_text() if (folder / "answer.md").exists() else ""
    candidates = []
    for receipt in result.get("artifacts", []):
        name = receipt["filename"]
        path = folder / "artifacts" / name
        if Path(name).name != name or path.suffix.lower() != ".pptx" or not path.is_file():
            continue
        try:
            final = stage_view(source, path, tasks, spec["slides"])
            package = package_bytes(path)
            final.update(
                filename=name,
                text_matches_expected=_texts(package) == _texts(expected),
                page_order_ok=_order(package) == _order(original),
                media_preserved=all(
                    package.get(n) == raw
                    for n, raw in original.items()
                    if n.startswith("ppt/media/")
                ),
                delivery_pass=receipt.get("status") == 200 and receipt["url"] in answer,
            )
            candidates.append(final)
        except Exception as exc:
            candidates.append({"filename": name, "error_type": type(exc).__name__})
    final = (
        max(candidates, key=lambda x: (len(x.get("applied", [])), x.get("delivery_pass", False)))
        if candidates
        else {}
    )
    states = {}
    for path in audit.glob("*.pptx"):
        try:
            states[path.stem] = stage_view(source, path, tasks, spec["slides"])
        except Exception as exc:
            states[path.stem] = {"error_type": type(exc).__name__}
    records = (
        [json.loads(line) for line in (audit / "revisions.jsonl").read_text().splitlines()]
        if (audit / "revisions.jsonl").exists()
        else []
    )
    rows, last_index = [], -1
    for i, (task_id, indexes) in enumerate(groups.items()):
        found = None
        for n, event in enumerate(records):
            if n <= last_index or event["event"] != "end":
                continue
            state = states.get(event.get("sha256"), {})
            if (
                i not in state.get("applied", [])
                or any(j > i for j in state["applied"])
                or not state.get("preservation_ok")
            ):
                continue
            for k in range(n + 1, len(records)):
                nxt = records[k]
                if nxt.get("sha256") != event["sha256"]:
                    break
                if (
                    nxt["event"] == "end"
                    and nxt["tool_call_id"] != event["tool_call_id"]
                    and nxt["tool"] in {"read_file", "execute"}
                ):
                    found = {
                        "revision": event["sha256"],
                        "saved_event": n,
                        "readback_event": k,
                        "readback_call": nxt["tool_call_id"],
                        "arguments": nxt.get("arguments"),
                    }
                    last_index = k
                    break
            if found:
                break
        correct = i in final.get("applied", [])
        rows.append(
            {
                "id": task_id,
                "slide": tasks[indexes[0]]["slide"],
                "final_edit_correct": correct,
                "stage_evidence": found,
                "status": "trace_review_required"
                if found and correct
                else "unverified"
                if correct
                else "failed",
            }
        )
    task_pass = (
        result.get("statistics", {}).get("status") == "done"
        and len(final.get("applied", [])) == len(groups)
        and final.get("slides") == spec["slides"]
        and all(
            final.get(k)
            for k in ["text_matches_expected", "page_order_ok", "media_preserved", "delivery_pass"]
        )
    )
    timed_rows = reviewed_timings(
        folder,
        [{"id": row["id"], "final_passed": row["final_edit_correct"]} for row in rows],
    )
    return {
        "version": spec["version"],
        "phase": phase,
        "final_correct": len(final.get("applied", [])),
        "total": len(groups),
        "correct_subitems": len(final.get("applied_edits", [])),
        "total_subitems": len(tasks),
        "final_preservation_ok": final.get("preservation_ok", False),
        "filename": final.get("filename"),
        "delivery_pass": final.get("delivery_pass", False),
        "final_text_matches_expected": final.get("text_matches_expected", False),
        "page_order_ok": final.get("page_order_ok", False),
        "media_preserved": final.get("media_preserved", False),
        "task_pass": bool(task_pass),
        "latency": latency_result(spec, phase, result.get("statistics", {}), timed_rows),
        "subtasks": rows,
        "sequence_auto_verified": False,
        "sequence_scope": "Tool/revision evidence requires review; final content does not certify order.",
        "visual_review": "excluded",
        "revision_count": len(states),
    }
