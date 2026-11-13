"""Fixed PPT inputs and provenance, never provided with the scoring oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.office.suites.common.files_config import write_json
from benchmark.paths import OFFICE_ROOT, PROJECT_ROOT

CASE_PATH = OFFICE_ROOT / "cases/ppt_update/spec.json"


def load_case(path: Path = CASE_PATH) -> tuple[dict[str, Any], Path]:
    case = json.loads(path.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / case["source"]
    if (
        source.is_symlink()
        or hashlib.sha256(source.read_bytes()).hexdigest() != case["input_sha256"]
    ):
        raise ValueError("PPT input differs from the fixed case SHA256")
    return case, source


async def collect_artifacts(
    client: Any, sid: str, run_id: str, folder: Path
) -> list[dict[str, Any]]:
    suffix = "-recovery" if (folder / "artifact-downloads.json").exists() else ""
    response = await client.get(f"/api/sessions/{sid}/artifacts")
    write_json(
        folder / f"artifact-list-response{suffix}.json",
        {"status": response.status_code, "body": response.text},
    )
    response.raise_for_status()
    records = []
    journal = folder / f"artifact-downloads{suffix}.json"
    write_json(journal, records)
    for item in response.json():
        if item["run_id"] != run_id:
            continue
        name = item["filename"]
        if Path(name).name != name:
            raise ValueError("Unexpected artifact filename")
        url = f"/api/sessions/{sid}/artifacts/{run_id}/{name}"
        record = {"filename": name, "url": url, "status": None}
        records.append(record)
        try:
            downloaded = await client.get(url)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            write_json(journal, records)
            raise
        record["status"] = downloaded.status_code
        if downloaded.status_code == 200:
            (folder / "artifacts").mkdir(exist_ok=True)
            (folder / "artifacts" / name).write_bytes(downloaded.content)
            record.update(
                bytes=len(downloaded.content), sha256=hashlib.sha256(downloaded.content).hexdigest()
            )
        write_json(journal, records)
    return records


def export_tool_evidence(folder: Path, trace: dict[str, Any]) -> None:
    """Keep declared calls, raw execution events and model-visible receipts side by side."""
    calls: dict[str, dict[str, Any]] = {}
    executions = []
    for event in trace["events"]:
        data = event["data"]
        if event["event"] == "stage" and data.get("name", "").startswith("tool."):
            executions.append(event)
        if event["event"] != "message":
            continue
        message = data["message"]
        if message["type"] == "ai":
            for call in message.get("tool_calls", []):
                calls[call["id"]] = {
                    "tool_call_id": call["id"],
                    "tool": call["name"],
                    "arguments": call["args"],
                    "declared_at_ms": event.get("elapsed_ms"),
                    "receipts": [],
                    "execution_events": [],
                }
        elif message["type"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id in calls:
                calls[call_id]["receipts"].append(message)
    for event in executions:
        call_id = event["data"].get("tool_call_id")
        if call_id in calls:
            calls[call_id]["execution_events"].append(event)
    write_json(folder / "tool-calls.json", list(calls.values()))
    write_json(folder / "tool-events.json", executions)
