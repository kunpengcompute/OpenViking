"""Capture actual PPT revisions at tool boundaries, identically for both arms."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class RevisionAudit:
    def __init__(self, workspace: Path, evidence: Path):
        self.workspace, self.evidence = workspace, evidence
        self.started = time.monotonic()
        evidence.mkdir(parents=True, exist_ok=True)

    def capture(self, event: str, call_id: str, tool: str, arguments: Any = None):
        path = self.workspace / "output/updated.pptx"
        sha = None
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            target = self.evidence / (sha + ".pptx")
            if not target.exists():
                target.write_bytes(raw)
        value = {
            "event": event,
            "tool_call_id": call_id,
            "tool": tool,
            "arguments": arguments,
            "sha256": sha,
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
        }
        with (self.evidence / "revisions.jsonl").open("a") as out:
            out.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
