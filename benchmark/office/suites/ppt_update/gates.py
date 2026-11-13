"""Check frozen case assets and real first model input for an H/Q sample."""

from __future__ import annotations

import hashlib
import json

from benchmark.office.suites.common.files_config import write_json
from benchmark.paths import OFFICE_ROOT, PROJECT_ROOT
from examples.deepagents_ov_context.chat.artifacts import valid_artifact_name


class CaseContract:
    """Bind H acceptance to the shared inputs and phase instructions."""

    def __init__(self, files):
        self.sha256 = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_case(cls, spec=None):
        if spec is None:
            spec = json.loads((OFFICE_ROOT / "cases/ppt_update/spec.json").read_text())
        files = {}
        for item in [*spec["phases"].values(), *spec["support_files"]]:
            name = item.get("prompt", item.get("source"))
            path = PROJECT_ROOT / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_symlink() or digest != item.get("prompt_sha256", item.get("sha256")):
                raise ValueError("PPT prompt/helper differs from the frozen case")
            if "filename" in item and not valid_artifact_name(item["filename"]):
                raise ValueError("Unexpected helper input filename")
            files[name] = digest
        return cls({"version": spec["version"], "files": files})

    def require_accepted(self, frozen):
        acceptance = frozen.get("artifact_acceptance", {})
        if acceptance.get("passed") is not True or acceptance.get("contract_sha256") != self.sha256:
            raise ValueError("Frozen H memory has no matching artifact acceptance")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


class FirstInputGate:
    def __init__(self, folder, prompt, system_prompt, phase, mode):
        self.folder, self.prompt, self.system_prompt = folder, prompt, system_prompt
        self.phase, self.mode = phase, mode
        self.recalled = []
        self.checked = False

    def __call__(self, event):
        data = event["data"]
        if event["event"] == "stage" and data.get("name") == "memory_recall_result":
            if data.get("count", 0) > 0 and data.get("content", "").strip():
                self.recalled.append(data["content"])
        if event["event"] != "model_input" or self.checked:
            return
        self.checked = True
        observed = "\n".join(strings(data.get("messages", [])))
        injected = any(text in observed for text in self.recalled)
        checks = {
            "task_preserved": self.prompt.strip() in observed,
            "execution_rules_present": self.system_prompt.strip() in observed,
            "memory_matches_mode": injected == (self.phase == "Q" and self.mode == "ordinary"),
        }
        write_json(self.folder / "first-input-gate.json", checks)
        if not all(checks.values()):
            raise RuntimeError("PPT first model input control gate failed")
