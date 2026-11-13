# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Validate six business materials and the shared helper uploaded in both arms."""

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.paths import OFFICE_ROOT, PROJECT_ROOT

CASE_PATH = OFFICE_ROOT / "cases/reqdoc_generation/spec.json"
INPUT_NAMES = {
    "business_background.md",
    "requirements_confirmed.md",
    "change_log.docx",
    "api_contract.json",
    "metrics.xlsx",
    "prototype.pdf",
    "record_helpers.py",
}


def load_case(path: Path = CASE_PATH) -> tuple[dict[str, Any], list[Path]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec["output"] != "requirements.md":
        raise ValueError("The frozen case delivers one requirements document")
    sources = []
    for item in spec["inputs"]:
        name = item["filename"]
        if name not in INPUT_NAMES:
            raise ValueError("Input must be one of the allowlisted case filenames")
        source = PROJECT_ROOT / spec["input_dir"] / name
        if (
            source.parent.is_symlink()
            or source.is_symlink()
            or (hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"])
        ):
            raise ValueError(f"Fixed input hash mismatch: {name}")
        sources.append(source)
    if len(sources) != len(INPUT_NAMES) or {p.name for p in sources} != INPUT_NAMES:
        raise ValueError("Six business materials and the shared helper are required")
    if set(spec["phases"]) != {"H", "Q"}:
        raise ValueError("The frozen case requires H and Q prompts")
    for phase in spec["phases"].values():
        prompt = PROJECT_ROOT / phase["prompt"]
        if (
            prompt.is_symlink()
            or hashlib.sha256(prompt.read_bytes()).hexdigest() != phase["sha256"]
        ):
            raise ValueError("Fixed task prompt hash mismatch")
    return spec, sources
