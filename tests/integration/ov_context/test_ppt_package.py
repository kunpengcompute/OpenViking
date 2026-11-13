# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Current PPT inputs, package parsing and Chat upload contracts."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.office.cases.ppt_update.files import load_case
from benchmark.office.cases.ppt_update.scoring import package_bytes, slide_parts, xml


async def test_upload_rejects_invalid_files_and_does_not_overwrite(chat: Any, skill: Path) -> None:
    client, runtime = chat
    runtime.settings = replace(runtime.settings, pptx_skill=skill)
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    prefix = f"/api/sessions/{sid}/inputs/"
    assert (await client.post(prefix + "broken.pptx", content=b"broken")).status_code == 400
    assert not (runtime.settings.data_dir / "inputs" / sid / "broken.pptx").exists()
    source = load_case()[1]
    assert (await client.post(prefix + source.name, content=source.read_bytes())).status_code == 200
    assert (await client.post(prefix + source.name, content=b"overwrite")).status_code == 409
    assert (await client.post(prefix + "script.py", content=b"print(1)")).status_code == 200
    assert (runtime.settings.data_dir / "inputs" / sid / "script.py").read_bytes() == b"print(1)"
    assert (await client.post(prefix + "script.exe", content=b"binary")).status_code == 400


def test_oracle_follows_presentation_slide_order() -> None:
    _, source = load_case()
    package = package_bytes(source)
    assert len(slide_parts(package)) == 14
    assert xml(package[slide_parts(package)[5]]) is not None
