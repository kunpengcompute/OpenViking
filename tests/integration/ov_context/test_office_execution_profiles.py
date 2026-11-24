# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Office runners submit the recorded formal settings to the shared Chat API."""

import importlib
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmark.office.suites.common.execution import load_execution_profile
from benchmark.paths import OFFICE_ROOT, PROJECT_ROOT

CASES = ("schedule", "timesheet", "mail", "ppt_update", "reqdoc_generation")


@pytest.mark.parametrize("case", CASES)
def test_formal_profiles_preserve_shared_and_case_specific_settings(case):
    profile = load_execution_profile(OFFICE_ROOT / "cases" / case / "execution.json")
    assert profile["max_tokens"] == 32768
    assert profile["recursion_limit"] == 1000
    assert profile["thinking"] is True
    assert profile["timeout_seconds"] == {
        "ppt_update": 1800,
        "reqdoc_generation": 3600,
    }.get(case, 900)
    if case in {"ppt_update", "reqdoc_generation"}:
        assert profile["context_window"] == 256000
        assert profile["model_call_timeout_seconds"] == 180
        assert profile["memory_detail"] == "full"
        assert profile["memory_budget_tokens"] == (8000 if case == "ppt_update" else 4000)
    else:
        assert "context_window" not in profile
    if case == "reqdoc_generation":
        assert profile["candidate_limit"] == 30


@pytest.mark.parametrize("payload", ['{"max_tokens":0}', '{"unknown":1}', "[]"])
def test_invalid_profile_is_rejected_before_execution(tmp_path, payload):
    path = tmp_path / "execution.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_execution_profile(path)


class SessionCaptured(Exception):
    """Stop before invoking a model; the test observes only the submitted settings."""


@pytest.mark.parametrize("case", ["schedule", "timesheet", "mail"])
def test_page_runner_submits_profile_and_saves_session_snapshot(tmp_path, monkeypatch, case):
    module = importlib.import_module("benchmark.office.suites.run")
    portal = importlib.import_module("benchmark.office.cases.portal.app")
    config = {
        "employee": {"number": "001", "name": "synthetic", "department": "test"},
        "anchor": "2026-11-19T09:00:00+08:00",
        "task_version": "test",
    }
    monkeypatch.setattr(portal, "OfficePortal", lambda _: SimpleNamespace(config=lambda _: config))
    monkeypatch.setattr(module, "task_for_workspace", lambda *args: ("Q", "test task"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "office",
            "--case",
            case,
            "--workspace",
            "test-q",
            "--registry",
            str(tmp_path / "registry"),
            "--output",
            str(tmp_path),
        ],
    )
    submitted = []
    snapshot = {"execution": {"profile": {"max_tokens": 32768}}}

    def respond(request, **kwargs):
        if request.full_url.endswith("/api/chat"):
            raise SessionCaptured()
        if request.data:
            submitted.append(json.loads(request.data))
            body = {"session_id": "test-session", "config_snapshot": snapshot}
        elif request.full_url.endswith("/api/sessions"):
            body = []
        else:
            body = {"benchmark_evidence": True}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    with pytest.raises(SessionCaptured):
        module.main()
    expected = load_execution_profile(OFFICE_ROOT / "cases" / case / "execution.json")
    assert submitted[0]["execution"] == expected | {
        "mode": "baseline",
        "system_prompt": submitted[0]["execution"]["system_prompt"],
    }
    folder = next(tmp_path.glob(f"{case}-baseline-*"))
    assert json.loads((folder / "session.json").read_text())["config_snapshot"] == snapshot
    condition = json.loads((folder / "condition.json").read_text())
    assert condition["execution"] == submitted[0]["execution"]
    assert (PROJECT_ROOT / condition["execution_profile"]).is_file()


@pytest.mark.parametrize("case", ["ppt_update", "reqdoc_generation"])
async def test_file_runner_submits_profile_to_chat(tmp_path, monkeypatch, case):
    module = importlib.import_module(f"benchmark.office.suites.{case}.run")
    spec_path = OFFICE_ROOT / "cases" / case / "spec.json"
    spec = json.loads(spec_path.read_text())
    actual_client = httpx.AsyncClient
    submitted = []

    def respond(request):
        if request.method == "POST":
            submitted.append(json.loads(request.content))
            raise SessionCaptured()
        return httpx.Response(
            200,
            json={
                "benchmark_evidence": True,
                "account_id": "test-account",
                "ov_url": "http://ov",
                "skills": {"pptx": {}, "office-documents": {}, "data-analysis": {}},
                "reader_defaults": spec.get("reader_contract", {}),
            },
        )

    def client(**kwargs):
        return actual_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    args = SimpleNamespace(
        spec=spec_path,
        phase="Q",
        mode="baseline",
        expected_memory=None,
        output=tmp_path / case,
        account="test-account",
        user="test-user",
        peer="test-peer",
        environment_id="daily",
        chat_url="http://chat",
        ov_url="http://ov",
        recall_retry_attempts=2,
    )
    with pytest.raises(SessionCaptured):
        await module.run(args)
    expected = load_execution_profile(PROJECT_ROOT / spec["execution_profile"])
    assert {key: submitted[0]["execution"][key] for key in expected} == expected
    assert submitted[0]["execution"]["mode"] == "baseline"
    assert submitted[0]["execution"]["memory_extraction"] is False
