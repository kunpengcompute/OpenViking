"""Task text, prepared data, and the Chat request share one H/Q contract."""

import hashlib
import io
import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmark.office.admin import datasets
from benchmark.office.cases.portal import data
from benchmark.office.cases.timesheet.data import inventory as timesheet_inventory
from benchmark.office.datasets.mail import data as mail
from benchmark.office.datasets.timesheet import data as timesheet
from benchmark.office.suites import run
from benchmark.office.suites.mail.scoring import check_export
from benchmark.office.suites.schedule.scoring import check as check_schedule
from benchmark.office.suites.timesheet.scoring import check as check_timesheet
from benchmark.office.tasks import CASES, DEFAULT_ANCHORS, PHASES, render_task, task_path
from benchmark.paths import PROJECT_ROOT


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("phase", PHASES)
def test_templates_render_the_same_text_for_manual_cli_and_runner(case, phase):
    config = {
        "task_case": case,
        "task_phase": phase,
        "task_version": "fixture",
        "mail_ready": True,
        "anchor": DEFAULT_ANCHORS[case][phase],
    }
    expected = render_task(case, phase, workspace="task-check", page_url="http://localhost:1234")
    assert "${" not in expected and "WORKSPACE_TO_BE_CREATED" not in expected
    assert "http://localhost:1234/mock/login?workspace=task-check" in expected
    assert run.task_for_workspace(case, phase, "task-check", "http://localhost:1234", config) == (
        phase,
        expected,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.office.tasks",
            "--case",
            case,
            "--phase",
            phase,
            "--workspace",
            "task-check",
            "--page-url",
            "http://localhost:1234",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("case", CASES)
def test_training_guidance_only_appears_in_H(case):
    h = render_task(case, "H", workspace="h-check")
    q = render_task(case, "Q", workspace="q-check")
    assert "执行方法指导" in h if case != "timesheet" else "执行时请采用以下具体方式" in h
    assert "执行方法指导" not in q and "执行时请采用以下具体方式" not in q


@pytest.mark.parametrize("case,phase", [("../mail", "H"), ("mail", "../Q"), ("unknown", "Q")])
def test_task_lookup_cannot_escape_the_template_directory(case, phase):
    with pytest.raises(ValueError):
        task_path(case, phase)


@pytest.mark.parametrize("phase", PHASES)
def test_mail_targets_match_visible_selection_and_exact_exports(tmp_path, phase):
    rows = mail.phase_rows(phase)
    assert len(rows) == 64 and len({row["id"] for row in rows}) == 64
    query = render_task("mail", phase, workspace="mail-check")
    goals = mail.goals(phase)
    assert len(goals) == (3 if phase == "H" else 6)
    for index, goal in enumerate(goals, 1):
        target = goal["message"]
        assert target["subject"] in query and goal["filename"] in query
        candidates = [
            r
            for r in rows
            if r["subject"] == target["subject"] and r["from_name"] == target["from_name"]
        ]
        if phase == "H" and index in (1, 2):
            candidates = [r for r in candidates if r["date"][:10] == target["date"][:10]]
        if phase == "Q" and index == 3:
            candidates = [r for r in candidates if r["date"][:10] == target["date"][:10]]
        if (phase == "Q" and index == 4) or (phase == "H" and index == 3):
            candidates = [max(candidates, key=lambda r: r["date"])]
        assert [row["id"] for row in candidates] == [target["id"]]
        path = tmp_path / goal["filename"]
        path.write_text(target["body"])
        assert all(check_export(path, phase, index).values())
        path.write_text(target["body"][:100])
        assert not check_export(path, phase, index)["body_complete"]
    if phase == "Q":
        assert len(goals[-1]["message"]["body"]) > 3000


def test_mail_H_contains_the_frozen_sort_and_stop_guidance():
    prompt = render_task("mail", "H", workspace="mail-h-check")
    assert "以下三封邮件" in prompt and "三项结果清单" in prompt
    assert "history_03.txt" in prompt and "history_04.txt" not in prompt
    assert "2026-10-20" in prompt and "2026-10-19" in prompt
    for method in ("一次 browser eval", "最新在前", "停止条件：", "翻页条件：", "当前DOM"):
        assert method in prompt
    assert mail.goals("H")[2]["message"]["subject"] == "历史版本变更说明"


def test_mail_H_and_Q_do_not_share_identifiers_or_bodies():
    h, q = mail.phase_rows("H"), mail.phase_rows("Q")
    assert not {r["id"] for r in h} & {r["id"] for r in q}
    assert not {r["body"] for r in h} & {r["body"] for r in q}
    assert not {r["message"]["subject"] for r in mail.goals("H")} & {
        r["message"]["subject"] for r in mail.goals("Q")
    }


@pytest.mark.parametrize("phase", PHASES)
def test_timesheet_targets_match_prompts_and_require_one_correct_increment(tmp_path, phase):
    registry = tmp_path / "registry"
    data.provision(registry, "time-check", anchor=DEFAULT_ANCHORS["timesheet"][phase])
    before = timesheet_inventory(registry, "time-check")
    after = deepcopy(before)
    query = render_task("timesheet", phase, workspace="time-check")
    for index, target in enumerate(timesheet.goals(phase), 1):
        assert target["content"] in query
        after["entries"].append({**target, "id": 1000 + index})
    result = check_timesheet(phase, before, after)
    assert result["all_backend_requirements_met"]
    assert (
        result["completed"] == 6 and result["new_entries"] == 6 and result["background_unchanged"]
    )
    after["entries"].append({**timesheet.goals(phase)[0], "id": 2000})
    assert check_timesheet(phase, before, after)["completed"] == 5
    assert not check_timesheet(phase, before, after)["all_backend_requirements_met"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("phase", PHASES)
def test_preparation_sets_phase_and_keeps_reference_out_of_browser_config(tmp_path, case, phase):
    registry = tmp_path / "registry"
    registry.mkdir()
    output = tmp_path / "reference.json"
    result = datasets.prepare(
        SimpleNamespace(
            registry=registry,
            namespace="prepared-check",
            phase=phase,
            anchor=None,
            case=case,
            page_url="http://localhost:4567",
            mail_settings=None,
            output=output,
        )
    )
    config = json.loads((registry / "prepared-check.json").read_text())
    reference = json.loads(output.read_text())
    assert result["case"] == config["task_case"] == case and config["task_phase"] == phase
    assert reference["task"] == render_task(
        case,
        phase,
        workspace="prepared-check",
        page_url="http://localhost:4567",
        anchor=config["anchor"],
    )
    assert "goals" not in config and "body" not in config and output.stat().st_mode & 0o777 == 0o600
    if case == "schedule":
        assert len(reference["reference"]["targets"]) == 6
        assert not check_schedule(
            reference["reference"],
            reference["reference"]["before"]
            | {"resources": reference["reference"]["resources_before"]},
        )["completed"]
    else:
        assert len(reference["goals"]) == (3 if case == "mail" and phase == "H" else 6)
    if case == "mail":
        assert config["mail_ready"] is False
        with pytest.raises(ValueError, match="mail-settings"):
            run.task_for_workspace(case, phase, "prepared-check", "http://localhost:4567", config)


def test_preparation_imports_the_mailbox_with_three_H_targets(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    registry.mkdir()
    settings = tmp_path / "mail.local.json"
    settings.write_text(
        json.dumps(
            {
                "username": "case@example.test",
                "password": "fixture",
                "imap_host": "example.test",
                "imap_port": 143,
                "webmail_url": "http://example.test",
            }
        )
    )
    original_load = datasets.load
    imported = []

    def load(name):
        if name == "admin.mail":
            return SimpleNamespace(
                seed=lambda settings, snapshot, fixture: imported.append((snapshot, fixture))
            )
        return original_load(name)

    monkeypatch.setattr(datasets, "load", load)
    datasets.prepare(
        SimpleNamespace(
            registry=registry,
            namespace="mail-ready",
            phase="H",
            anchor=None,
            case="mail",
            page_url="http://localhost:4567",
            mail_settings=settings,
            output=tmp_path / "reference.json",
        )
    )
    assert imported[0][0] == "history" and len(imported[0][1]["snapshots"]["history"]) == 64
    config = json.loads((registry / "mail-ready.json").read_text())
    assert config["mail_ready"] is True and config["mail_read"]["target_count"] == 3


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("phase", PHASES)
def test_chat_endpoint_receives_the_central_rendered_task(tmp_path, monkeypatch, case, phase):
    registry = tmp_path / "registry"
    registry.mkdir()
    config = {
        "namespace": "run-check",
        "case": "office",
        "task_case": case,
        "task_phase": phase,
        "task_version": "fixture",
        "mail_ready": True,
        "anchor": DEFAULT_ANCHORS[case][phase],
        "employee": {"number": "100001", "name": "林澈", "department": "产品部"},
    }
    (registry / "run-check.json").write_text(json.dumps(config))
    sent = []

    def response(request, timeout):
        if request.full_url.endswith("/api/execution-profiles"):
            return io.BytesIO(b'{"automatic_benchmark":true}')
        if request.full_url.endswith("/api/chat"):
            sent.append(json.loads(request.data))
            return io.BytesIO(
                b'event: start\ndata: {"run_id":"test-run"}\n\nevent: done\ndata: {}\n\n'
            )
        if request.full_url.endswith("/benchmark"):
            return io.BytesIO(
                json.dumps({"automatic": True, "case": case, "phase": phase}).encode()
            )
        if request.full_url.endswith("/runs/test-run"):
            return io.BytesIO(b'{"statistics":{"status":"done","seconds":1}}')
        return io.BytesIO(b'{"session_id":"test-session"}' if request.data else b"[]")

    monkeypatch.setattr(run.urllib.request, "urlopen", response)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "office-run",
            "--case",
            case,
            "--phase",
            phase,
            "--workspace",
            "run-check",
            "--registry",
            str(registry),
            "--output",
            str(tmp_path / "result"),
        ],
    )
    run.main()
    assert sent == [
        {"session_id": "test-session", "message": render_task(case, phase, workspace="run-check")}
    ]


def test_runner_rejects_old_workspaces_and_phase_mismatches():
    with pytest.raises(ValueError, match="prepare --case"):
        run.task_for_workspace(
            "schedule", "Q", "old", "http://localhost", {"anchor": "2026-10-31T09:00+08:00"}
        )
    with pytest.raises(ValueError, match="phase"):
        run.task_for_workspace(
            "schedule",
            "Q",
            "wrong-phase",
            "http://localhost",
            {"task_case": "schedule", "task_phase": "H"},
        )


def test_Q_mail_fixture_preserves_the_frozen_six_target_dataset():
    digest = hashlib.sha256(
        json.dumps(mail.phase_rows("Q"), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    assert digest == "95a420795fb83f9ab9f33408866936f7db349bcd54020e79f4f333e1486f66dd"
