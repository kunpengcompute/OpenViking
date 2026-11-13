"""Public benchmark entry points work independently of the Chat source directory."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmark.office.admin import datasets
from benchmark.paths import PROJECT_ROOT
from tests.st.ov_context import run as live


def test_office_admin_loads_dataset_assets_and_business_modules():
    assert datasets.check()["assets_ready"]
    assert callable(datasets.load("cases.portal.data").provision)
    assert callable(datasets.load("cases.schedule.data").inventory)
    assert callable(datasets.load("cases.timesheet.data").inventory)
    assert callable(datasets.load("cases.timesheet.app").handle)
    assert callable(datasets.load("cases.mail.data").check_export)
    assert callable(datasets.load("suites.schedule.scoring").check_at_deadline)
    assert callable(datasets.load("suites.timesheet.scoring").check)
    assert callable(datasets.load("suites.mail.scoring").check_export)
    assert callable(datasets.load("admin.mail").seed)


@pytest.mark.parametrize(
    "module",
    [
        "admin.datasets",
        "tasks",
        "cases.page_layer",
        "suites.run",
    ],
)
def test_office_cli_entry_point_help(module):
    result = subprocess.run(
        [sys.executable, "-m", f"benchmark.office.{module}", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("module", ["run", "validate_memory", "validate_multi_session"])
def test_system_test_cli_entry_point_help(module):
    result = subprocess.run(
        [sys.executable, "-m", f"tests.st.ov_context.{module}", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_removed_benchmark_entry_files_are_absent():
    for relative in (
        "suites/portal/run.py",
        "suites/office_cases/run.py",
        "suites/memory_comparison/__main__.py",
        "suites/memory_comparison/prepare.py",
        "suites/schedule/run.py",
        "suites/evaluation/run.py",
        "suites/integration/run.py",
    ):
        assert not (PROJECT_ROOT / "benchmark/office" / relative).is_file()


@pytest.mark.parametrize("destination", ["default", "external", "source"])
def test_live_runner_keeps_evidence_in_result_or_external_directory(
    tmp_path, monkeypatch, destination
):
    repo = tmp_path / "checkout"
    repo.mkdir()
    results = repo / "result"
    monkeypatch.setattr(live, "PROJECT_ROOT", repo)
    monkeypatch.setattr(live.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(live, "RESULT_ROOT", results)
    model = tmp_path / "model.json"
    model.write_text('{"chat_model": "fixture"}')
    key = tmp_path / "key"
    key.write_text("fixture")
    argv = [
        "live",
        "--ov-url",
        "http://localhost:1",
        "--api-key-file",
        str(key),
        "--model-config",
        str(model),
    ]
    if destination != "default":
        argv += [
            "--output-dir",
            str(repo / "benchmark" if destination == "source" else tmp_path / "outside"),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def start(*args, **kwargs):
        assert "tests/st/ov_context/live" in args[0]
        assert "tests/integration/ov_context/live" not in args[0]
        assert kwargs["env"]["OV_LIVE_COMPONENT_PARALLELISM"] == "4"
        assert kwargs["env"]["OV_LIVE_COMPONENT_TIMEOUT"] == "300"
        assert kwargs["env"]["OV_LIVE_API_KEY_FILE"] == str(key.resolve())
        calls.append(kwargs["cwd"])
        return SimpleNamespace(stdout=[], wait=lambda: 0)

    monkeypatch.setattr(live.subprocess, "Popen", start)
    if destination == "source":
        with pytest.raises(SystemExit) as exc:
            live.main()
        assert exc.value.code == 2 and not calls
        assert not (repo / "benchmark").exists()
    else:
        assert live.main() == 0
        root = results if destination == "default" else tmp_path / "outside"
        assert len(list(root.rglob("environment.json"))) == 1
        assert calls == [repo]
        metadata = __import__("json").loads(next(root.rglob("environment.json")).read_text())
        assert metadata["component_parallelism"] == 4
        assert metadata["component_timeout"] == 300


def test_retired_memory_suite_is_absent_from_delivery():
    root = PROJECT_ROOT / "benchmark/office/suites"
    assert not (root / "memory_comparison").exists()
    assert not (PROJECT_ROOT / "tests/st/ov_context/validate_components.py").exists()
    assert (root / "common/chat.py").is_file()
    assert (PROJECT_ROOT / "benchmark/office/cases/ppt_update/scoring.py").is_file()
    assert (PROJECT_ROOT / "benchmark/office/suites/reqdoc_generation/scoring.py").is_file()
