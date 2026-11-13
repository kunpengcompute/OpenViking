"""Exercise dataset preparation through the same administrator CLI used by containers."""

import json
import stat
import subprocess
import sys

import pytest

from benchmark.paths import PROJECT_ROOT


def datasets(*args):
    return subprocess.run(
        [sys.executable, "-m", "benchmark.office.admin.datasets", *map(str, args)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("phase", ["H", "Q"])
def test_prepare_and_inventory_are_consistent_and_private(tmp_path, phase):
    namespace = f"office-{phase.lower()}"
    registry = tmp_path / "registry"
    reference = tmp_path / "admin" / "reference.json"
    prepared = datasets(
        "prepare",
        "--registry",
        registry,
        "--namespace",
        namespace,
        "--phase",
        phase,
        "--anchor",
        "2026-11-05T14:35:00+08:00",
        "--output",
        reference,
    )
    assert prepared.returncode == 0, prepared.stderr
    result = json.loads(prepared.stdout)
    assert result["calendar_records"] == 42
    assert result["timesheet_records"] == 56
    assert result["mail_imported"] is False
    assert stat.S_IMODE(reference.stat().st_mode) == 0o600
    saved = json.loads(reference.read_text())
    assert saved["phase"] == phase

    inventory = tmp_path / "inventory.json"
    captured = datasets(
        "inventory", "--registry", registry, "--namespace", namespace, "--output", inventory
    )
    assert captured.returncode == 0, captured.stderr
    assert json.loads(inventory.read_text()) == saved["before"]

    original = reference.read_bytes()
    repeated = datasets(
        "prepare",
        "--registry",
        registry,
        "--namespace",
        namespace,
        "--phase",
        phase,
        "--anchor",
        "2026-11-05T14:35:00+08:00",
        "--output",
        reference,
    )
    assert repeated.returncode == 2
    assert reference.read_bytes() == original
