"""CI entrypoints preserve failures and forward test/report options."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]


@pytest.fixture
def ci_env(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "Cargo.toml"):
        (source / name).touch()
    (source / "tests/unit").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("gcc", "g++", "rustc", "cargo", "cmake"):
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\nexit 0\n")
        path.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text('#!/usr/bin/env bash\n[[ "$1" != sync ]] || exit 17\n')
    uv.chmod(0o755)
    venv_bin = tmp_path / "venv/bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['-m', 'pytest']:\n"
        "    pathlib.Path(os.environ['ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
        "    sys.exit(int(os.environ.get('PYTEST_EXIT', '0')))\n"
        "sys.stdin.read()\n"
    )
    python.chmod(0o755)
    return {
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "OV_SOURCE_DIR": str(source),
        "UV_PROJECT_ENVIRONMENT": str(venv_bin.parent),
        "REPORT_DIR": str(tmp_path / "reports"),
        "OV_SKIP_BUILD": "1",
        "OV_SKIP_UV_SYNC": "0",
        "CI_BUILD_JOBS": "4",
        "UT_SCOPE": "unit",
        "UT_WORKERS": "1",
        "ARGS_FILE": str(tmp_path / "pytest-args.json"),
    }


def run(script, env, *args):
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_build_propagates_dependency_sync_failure(ci_env):
    result = run("build_ci.sh", ci_env)
    assert result.returncode == 17
    assert "退出码 17" in result.stdout
    assert not (Path(ci_env["REPORT_DIR"]) / "build-result.json").exists()


@pytest.mark.parametrize("jobs", ["0", "-1", "auto"])
def test_build_rejects_invalid_parallelism(ci_env, jobs):
    result = run("build_ci.sh", {**ci_env, "CI_BUILD_JOBS": jobs})
    assert result.returncode == 2
    assert "CI_BUILD_JOBS" in result.stdout


def test_unit_scope_and_report_paths_are_forwarded(ci_env):
    result = run("ut_ci.sh", ci_env)
    assert result.returncode == 0, result.stdout
    args = json.loads(Path(ci_env["ARGS_FILE"]).read_text())
    assert args[-1] == "tests/unit"
    assert f"--junitxml={ci_env['REPORT_DIR']}/junit.xml" in args
    assert f"--cov-report=xml:{ci_env['REPORT_DIR']}/coverage.xml" in args
    assert not (Path(ci_env["REPORT_DIR"]) / "build.log").exists()


def test_custom_test_selection_and_failure_reach_the_ci_runner(ci_env):
    result = run("ut_ci.sh", {**ci_env, "PYTEST_EXIT": "5"}, "tests/unit/test_target.py", "-x")
    assert result.returncode == 5
    args = json.loads(Path(ci_env["ARGS_FILE"]).read_text())
    assert args[-2:] == ["tests/unit/test_target.py", "-x"]
    assert "UT 验证通过" not in result.stdout


def test_parallel_workers_are_forwarded(ci_env):
    result = run("ut_ci.sh", {**ci_env, "UT_WORKERS": "2"})
    assert result.returncode == 0, result.stdout
    args = json.loads(Path(ci_env["ARGS_FILE"]).read_text())
    assert args[args.index("-n") + 1] == "2"
    assert "--dist=loadscope" in args


@pytest.mark.parametrize("key,value", [("UT_SCOPE", "unknown"), ("UT_WORKERS", "0")])
def test_invalid_test_settings_fail_the_job(ci_env, key, value):
    result = run("ut_ci.sh", {**ci_env, key: value})
    assert result.returncode == 2
    assert key in result.stdout
    assert not Path(ci_env["ARGS_FILE"]).exists()


@pytest.mark.parametrize("script", ["build_ci.sh", "ut_ci.sh"])
def test_root_entrypoints_resolve_source_and_sibling_build(ci_env, tmp_path, script):
    source = Path(ci_env["OV_SOURCE_DIR"])
    for name in ("build_ci.sh", "ut_ci.sh"):
        shutil.copy2(SCRIPTS / name, source / name)
    env = {key: value for key, value in ci_env.items() if key != "OV_SOURCE_DIR"}
    env["OV_SKIP_BUILD"] = "0"
    result = subprocess.run(
        ["bash", str(source / script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert "退出码 17" in result.stdout
