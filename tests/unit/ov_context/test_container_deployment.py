"""Container configuration stays private and tools are resolved inside the image."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from examples.deepagents_ov_context.docker import runtime
from examples.deepagents_ov_context.docker import services as processes

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "examples/deepagents_ov_context/docker"


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    monkeypatch.setenv("OV_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OV_SERVICE_API_KEY", raising=False)
    monkeypatch.setenv("OV_CASE_IMAP_PORT", "19493")
    monkeypatch.setenv("OV_CASE_PAGE_PORT", "19482")
    monkeypatch.setenv("OV_CASE_WEBMAIL_PORT", "19480")
    monkeypatch.setenv("OV_CASES_ENABLED", "false")
    return tmp_path


def test_initialize_preserves_private_configuration(mounted):
    runtime.initialize()
    model = mounted / "config/models.local.json"
    assert model.stat().st_mode & 0o777 == 0o600
    model.write_text('{"api_key":"keep-private"}')
    (mounted / "config/.env").write_text("KEEP=explicit\n")
    runtime.initialize()
    assert json.loads(model.read_text())["api_key"] == "keep-private"
    assert (mounted / "config/.env").read_text() == "KEEP=explicit\n"
    assert (mounted / "data/chat").is_dir()
    assert (mounted / "data/cases/admin").is_dir()


def test_configure_uses_data_mount_and_stable_service_key(mounted, capsys):
    runtime.initialize()
    model = mounted / "config/models.local.json"
    model.write_text(
        json.dumps(
            {
                "api_key": "fixture-only-private",
                "api_base": "https://example.test/v1",
                "chat_model": "test-model",
                "embedding_model": "test-embedding",
                "embedding_dimension": 8,
            }
        )
    )
    runtime.configure()
    path = mounted / "config/ov/ov.conf"
    settings = json.loads(path.read_text())
    key = (path.parent / "service.key").read_text()
    assert settings["server"]["auth_mode"] == "trusted"
    assert settings["storage"]["workspace"] == str(mounted / "data/ov")
    # A killed worker's lease must expire well before the 300-second archive wait.
    from openviking_cli.utils.config.open_viking_config import OpenVikingConfig

    parsed = OpenVikingConfig.from_dict(settings)
    assert parsed.storage.transaction.lock_expire == 30
    runtime.configure()
    assert (path.parent / "service.key").read_text() == key
    assert path.stat().st_mode & 0o777 == 0o600
    assert "fixture-only-private" not in capsys.readouterr().out


def test_credential_from_readonly_mount_does_not_override_explicit_key(mounted, monkeypatch):
    folder = mounted / "config/ov"
    folder.mkdir(parents=True)
    (folder / "service.key").write_text("service-fixture")
    runtime.load_credential()
    assert os.environ["OV_SERVICE_API_KEY"] == "service-fixture"
    monkeypatch.setenv("OV_SERVICE_API_KEY", "explicit")
    runtime.load_credential()
    assert os.environ["OV_SERVICE_API_KEY"] == "explicit"


def test_chat_uses_baked_tools_and_mounted_configuration(mounted, monkeypatch):
    tools = mounted / "image-tools"
    chrome = tools / "playwright/chromium-test/chrome-linux/headless_shell"
    chrome.parent.mkdir(parents=True)
    chrome.touch()
    monkeypatch.setenv("OV_CHAT_TOOLS", str(tools))
    args = runtime.chat_args()
    assert args[args.index("--data-dir") + 1] == str(mounted / "data/chat")
    assert args[args.index("--browser-executable") + 1] == str(chrome)
    assert args[args.index("--agent-browser") + 1].startswith(str(tools))
    assert args[args.index("--pptx-skill") + 1].startswith(str(tools))
    assert "--office-skills" in args
    chrome.unlink()
    with pytest.raises(ValueError, match="rebuild"):
        runtime.chat_args()


def test_chat_start_does_not_run_installers(mounted, monkeypatch):
    monkeypatch.setattr(runtime.sys, "argv", ["runtime", "chat", "--port", "19999"])
    monkeypatch.setattr(runtime, "chat_args", lambda: ["--port", "1934"])
    called = []
    monkeypatch.setattr(processes, "run_chat", lambda args, data: called.append((args, data)) or 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("runtime installer"))
    with pytest.raises(SystemExit) as result:
        runtime.main()
    assert result.value.code == 0
    assert called == [(["--port", "1934", "--port", "19999"], mounted / "data")]


def test_mailbox_configuration_is_private_and_matches_volume_owner(mounted, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "fixture-hash\n")
    monkeypatch.setattr(os, "getuid", lambda: 1234)
    monkeypatch.setattr(os, "getgid", lambda: 1235)
    runtime.initialize_mailbox()
    root = mounted / "data/cases/mailbox"
    settings = root / "mailbox.local.json"
    assert settings.stat().st_mode & 0o777 == 0o600
    secret = json.loads(settings.read_text())["password"]
    assert secret not in (root / "dovecot/users").read_text()
    assert (root / "roundcube/config.inc.php").stat().st_mode & 0o777 == 0o600
    config = processes.dovecot_config(root, 1234, 1235, 19493)
    assert "uid=1234 gid=1235" in config
    assert "first_valid_uid = 1234" in config
    assert "listen = 127.0.0.1" in config and "port = 19493" in config
    assert str(root / "mail-data/%u/Maildir") in config
    php = (root / "roundcube/config.inc.php").read_text()
    assert "sqlite:////" in php and "127.0.0.1:19493" in php
    assert secret not in php
    with pytest.raises(FileExistsError):
        runtime.initialize_mailbox()


def test_compose_separates_tools_from_writable_data():
    services = yaml.safe_load((DOCKER / "compose.yaml").read_text())["services"]
    assert set(services) == {"ov", "chat"}
    for name in ("ov", "chat"):
        assert services[name]["network_mode"] == "host"
        assert any(v.endswith(":/config:ro") for v in services[name]["volumes"])
        assert all("ov-tools" not in v for v in services[name]["volumes"])
        assert all("docker.sock" not in v for v in services[name]["volumes"])
    assert services["ov"]["build"]["target"] == "ov"
    assert services["chat"]["build"]["target"] == "chat"
    assert services["chat"]["user"] == "0:0"
    assert services["chat"]["environment"]["OV_RUN_UID"] == "${OV_UID:-1000}"
    assert any(v.endswith(":/data/cases") for v in services["chat"]["volumes"])
    assert services["chat"]["depends_on"]["ov"]["condition"] == "service_healthy"
    assert services["chat"]["build"]["context"] == "../../.."
    admin = yaml.safe_load((DOCKER / "compose.admin.yaml").read_text())["services"]
    assert set(admin) == {"chat"}
    mounts = {mount["target"]: mount for mount in admin["chat"]["volumes"]}
    assert mounts["/config"]["read_only"] is False
    assert mounts["/data"]["read_only"] is False


def test_image_installs_tools_at_build_time_and_excludes_secrets():
    dockerfile = (DOCKER / "Dockerfile").read_text()
    for step in (
        "install-browser",
        "install-pptx-renderer",
        "install-pptx-skill",
        "install-office-skills",
    ):
        assert f"deploy.sh {step}" in dockerfile
    ignored = (DOCKER / "Dockerfile.dockerignore").read_text()
    for pattern in ("**/*.local.json", "**/service.key", "**/.env", "**/*.sqlite3", "**/*.so"):
        assert pattern in ignored


def test_wrapper_is_valid_shell():
    script = DOCKER.parent / "scripts/container.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert "datasets-up" in result.stdout


def test_wrapper_enables_cases_persistently_without_extra_services(tmp_path):
    executable = tmp_path / "bin/docker"
    executable.parent.mkdir()
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    executable.chmod(0o755)
    home = tmp_path / "deployment"
    home.mkdir()
    (home / "compose.env").write_text("OV_CHAT_IMAGE=my-chat:test\nOV_CASES_ENABLED=false\n")
    env = {
        **os.environ,
        "PATH": str(executable.parent) + os.pathsep + os.environ["PATH"],
        "CALLS": str(tmp_path / "calls.txt"),
        "OV_DEPLOY_HOME": str(home),
        "OV_UID": "1234",
        "OV_GID": "1235",
    }
    subprocess.run(
        ["bash", str(DOCKER.parent / "scripts/container.sh"), "datasets-up"], env=env, check=True
    )
    settings = (home / "compose.env").read_text()
    assert "OV_CHAT_IMAGE=my-chat:test" in settings
    assert settings.count("OV_CASES_ENABLED=") == 1 and "OV_CASES_ENABLED=true" in settings
    calls = (tmp_path / "calls.txt").read_text()
    assert "up -d --wait --force-recreate chat" in calls
    assert "case-admin" not in calls and " imap" not in calls and " webmail" not in calls


@pytest.mark.parametrize("uid,gid", [("0", "1000"), ("1000", "0"), ("-1", "1000")])
def test_model_process_requires_nonroot_identity(monkeypatch, uid, gid):
    monkeypatch.setenv("OV_RUN_UID", uid)
    monkeypatch.setenv("OV_RUN_GID", gid)
    with pytest.raises(ValueError, match="non-root"):
        processes.run_identity()


def test_worker_home_is_writable_and_separate_from_supervisor(mounted, monkeypatch):
    home = mounted / "worker-home"
    monkeypatch.setenv("OV_RUNTIME_HOME", str(home))
    monkeypatch.setattr(os, "getuid", lambda: 1234)
    env = processes.child_environment(1234, 1235)
    assert env["HOME"] == str(home) and home.stat().st_mode & 0o777 == 0o700
    assert os.environ["HOME"] == str(mounted / "home")


def test_worker_home_rejects_symlinks(mounted, monkeypatch):
    target = mounted / "unrelated"
    target.mkdir()
    home = mounted / "worker-home"
    home.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("OV_RUNTIME_HOME", str(home))
    with pytest.raises(ValueError, match="ordinary directory"):
        processes.child_environment(1234, 1235)


@pytest.mark.parametrize(
    "imap,webmail", [("19482", "19480"), ("19493", "1934"), ("19493", "19493"), ("80", "19480")]
)
def test_case_ports_reject_conflicts(monkeypatch, imap, webmail):
    monkeypatch.setenv("OV_CHAT_PORT", "1934")
    monkeypatch.setenv("OV_CHAT_OV_PORT", "1933")
    monkeypatch.setenv("OV_CASE_IMAP_PORT", imap)
    monkeypatch.setenv("OV_CASE_WEBMAIL_PORT", webmail)
    with pytest.raises(ValueError):
        processes.case_ports()


def test_custom_case_ports_are_used_in_commands_and_mail_links(mounted, monkeypatch):
    monkeypatch.setenv("OV_CASE_PAGE_PORT", "19582")
    monkeypatch.setenv("OV_CASE_WEBMAIL_PORT", "19580")
    monkeypatch.setenv("OV_CASE_IMAP_PORT", "19593")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "fixture-hash\n")
    monkeypatch.setattr(os, "getuid", lambda: 1234)
    monkeypatch.setattr(os, "getgid", lambda: 1235)
    runtime.initialize_mailbox()
    from benchmark.office.admin import datasets
    from benchmark.office.cases.portal.catalog import (
        applications,
    )

    monkeypatch.setattr(datasets, "check", lambda: {})
    monkeypatch.setattr(os, "getuid", lambda: 0)
    specs = processes.case_commands(mounted / "data", {})
    pages = next(command for name, command, _ in specs if name == "pages")
    assert pages[pages.index("--port") + 1] == "19582"
    assert applications("demo")[0]["url"].startswith("http://127.0.0.1:19580/")
    monkeypatch.setenv("OV_CASE_WEBMAIL_PORT", "19582")
    with pytest.raises(ValueError):
        processes.case_ports()


@pytest.mark.parametrize("port", ["80", "65536", "1934", "1933"])
def test_page_port_rejects_invalid_or_main_service_ports(monkeypatch, port):
    monkeypatch.setenv("OV_CASE_PAGE_PORT", port)
    monkeypatch.setenv("OV_CHAT_PORT", "1934")
    monkeypatch.setenv("OV_CHAT_OV_PORT", "1933")
    with pytest.raises(ValueError):
        processes.page_port()


def test_case_services_use_same_data_and_nonroot_workers(mounted, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "fixture-hash\n")
    monkeypatch.setattr(os, "getuid", lambda: 1234)
    monkeypatch.setattr(os, "getgid", lambda: 1235)
    runtime.initialize_mailbox()
    from benchmark.office.admin import datasets

    monkeypatch.setattr(datasets, "check", lambda: {})
    monkeypatch.setattr(os, "getuid", lambda: 0)
    env = {}
    specs = processes.case_commands(mounted / "data", env)
    assert {name: privileged for name, _, privileged in specs} == {
        "imap": True,
        "pages": False,
        "webmail": False,
    }
    assert "127.0.0.1:19480" in specs[2][1]
    assert env["RCUBE_CONFIG_PATH"] == str(mounted / "data/cases/mailbox/roundcube")


def test_case_start_requires_explicit_initialization(mounted, monkeypatch):
    from benchmark.office.admin import datasets

    monkeypatch.setattr(datasets, "check", lambda: {})
    monkeypatch.setattr(os, "getuid", lambda: 0)
    with pytest.raises(ValueError, match="datasets-init"):
        processes.case_commands(mounted / "data", {})


def test_supervisor_stops_stack_when_a_service_exits(mounted, monkeypatch):
    monkeypatch.setattr(processes, "run_identity", lambda: (1234, 1235))
    monkeypatch.setattr(processes, "child_environment", lambda uid, gid: {})
    monkeypatch.setattr(
        processes, "privilege_options", lambda uid, gid: {"user": uid, "group": gid}
    )
    calls = []
    child = type("Child", (), {"pid": 9876, "poll": lambda self: 0})()
    monkeypatch.setattr(
        subprocess, "Popen", lambda command, **kw: calls.append((command, kw)) or child
    )
    monkeypatch.setattr(processes, "stop_children", lambda children: calls.append(children))
    assert processes.run_chat(["--port", "1934"], mounted / "data") == 1
    assert calls[0][1]["user"] == 1234 and calls[-1] == [child]


def test_supervisor_kills_stubborn_child_process_group():
    import signal
    import sys

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        processes.stop_children([child], timeout=0.1)
        assert child.returncode == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_case_data_command_dispatches_dataset_admin(mounted, monkeypatch):
    monkeypatch.setattr(runtime.sys, "argv", ["runtime", "case-data", "check"])
    dispatched = []
    monkeypatch.setattr(
        runtime, "execute_module", lambda module, args: dispatched.append((module, args))
    )
    runtime.main()
    assert dispatched == [("benchmark.office.admin.datasets", ["check"])]
