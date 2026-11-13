# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Container entry point: configure mounted files and start preinstalled services."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
PREFIX = "examples.deepagents_ov_context.scripts."


def config_dir() -> Path:
    return Path(os.environ.get("OV_CONFIG_DIR", "/config"))


def data_dir() -> Path:
    return Path(os.environ.get("OV_DATA_DIR", "/data"))


def write_private(path: Path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value)


def initialize() -> None:
    config = config_dir()
    config.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, template in (
        ("models.local.json", "models.example.json"),
        (".env", "chat.example.env"),
    ):
        path = config / name
        if not path.exists():
            write_private(path, (SOURCE / "config" / template).read_text())
    for name in ("ov", "chat", "cases/admin", "cases/inputs", "cases/results", "cases/registry"):
        (data_dir() / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    print("Configuration templates and data directories ready; fill models.local.json.")


def configure() -> None:
    from examples.deepagents_ov_context.chat.config import ModelConfig, write_server_config

    model = ModelConfig.load(config_dir() / "models.local.json")
    path = write_server_config(
        model, config_dir() / "ov", int(os.environ.get("OV_CHAT_OV_PORT", "1933"))
    )
    settings = json.loads(path.read_text())
    settings["storage"]["workspace"] = str(data_dir() / "ov")
    path.write_text(json.dumps(settings, indent=2) + "\n")
    print("Trusted OV configuration ready; service credential preserved.")


def initialize_mailbox() -> None:
    from .services import case_ports

    imap_port, webmail_port = case_ports()
    root = data_dir() / "cases/mailbox"
    root.mkdir(parents=True, mode=0o755, exist_ok=False)
    root.chmod(0o755)
    password = secrets.token_urlsafe(24)
    settings = {
        "username": "tester@example.test",
        "password": password,
        "imap_host": "127.0.0.1",
        "imap_port": imap_port,
        "webmail_url": f"http://127.0.0.1:{webmail_port}/",
    }
    write_private(root / "mailbox.local.json", json.dumps(settings))
    for name in (
        "dovecot",
        "roundcube",
        "mail-data",
        "roundcube/db",
        "roundcube/temp",
        "roundcube/logs",
    ):
        (root / name).mkdir(mode=0o755)
        (root / name).chmod(0o755)
    mail_uid = os.getuid() or 1000
    mail_gid = os.getgid() or 1000
    if os.getuid() == 0:
        os.chown(root / "mail-data", mail_uid, mail_gid)
    hashed = subprocess.check_output(
        ["openssl", "passwd", "-6", "-stdin"], input=password + "\n", text=True
    ).strip()
    users = root / "dovecot/users"
    users.write_text(settings["username"] + ":{SHA512-CRYPT}" + hashed + "\n")
    php = root / "roundcube/config.inc.php"
    php.write_text(
        "<?php\n"
        f"$config['db_dsnw'] = {json.dumps('sqlite:///' + str(root / 'roundcube/db/sqlite.db') + '?mode=0600')};\n"
        f"$config['imap_host'] = '127.0.0.1:{imap_port}';\n"
        "$config['smtp_host'] = '127.0.0.1:9';\n"
        f"$config['des_key'] = '{secrets.token_hex(12)}';\n"
        f"$config['temp_dir'] = {json.dumps(str(root / 'roundcube/temp'))};\n"
        f"$config['log_dir'] = {json.dumps(str(root / 'roundcube/logs'))};\n"
        "$config['language'] = 'zh_CN';\n$config['timezone'] = 'Asia/Shanghai';\n"
        "$config['date_format'] = 'Y-m-d';\n$config['session_lifetime'] = 240;\n"
        "$config['disabled_actions'] = ['mail.compose', 'mail.send', 'mail.delete'];\n"
        "$config['enable_installer'] = false;\n$config['plugins'] = [];\n"
    )
    for path in (users, php):
        path.chmod(0o600)
    print("Dedicated mailbox configuration ready; private credentials saved.")


def load_credential() -> None:
    key = config_dir() / "ov/service.key"
    if key.is_file() and not os.environ.get("OV_SERVICE_API_KEY"):
        os.environ["OV_SERVICE_API_KEY"] = key.read_text().strip()


def execute_module(module: str, args: list[str]) -> None:
    os.execv(sys.executable, [sys.executable, "-m", module, *args])


def chat_args() -> list[str]:
    tools = Path(os.environ.get("OV_CHAT_TOOLS", "/opt/ov-tools"))
    browsers = sorted((tools / "playwright").rglob("headless_shell"))
    if not browsers:
        raise ValueError("Chromium missing from image; rebuild the chat image")
    return [
        "--model-config",
        str(config_dir() / "models.local.json"),
        "--env-file",
        str(config_dir() / ".env"),
        "--data-dir",
        str(data_dir() / "chat"),
        "--ov-url",
        os.environ.get("OV_CHAT_OV_URL", "http://127.0.0.1:1933"),
        "--port",
        os.environ.get("OV_CHAT_PORT", "1934"),
        "--agent-browser",
        str(tools / "node_modules/.bin/agent-browser"),
        "--browser-executable",
        str(browsers[0]),
        "--pptx-skill",
        str(tools / "anthropic-skills/skills/pptx"),
        "--skill-node-modules",
        str(tools / "node_modules"),
        "--office-skills",
        "--thinking",
        "--max-tokens",
        "16384",
        "--recursion-limit",
        "1000",
    ]


def main() -> None:
    os.umask(0o077)
    Path(os.environ.get("HOME", "/tmp/ov-home")).mkdir(parents=True, exist_ok=True)
    command, *args = sys.argv[1:] or ["chat"]
    load_credential()
    if command == "init":
        initialize()
    elif command == "configure":
        configure()
    elif command == "init-mailbox":
        initialize_mailbox()
    elif command == "health":
        from .services import cases_enabled, check_cases, page_port

        name = args[0]
        if name == "cases":
            print(json.dumps(check_cases()))
            return
        port = (
            os.environ.get("OV_CHAT_OV_PORT", "1933")
            if name == "ov"
            else os.environ.get("OV_CHAT_PORT", "1934")
        )
        path = "/ready" if name == "ov" else "/api/health"
        if name == "pages":
            port, path = str(page_port()), "/office-assets/ui.js"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}{path}", timeout=4) as response:
            if response.status != 200:
                raise ValueError("Service health failed")
        if name == "chat" and cases_enabled():
            check_cases()
    elif command == "ov":
        execute_module(
            "openviking.server.bootstrap",
            [
                "--config",
                str(config_dir() / "ov/ov.conf"),
                "--host",
                "127.0.0.1",
                "--port",
                os.environ.get("OV_CHAT_OV_PORT", "1933"),
                *args,
            ],
        )
    elif command == "chat":
        from .services import run_chat

        raise SystemExit(run_chat([*chat_args(), *args], data_dir()))
    elif command == "pages":
        from benchmark.office.admin.datasets import check

        check()
        execute_module(
            "benchmark.office.cases.page_layer",
            [
                "--registry",
                str(data_dir() / "cases/registry"),
                *args,
            ],
        )
    elif command == "case-data":
        execute_module("benchmark.office.admin.datasets", args)
    elif command == "python":
        os.execv(sys.executable, [sys.executable, *args])
    elif command == "check":
        from examples.deepagents_ov_context.scripts.check import (
            check_install,
            check_office_skills,
            check_pptx_skill,
        )

        result = check_install()
        tools = Path(os.environ.get("OV_CHAT_TOOLS", "/opt/ov-tools"))
        result["office"] = check_office_skills()
        result["pptx"] = check_pptx_skill(tools / "anthropic-skills/skills/pptx", tools)
        if shutil.which("node") is None:
            raise ValueError("Node missing from image")
        for command in ("dovecot", "php"):
            if shutil.which(command) is None:
                raise ValueError(f"{command} missing from Chat image")
        subprocess.run(
            [
                "php",
                "-r",
                "exit(extension_loaded('pdo_sqlite') && extension_loaded('mbstring') ? 0 : 1);",
            ],
            check=True,
            timeout=10,
        )
        print(json.dumps(result, ensure_ascii=False))
    elif command == "check-services":
        execute_module(
            PREFIX + "check",
            [
                "services",
                "--ov-url",
                os.environ.get("OV_CHAT_OV_URL", "http://127.0.0.1:1933"),
                "--url",
                os.environ.get("OV_CHAT_URL", "http://127.0.0.1:1934"),
                "--browser",
                *args,
            ],
        )
    elif command == "smoke":
        execute_module(
            PREFIX + "smoke",
            [
                "--url",
                os.environ.get("OV_CHAT_URL", "http://127.0.0.1:1934"),
                "--output",
                str(data_dir() / "chat/smoke.json"),
                *args,
            ],
        )
    else:
        raise ValueError(f"Unknown container command: {command}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None
