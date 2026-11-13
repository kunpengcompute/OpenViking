# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run Chat and optional case fixtures in one container, with bounded shutdown."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def cases_enabled() -> bool:
    value = os.environ.get("OV_CASES_ENABLED", "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("OV_CASES_ENABLED must be true or false")
    return value in {"true", "1"}


def run_identity() -> tuple[int, int]:
    uid = int(os.environ.get("OV_RUN_UID", str(os.getuid() or 1000)))
    gid = int(os.environ.get("OV_RUN_GID", str(os.getgid() or 1000)))
    if uid <= 0 or gid <= 0:
        raise ValueError("OV_RUN_UID and OV_RUN_GID must identify a non-root user")
    if os.getuid() not in (0, uid) or (os.getuid() != 0 and os.getgid() != gid):
        raise ValueError("Runtime user differs from OV_RUN_UID / OV_RUN_GID")
    return uid, gid


def page_port() -> int:
    port = int(os.environ.get("OV_CASE_PAGE_PORT", "19482"))
    reserved = {
        int(os.environ.get("OV_CHAT_PORT", "1934")),
        int(os.environ.get("OV_CHAT_OV_PORT", "1933")),
    }
    if port < 1024 or port > 65535 or port in reserved:
        raise ValueError("Office page port must be unprivileged and distinct from OV and Chat")
    return port


def case_ports() -> tuple[int, int]:
    imap = int(os.environ.get("OV_CASE_IMAP_PORT", "19493"))
    webmail = int(os.environ.get("OV_CASE_WEBMAIL_PORT", "19480"))
    if any(port < 1024 or port > 65535 for port in (imap, webmail)) or imap == webmail:
        raise ValueError("Case service ports must be distinct ports between 1024 and 65535")
    reserved = {
        page_port(),
        int(os.environ.get("OV_CHAT_PORT", "1934")),
        int(os.environ.get("OV_CHAT_OV_PORT", "1933")),
    }
    if imap in reserved or webmail in reserved:
        raise ValueError("Case ports conflict with OV, Chat or the office page service")
    return imap, webmail


def check_cases() -> dict[str, int]:
    imap, webmail = case_ports()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = {}
    for name, url in (
        ("pages", f"http://127.0.0.1:{page_port()}/office-assets/ui.js"),
        ("webmail", f"http://127.0.0.1:{webmail}/"),
    ):
        with opener.open(url, timeout=1) as response:
            if response.status != 200:
                raise ValueError(f"{name} health failed")
            result[name] = response.status
    with socket.create_connection(("127.0.0.1", imap), timeout=1) as connection:
        if not connection.recv(512).startswith(b"* OK"):
            raise ValueError("IMAP greeting missing")
    result["imap"] = imap
    return result


def child_environment(uid: int, gid: int) -> dict[str, str]:
    env = os.environ.copy()
    home = Path(os.environ.get("OV_RUNTIME_HOME", f"/tmp/ov-chat-home-{uid}"))
    if home.is_symlink():
        raise ValueError("Runtime HOME must be an ordinary directory")
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.getuid() == 0:
        os.chown(home, uid, gid, follow_symlinks=False)
    env["HOME"] = str(home)
    return env


def privilege_options(uid: int, gid: int) -> dict:
    return {"user": uid, "group": gid, "extra_groups": []} if os.getuid() == 0 else {}


def dovecot_config(mailbox: Path, uid: int, gid: int, port: int) -> str:
    return (
        "protocols = imap\nlisten = 127.0.0.1\nssl = no\ndisable_plaintext_auth = no\n"
        "log_path = /dev/stderr\ninfo_log_path = /dev/stdout\n"
        "base_dir = /run/ov-case-dovecot\nauth_mechanisms = plain login\n"
        f"first_valid_uid = {uid}\nlast_valid_uid = {uid}\n"
        f"mail_location = maildir:{mailbox}/mail-data/%u/Maildir\n"
        f"passdb {{\n driver = passwd-file\n args = {mailbox}/dovecot/users\n}}\n"
        f"userdb {{\n driver = static\n args = uid={uid} gid={gid} home={mailbox}/mail-data/%u\n}}\n"
        f"service imap-login {{\n inet_listener imap {{\n port = {port}\n }}\n}}\n"
        "service auth {\n user = root\n}\n"
    )


def prepare_dovecot(data: Path, uid: int, gid: int) -> None:
    # Privileged service configuration is generated from code, outside writable case data.
    folder = Path("/run/ov-case-dovecot")
    folder.mkdir(mode=0o755, parents=True, exist_ok=True)
    if folder.is_symlink() or folder.stat().st_uid != 0:
        raise ValueError("Dovecot runtime directory must be root-owned")
    folder.chmod(0o755)
    config = folder / "service.conf"
    config.write_text(dovecot_config(data / "cases/mailbox", uid, gid, case_ports()[0]))
    config.chmod(0o600)


def case_commands(data: Path, env: dict[str, str]) -> list[tuple[str, list[str], bool]]:
    from benchmark.office.admin.datasets import check

    check()
    if os.getuid() != 0:
        raise ValueError("Case services require the container supervisor to start as root")
    mailbox = data / "cases/mailbox"
    for name in ("dovecot/users", "roundcube/config.inc.php"):
        if not (mailbox / name).is_file():
            raise ValueError("Initialize case data with container.sh datasets-init first")
    _, webmail = case_ports()
    app = Path(os.environ.get("OV_ROUNDCUBE_DIR", "/opt/ov-tools/roundcube"))
    env["RCUBE_CONFIG_PATH"] = str(mailbox / "roundcube")
    env["PHP_CLI_SERVER_WORKERS"] = "4"
    return [
        ("imap", ["dovecot", "-F", "-c", "/run/ov-case-dovecot/service.conf"], True),
        (
            "pages",
            [
                sys.executable,
                "-m",
                "benchmark.office.cases.page_layer",
                "--registry",
                str(data / "cases/registry"),
                "--port",
                str(page_port()),
            ],
            False,
        ),
        (
            "webmail",
            [
                "php",
                "-d",
                f"session.save_path={mailbox / 'roundcube/temp'}",
                "-S",
                f"127.0.0.1:{webmail}",
                "-t",
                str(app),
                str(Path(__file__).with_name("roundcube_router.php")),
            ],
            False,
        ),
    ]


def stop_children(children: list[subprocess.Popen], timeout: float = 10) -> None:
    # Stop the service groups, including PHP's worker children.
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for child in children:
        # Reap a process group even if its leader exited before its descendants.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_chat(args: list[str], data: Path) -> int:
    uid, gid = run_identity()
    env = child_environment(uid, gid)
    specs = case_commands(data, env) if cases_enabled() else []
    specs.append(
        ("chat", [sys.executable, "-m", "examples.deepagents_ov_context.chat.server", *args], False)
    )
    children: list[subprocess.Popen] = []
    logs = []
    stopping = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        log_dir = data / "cases/logs"
        if cases_enabled():
            prepare_dovecot(data, uid, gid)
            log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.getuid() == 0:
                os.chown(log_dir, uid, gid)
            app = Path(os.environ.get("OV_ROUNDCUBE_DIR", "/opt/ov-tools/roundcube"))
            subprocess.run(
                ["php", str(app / "bin/initdb.sh"), f"--dir={app / 'SQL'}", "--update"],
                env=env,
                check=True,
                timeout=45,
                **privilege_options(uid, gid),
            )
        for name, command, privileged in specs:
            if stopping:
                return 0
            output = None
            if name != "chat":
                output = (log_dir / f"{name}.log").open("a")
                logs.append(output)
                if os.getuid() == 0:
                    os.fchown(output.fileno(), uid, gid)
            children.append(
                subprocess.Popen(
                    command,
                    env=env,
                    start_new_session=True,
                    stdout=output,
                    stderr=subprocess.STDOUT if output else None,
                    **({} if privileged else privilege_options(uid, gid)),
                )
            )
        while not stopping:
            for (name, _, _), child in zip(specs, children, strict=True):
                status = child.poll()
                if status is not None:
                    print(f"{name} exited ({status}); stopping Chat container", file=sys.stderr)
                    return status if status > 0 else 1
            time.sleep(0.2)
        return 0
    finally:
        stop_children(children)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
