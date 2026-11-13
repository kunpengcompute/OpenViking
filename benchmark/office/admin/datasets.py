# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Prepare and inspect office datasets, workspaces, mail accounts and file inputs."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import re
import secrets
import subprocess
from pathlib import Path

from benchmark.office.tasks import CASES, DEFAULT_ANCHORS, render_task
from benchmark.paths import OFFICE_ROOT

PACKAGE = "benchmark.office"
REQUIRED = (
    "admin/mail.py",
    "cases/page_layer.py",
    "cases/portal/data.py",
    "cases/portal/store.py",
    "cases/timesheet/data.py",
    "cases/timesheet/app.py",
    "cases/timesheet/views.py",
    "datasets/timesheet/catalog.py",
    "cases/timesheet/ui.js",
    "cases/timesheet/ui.css",
    "cases/mail/data.py",
    "suites/run.py",
    *(
        f"suites/{case}/scoring.py"
        for case in (
            "schedule",
            "timesheet",
            "mail",
            "ppt_update",
            "reqdoc_generation",
        )
    ),
    "cases/schedule/data.py",
    *(f"tasks/{case}/{phase}.md" for case in CASES for phase in ("H", "Q")),
    "datasets/schedule/data.py",
    "datasets/mail/data.py",
    "datasets/timesheet/data.py",
    "datasets/ppt_update/modify_test.pptx",
    "datasets/reqdoc_generation/oracle.json",
    *(
        f"datasets/reqdoc_generation/inputs/{name}"
        for name in (
            "business_background.md",
            "requirements_confirmed.md",
            "change_log.docx",
            "api_contract.json",
            "metrics.xlsx",
            "prototype.pdf",
            "record_helpers.py",
        )
    ),
)


def check(source: Path | None = None) -> dict:
    folder = source if source is not None else OFFICE_ROOT
    missing = [name for name in REQUIRED if not (folder / name).is_file()]
    if missing:
        raise ValueError(
            "Office case assets are missing. Deploy the matching case package under "
            f"{folder} (resolved: {folder.resolve()}) first: " + ", ".join(missing)
        )
    return {
        "assets_ready": True,
        "source": str(folder),
        "resolved_source": str(folder.resolve()),
        "source_is_symlink": folder.is_symlink(),
    }


def load(name: str):
    check()
    return importlib.import_module(f"{PACKAGE}.{name}")


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def inventory(registry: Path, namespace: str) -> dict:
    config = json.loads((registry / f"{namespace}.json").read_text())
    if config.get("case") != "office" or config.get("namespace") != namespace:
        raise ValueError("Expected an initialized office workspace")
    return {
        "namespace": namespace,
        "anchor": config["anchor"],
        "calendar": load("cases.schedule.data").inventory(registry, namespace),
        "timesheet": load("cases.timesheet.data").inventory(registry, namespace),
    }


def mail_account(args: argparse.Namespace) -> dict:
    """Add one account to the bind-mounted passwd file of the Dovecot fixture."""
    settings = json.loads((args.mail_root / "mailbox.local.json").read_text())
    username = f"{args.namespace}@example.test"
    password = secrets.token_urlsafe(24)
    hashed = subprocess.check_output(
        ["openssl", "passwd", "-6", "-stdin"], input=password + "\n", text=True
    ).strip()
    if args.output.exists():
        raise FileExistsError("Choose a new mail settings file")
    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (args.mail_root / "dovecot/users").open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        entries = handle.read()
        if any(line.partition(":")[0] == username for line in entries.splitlines()):
            raise ValueError("Mail account already exists; choose a fresh namespace")
        settings.update(username=username, password=password, mailbox="INBOX")
        save(args.output, settings)
        handle.seek(0, os.SEEK_END)
        handle.write(
            ("\n" if entries and not entries.endswith("\n") else "")
            + username
            + ":{SHA512-CRYPT}"
            + hashed
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return {"namespace": args.namespace, "mail_settings": str(args.output)}


def prepare(args: argparse.Namespace) -> dict:
    data = load("cases.portal.data")
    mail = load("cases.mail.data")
    # Validate private inputs before creating a workspace. Each mail workspace
    # requires an administrator-provisioned, separate IMAP account.
    settings = None
    if args.mail_settings:
        settings = json.loads(args.mail_settings.read_text())
        for key in ("username", "password", "imap_host", "imap_port", "webmail_url"):
            if not settings.get(key):
                raise ValueError(f"Missing mail setting: {key}")
        private_folder = args.registry / "office-private"
        if private_folder.exists():
            for path in private_folder.glob("*.local.json"):
                other = json.loads(path.read_text()).get("mail", {})
                if all(
                    other.get(key) == settings.get(key)
                    for key in ("username", "imap_host", "imap_port")
                ):
                    raise ValueError("Use a separate IMAP account for each workspace")
    if args.output.exists():
        raise FileExistsError("Choose a new output file")
    case = getattr(args, "case", None)
    page_url = getattr(args, "page_url", "http://127.0.0.1:19482")
    anchor = args.anchor or (DEFAULT_ANCHORS[case][args.phase] if case else None)
    task_text = None
    if case:
        task_text = render_task(
            case, args.phase, workspace=args.namespace, page_url=page_url, anchor=anchor
        )
    if case == "schedule":
        if anchor != DEFAULT_ANCHORS[case][args.phase]:
            raise ValueError("The six-meeting fixture uses the frozen schedule anchor")
        spec = load("cases.schedule.resource_six_case").prepare(
            args.registry, args.namespace, page_url
        )
    else:
        spec = data.provision(args.registry, args.namespace, anchor=anchor)
    config_path = args.registry / f"{args.namespace}.json"
    config = json.loads(config_path.read_text())
    config["mail_read"] = mail.metadata(args.phase)
    goals = None
    if case:
        config.update(task_case=case, task_phase=args.phase)
        if case == "mail":
            dataset = load("datasets.mail.data")
            config.update(mail_ready=settings is not None, task_version=dataset.VERSION)
            config["mail_read"] = {
                "version": dataset.VERSION,
                "phase": args.phase,
                "target_count": len(dataset.goals(args.phase)),
            }
            goals = dataset.goals(args.phase)
        elif case == "timesheet":
            dataset = load("datasets.timesheet.data")
            config["task_version"] = dataset.VERSION
            goals = dataset.goals(args.phase)
        else:
            config["task_version"] = spec["version"]
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    mail_result = None
    if settings:
        mail_result = load("admin.mail").seed(
            settings,
            "history" if args.phase == "H" else "target",
            load("datasets.mail.data").fixture(settings["username"], args.phase)
            if case == "mail"
            else mail.fixture(settings["username"]),
        )
        private_folder = args.registry / "office-private"
        private_folder.mkdir(mode=0o700, exist_ok=True)
        private_folder.chmod(0o700)
        save(private_folder / f"{args.namespace}.local.json", {"mail": settings})
    snapshot = inventory(args.registry, args.namespace)
    save(
        args.output,
        {
            "phase": args.phase,
            "case": case,
            "task": task_text,
            "goals": goals,
            "reference": spec,
            "mail_task": task_text
            if case == "mail"
            else mail.task(config)
            if case is None
            else None,
            "mail_import": mail_result,
            "before": snapshot,
        },
    )
    return {
        "namespace": args.namespace,
        "phase": args.phase,
        "case": case,
        "login_path": "/mock/login?workspace=" + args.namespace,
        "calendar_records": len(snapshot["calendar"]["events"]),
        "timesheet_records": len(snapshot["timesheet"]["entries"]),
        "mail_imported": settings is not None,
        "reference_file": str(args.output),
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asset_check = commands.add_parser("check")
    asset_check.add_argument("--source", type=Path, help="Check a case package before installation")
    account = commands.add_parser("mail-account")
    account.add_argument("--mail-root", type=Path, required=True)
    account.add_argument("--namespace", required=True)
    account.add_argument("--output", type=Path, required=True)
    init = commands.add_parser("prepare")
    init.add_argument("--registry", type=Path, required=True)
    init.add_argument("--namespace", required=True)
    init.add_argument("--phase", choices=("H", "Q"), required=True)
    init.add_argument("--case", choices=CASES, help="Prepare the corresponding H/Q office workload")
    init.add_argument("--page-url", default="http://127.0.0.1:19482")
    init.add_argument(
        "--anchor", help="ISO date-time; defaults to the selected task's frozen clock"
    )
    init.add_argument("--mail-settings", type=Path)
    init.add_argument("--output", type=Path, required=True)
    dump = commands.add_parser("inventory")
    dump.add_argument("--registry", type=Path, required=True)
    dump.add_argument("--namespace", required=True)
    dump.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "namespace") and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.namespace):
        parser.error("namespace must be 1-80 lowercase letters, digits or hyphens")
    try:
        if args.command == "check":
            result = check(args.source)
        elif args.command == "mail-account":
            result = mail_account(args)
        elif args.command == "prepare":
            args.registry.mkdir(parents=True, mode=0o700, exist_ok=True)
            args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            result = prepare(args)
        elif args.command == "inventory":
            save(args.output, inventory(args.registry, args.namespace))
            result = {"snapshot": str(args.output)}
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
