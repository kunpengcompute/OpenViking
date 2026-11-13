# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deployment checks with temporary artifact parsing; no model calls or service changes."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import httpx


class CheckError(RuntimeError):
    """A deployment prerequisite or acceptance condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def check_install() -> dict[str, str]:
    import openviking
    from openviking.pyagfs import get_binding_client
    from openviking.storage.vectordb import engine

    root = Path(__file__).resolve().parents[3]
    source = Path(openviking.__file__).resolve()
    require(source.parent == root / "openviking", f"Unexpected source: {source}")
    deepagents_version = package_version("deepagents")
    require(
        deepagents_version == "0.7.3",
        f"DeepAgents 0.7.3 required, found {deepagents_version}; reinstall .[deepagents] or rebuild the image",
    )
    importlib.import_module("examples.deepagents_ov_context.chat.server")
    importlib.import_module("langgraph.checkpoint.sqlite.aio")
    get_binding_client()
    require(engine.ENGINE_VARIANT != "unavailable", "VectorDB backend unavailable")
    codec = engine.BytesRow(
        engine.Schema(
            [
                {
                    "name": "fields",
                    "id": 0,
                    "data_type": engine.FieldType.text,
                    "legacy_data_type": engine.FieldType.string,
                }
            ]
        )
    )
    sample = {"fields": "x" * 65536}
    encoded = codec.serialize(sample)
    require(
        encoded.startswith(b"\x00\x01") and codec.deserialize(encoded) == sample,
        "Rebuild the native VectorDB backend: versioned TEXT codec unavailable",
    )
    return {
        "python": sys.executable,
        "platform": f"{platform.system()} {platform.machine()}",
        "source": str(source),
        "deepagents": deepagents_version,
        "vectordb": engine.ENGINE_VARIANT,
        "checkpoint": "AsyncSqliteSaver",
        "storage_format": "versioned TEXT; 65536-byte roundtrip OK",
        "result": "Chat imports OK; native storage OK",
    }


def version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def check_browser(tools: Path, chrome: Path) -> dict[str, str]:
    import pdfplumber

    node = shutil.which("node")
    require(node is not None, "Node not found; use deploy.sh to set PATH")
    node_version = version([str(node), "--version"])
    require(node_version.startswith("v24."), f"Node 24 required, found {node_version}")
    cli = tools / "node_modules/.bin/agent-browser"
    require(cli.is_file(), "agent-browser missing; run install-browser")
    require(chrome.is_file(), "Chromium missing; run install-browser or set OV_CHAT_CHROME")
    return {
        "node": str(node),
        "node_version": node_version,
        "browser": version([str(cli), "--version"]),
        "chromium": version([str(chrome), "--version"]),
        "pdfplumber": pdfplumber.__version__,
    }


def check_pptx_skill(skill: Path, tools: Path) -> dict[str, Any]:
    from examples.deepagents_ov_context.chat.office_skills import validate_skill

    validate_skill(skill)
    for name in ("pptx", "markitdown", "PIL", "defusedxml", "lxml"):
        importlib.import_module(name)
    for name in ("node", "soffice", "pdftoppm"):
        require(
            shutil.which(name) is not None, f"Missing {name}; see the pptx Skill deployment steps"
        )
    script = "for(const n of ['pptxgenjs','react','react-dom','react-icons','sharp']) require(require.resolve(n,{paths:[process.argv[1]]})); console.log('PPTX Skill dependencies OK')"
    try:
        node = version(["node", "-e", script, str(tools.resolve())])
    except subprocess.SubprocessError as exc:
        raise CheckError("Skill Node dependencies unavailable; rerun install-pptx-skill") from exc
    return {"skill": str(skill.resolve()), "node": node}


def check_office_skills() -> dict[str, Any]:
    from examples.deepagents_ov_context.chat.office_skills import BUNDLED_SKILLS, validate_skill

    names = ("office-documents", "data-analysis", "agent-browser")
    for name in names:
        validate_skill(BUNDLED_SKILLS / name, name)
    for name in ("docx", "openpyxl", "pypdf", "pdfplumber", "reportlab", "pandas", "duckdb"):
        importlib.import_module(name)
    for name in ("soffice", "pdftoppm"):
        require(shutil.which(name) is not None, f"Missing {name}; run install-pptx-renderer")
    return {"skills": list(names), "files": ["docx", "xlsx", "pdf", "csv", "json", "md", "txt"]}


def get_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    result = response.json()
    require(isinstance(result, dict), "Expected JSON object")
    return result


def check_services(ov_url: str, chat_url: str, require_browser: bool = False) -> dict[str, Any]:
    with httpx.Client(trust_env=False, timeout=30) as client:
        health = get_json(client, ov_url.rstrip("/") + "/health")
        ready = get_json(client, ov_url.rstrip("/") + "/ready")
        chat = get_json(client, chat_url.rstrip("/") + "/api/health")
    require(health.get("status") == "ok", "OV health failed")
    require(ready.get("status") == "ready", "OV is not ready")
    ov = chat.get("ov")
    require(isinstance(ov, dict) and ov.get("healthy") is True, "Chat cannot reach OV")
    if require_browser:
        require(chat.get("browser") == "agent-browser", "Browser tools are not enabled")
        require(chat.get("thinking") is True, "Thinking is not enabled")
    return {"ov_health": health, "ov_ready": ready, "chat": chat}


def validate_session(detail: dict[str, Any]) -> dict[str, Any]:
    from examples.deepagents_ov_context.chat.server import SessionDetail

    parsed = SessionDetail.model_validate(detail)
    context = parsed.context
    require(not parsed.busy, "Session still has an active Run")
    require(context.source == "checkpoint", "Expected checkpoint; check before restarting Chat")
    require(context.write_index == context.raw_message_count > 0, "Write cursor has not caught up")
    require(context.cached_message_count > 0, "Message cache is empty")
    require(context.run_id is None, "Run temporary state has not been cleared")
    require(context.completed_runs >= 1, "No completed Run")
    return context.model_dump(exclude={"messages", "owner", "archive_id"})


def check_session(url: str, session_id: str) -> dict[str, Any]:
    require(bool(session_id) and "/" not in session_id, "Invalid session_id")
    with httpx.Client(trust_env=False, timeout=30) as client:
        detail = get_json(client, url.rstrip("/") + "/api/sessions/" + session_id)
    return validate_session(detail)


def check_downloads(data_dir: Path, run_id: str) -> dict[str, Any]:
    require(
        bool(run_id) and Path(run_id).name == run_id and run_id not in {".", ".."}, "Invalid run_id"
    )
    directory = data_dir / "downloads" / run_id
    require(directory.is_dir(), "Run download directory not found")
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".pdf")
    require(bool(files), "No downloaded PDFs in this Run")
    result = []
    for path in files:
        require(not path.is_symlink(), "PDF must not be a symbolic link")
        with path.open("rb") as stream:
            require(stream.read(5) == b"%PDF-", f"Not a PDF: {path.name}")
        result.append({"name": path.name, "bytes": path.stat().st_size})
    return {"directory": str(directory), "files": result}


def check_artifacts(url: str, session_id: str, slides: int | None = None) -> dict[str, Any]:
    from examples.deepagents_ov_context.chat.artifacts import FileArtifact, valid_artifact_name
    from examples.deepagents_ov_context.chat.office_skills import validate_output

    require(bool(session_id) and "/" not in session_id, "Invalid session_id")
    base = url.rstrip("/")
    with httpx.Client(trust_env=False, timeout=30) as client:
        health = get_json(client, base + "/api/health")
        require(bool(health.get("skills")), "File Skills are not enabled")
        response = client.get(base + f"/api/sessions/{session_id}/artifacts")
        response.raise_for_status()
        items = [FileArtifact.model_validate(item) for item in response.json()]
        require(bool(items), "No artifacts in this session")
        result = []
        for item in items:
            require(valid_artifact_name(item.filename), "Invalid artifact filename")
            route = f"/api/sessions/{session_id}/artifacts/{item.run_id}/{item.filename}"
            require(item.url == route, "Unexpected artifact route")
            response = client.get(base + route)
            response.raise_for_status()
            require(len(response.content) == item.bytes, "Artifact size mismatch")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / item.filename
                path.write_bytes(response.content)
                count = validate_output(path)
                if item.kind == "presentation":
                    require(count == item.slides, "PPTX slide count mismatch")
                    if slides is not None:
                        require(count == slides, "Unexpected slide count")
            result.append(item.model_dump())
    return {"artifacts": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install")
    commands.add_parser("office-skills")
    pptx_skill = commands.add_parser("pptx-skill")
    pptx_skill.add_argument("--skill", required=True, type=Path)
    pptx_skill.add_argument("--tools", required=True, type=Path)
    browser = commands.add_parser("browser")
    browser.add_argument("--tools", type=Path, required=True)
    browser.add_argument("--chrome", type=Path, required=True)
    services = commands.add_parser("services")
    services.add_argument("--ov-url", default="http://127.0.0.1:1933")
    services.add_argument("--url", default="http://127.0.0.1:1934")
    services.add_argument("--browser", action="store_true", help="Require browser and Thinking")
    session = commands.add_parser("session")
    session.add_argument("session_id")
    session.add_argument("--url", default="http://127.0.0.1:1934")
    downloads = commands.add_parser("downloads")
    downloads.add_argument("run_id")
    downloads.add_argument("--data-dir", type=Path, required=True)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("session_id")
    artifacts.add_argument("--url", default="http://127.0.0.1:1934")
    artifacts.add_argument("--slides", type=int)
    args = parser.parse_args()
    try:
        if args.command == "install":
            result = check_install()
        elif args.command == "pptx-skill":
            result = check_pptx_skill(args.skill, args.tools)
        elif args.command == "office-skills":
            result = check_office_skills()
        elif args.command == "browser":
            result = check_browser(args.tools, args.chrome)
        elif args.command == "services":
            result = check_services(args.ov_url, args.url, args.browser)
        elif args.command == "session":
            result = check_session(args.url, args.session_id)
        elif args.command == "artifacts":
            result = check_artifacts(args.url, args.session_id, args.slides)
        else:
            result = check_downloads(args.data_dir, args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (
        CheckError,
        ImportError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
        httpx.HTTPError,
    ) as exc:
        # HTTP request errors can contain credentials; do not dump request bodies/config.
        print(
            f"Check failed: {exc}"
            if isinstance(exc, CheckError)
            else f"Check failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
