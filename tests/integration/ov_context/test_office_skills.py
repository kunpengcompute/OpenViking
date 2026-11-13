"""Office Skills exercise real files through native tools and Chat downloads."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from docx import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openpyxl import load_workbook
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from examples.deepagents_ov_context.chat.artifacts import list_artifacts
from examples.deepagents_ov_context.chat.office_skills import (
    BUNDLED_SKILLS,
    OfficeSkillRun,
    validate_output,
    validate_skill,
)
from examples.deepagents_ov_context.chat.server import ActiveRun

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events

GENERATE = """
from docx import Document
from openpyxl import Workbook
from reportlab.pdfgen.canvas import Canvas
d=Document(); d.add_heading('Sales summary', 0); d.add_paragraph('Total: 120 CNY')
d.save('output/summary.docx')
w=Workbook(); s=w.active; s.append(['Product','Quantity','Price','Amount'])
s.append(['A',2,30,'=B2*C2']); s.append(['B',3,20,'=B3*C3'])
s.append(['Total',None,None,'=SUM(D2:D3)']); w.save('output/sales.xlsx')
c=Canvas('output/summary.pdf'); c.drawString(72,720,'Total: 120 CNY'); c.save()
"""


def office_run(root: Path, run_id: str = "run-a") -> OfficeSkillRun:
    return OfficeSkillRun(
        {"office-documents": BUNDLED_SKILLS / "office-documents"},
        None,
        root,
        "session-a",
        run_id,
    )


def test_browser_only_backend_keeps_skills_and_blocks_host_files(tmp_path: Path) -> None:
    from deepagents.middleware.filesystem import FilesystemMiddleware

    secret = tmp_path / "fixture.sqlite3"
    secret.write_text("private fixture contents")
    run = OfficeSkillRun(
        {"agent-browser": BUNDLED_SKILLS / "agent-browser"},
        None,
        tmp_path / "workspaces",
        "browser-session",
        "browser-run",
        shell_enabled=False,
    )
    names = {tool.name for tool in FilesystemMiddleware(backend=run.backend).tools}
    assert "read_file" in names and "write_file" in names
    # The middleware filters execute from the model schema at bind time.
    assert not callable(getattr(run.backend, "aexecute", None))
    skill = run.backend.read("/skills/agent-browser/SKILL.md")
    assert skill.error is None and "agent-browser" in str(skill.file_data)
    outside = run.backend.read(str(secret))
    assert outside.error and outside.file_data is None
    with pytest.raises(ValueError, match="traversal"):
        run.backend._resolve_path("/../../fixture.sqlite3")
    (run.folder / "outside").symlink_to(secret)
    with pytest.raises(ValueError, match="outside root"):
        run.backend._resolve_path("/outside")
    assert "execute" not in run.instructions


def test_bundled_skill_names_and_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.deepagents_ov_context.scripts import check as deploy_check

    for name in ("office-documents", "data-analysis", "agent-browser"):
        validate_skill(BUNDLED_SKILLS / name, name)
    monkeypatch.setattr(deploy_check.shutil, "which", lambda name: None)
    with pytest.raises(deploy_check.CheckError, match="soffice"):
        deploy_check.check_office_skills()


async def test_real_office_files_publish_only_outputs_and_keep_runs_separate(
    tmp_path: Path,
) -> None:
    run = office_run(tmp_path)
    (run.folder / "internal.json").write_text('{"private":"intermediate"}')
    (run.folder / "generate.py").write_text(GENERATE)
    result = await run.backend.aexecute(f"{shlex.quote(sys.executable)} generate.py")
    assert result.exit_code == 0, result.output
    assert not list_artifacts(tmp_path, "session-a")
    artifacts = run.publish()
    assert {a.kind for a in artifacts} == {"document", "spreadsheet", "pdf"}
    assert all(a.slides is None for a in artifacts)
    assert len(list_artifacts(tmp_path, "session-a")) == 3
    assert not list_artifacts(tmp_path, "other-session")
    # Same filenames in a later Run preserve the earlier download and contents.
    second = office_run(tmp_path, "run-b")
    listed = second.file_tool().invoke({"action": "list"})
    assert len(listed["files"]) == 3
    loaded = second.file_tool().invoke(
        {"action": "load", "run_id": "run-a", "filename": "summary.docx"}
    )
    copied = second.folder / str(loaded["path"]).lstrip("/")
    assert copied.read_bytes() == (run.folder / "summary.docx").read_bytes()
    original = (run.folder / "summary.docx").read_bytes()
    document = Document(copied)
    document.add_paragraph("Revised in a new Run")
    document.save(copied)
    assert (run.folder / "summary.docx").read_bytes() == original
    outsider = OfficeSkillRun(
        {"office-documents": BUNDLED_SKILLS / "office-documents"},
        None,
        tmp_path,
        "other-session",
        "run-c",
    )
    with pytest.raises(FileNotFoundError):
        outsider.file_tool().invoke(
            {"action": "load", "run_id": "run-a", "filename": "summary.docx"}
        )
    (second.folder / "output/summary.docx").write_bytes((run.folder / "summary.docx").read_bytes())
    assert second.publish()[0].url != next(a.url for a in artifacts if a.kind == "document")
    assert len(list_artifacts(tmp_path, "session-a")) == 4
    assert all(a.filename != "internal.json" for a in list_artifacts(tmp_path, "session-a"))


@pytest.mark.parametrize(
    "filename", ["invalid.docx", "invalid.xlsx", "invalid.pdf", "invalid.json"]
)
def test_corrupt_outputs_publish_no_manifests(tmp_path: Path, filename: str) -> None:
    run = office_run(tmp_path)
    (run.folder / "output" / filename).write_bytes(b"not a valid file")
    with pytest.raises((ValueError, zipfile.BadZipFile, PdfReadError)):
        run.publish()
    assert not list_artifacts(tmp_path, "session-a")


def test_json_nonfinite_values_are_not_published(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="JSON constant"):
        validate_output(path)


@pytest.mark.parametrize(
    "content",
    ['city,aqi\n"Hangzhou,92\n', 'city,aqi\nHangzhou,92\n"Nanjing,102\n'],
)
def test_csv_corrupt_data_rows_are_not_published(tmp_path: Path, content: str) -> None:
    run = office_run(tmp_path)
    (run.folder / "output/report.csv").write_text(content)
    with pytest.raises(csv.Error, match="unexpected end of data"):
        run.publish()
    assert not list_artifacts(tmp_path, "session-a")


def test_csv_quoted_multiline_records_are_published(tmp_path: Path) -> None:
    run = office_run(tmp_path)
    content = 'city,note\nHangzhou,"Line one\nLine two, with a comma"\n'
    (run.folder / "output/report.csv").write_text(content)
    artifacts = run.publish()
    assert len(artifacts) == 1 and artifacts[0].kind == "data"
    assert (run.folder / "report.csv").read_text() == content


def test_empty_csv_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        validate_output(path)


def test_failed_batch_keeps_all_files_unpublished(tmp_path: Path) -> None:
    run = office_run(tmp_path)
    (run.folder / "output/valid.txt").write_text("ready")
    (run.folder / "output/invalid.json").write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        run.publish()
    assert not list_artifacts(tmp_path, "session-a")


def test_data_analysis_handles_zero_null_and_quoted_filenames(tmp_path: Path) -> None:
    path = BUNDLED_SKILLS / "data-analysis/scripts/analyze.py"
    spec = importlib.util.spec_from_file_location("office_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "sales' records.csv"
    data.write_text("category,amount\na,0\na,60\nb,60\nb,\n")
    inspected = module.analyze([data], "inspect")
    table, evidence = next(iter(inspected.items()))
    assert evidence["rows"] == 4 and evidence["nulls"]["amount"] == 1
    assert evidence["sample"][0]["amount"] == 0
    result = module.analyze([data], "query", f'SELECT SUM(amount) AS total FROM "{table}"')
    assert result["records"] == [{"total": 120.0}]
    with pytest.raises(ValueError, match="SELECT"):
        module.analyze([data], "query", 'DELETE FROM "sales"')
    # Multiple inputs with equal stems remain independently queryable.
    other = tmp_path / "other"
    other.mkdir()
    copy = other / data.name
    copy.write_text(data.read_text())
    assert len(module.load_tables([data, copy])) == 2
    output = tmp_path / "output/empty.csv"
    executed = subprocess.run(
        [
            sys.executable,
            str(path),
            "--files",
            str(data),
            "--action",
            "query",
            "--sql",
            f'SELECT amount FROM "{table}" WHERE amount < 0',
            "--output-file",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert json.loads(executed.stdout)["rows"] == 0
    assert output.read_text(encoding="utf-8-sig").strip() == "amount"


class OfficeModel(NoToolModel):
    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            assert "office-documents" in str(messages[0].content)
            name, args = (
                "write_todos",
                {"todos": [{"content": "Prepare sales files", "status": "in_progress"}]},
            )
        elif len(results) == 1:
            name, args = "read_file", {"file_path": "/skills/office-documents/SKILL.md"}
        elif len(results) == 2:
            assert "办公文件处理" in str(results[-1].content)
            name, args = "write_file", {"file_path": "/generate.py", "content": GENERATE}
        elif len(results) == 3:
            name, args = "execute", {"command": f"{shlex.quote(sys.executable)} generate.py"}
        elif len(results) == 4:
            name, args = (
                "write_todos",
                {"todos": [{"content": "Prepare sales files", "status": "completed"}]},
            )
        else:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Files ready"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[{"id": f"office-{len(results)}", "name": name, "args": args}],
                    )
                )
            ]
        )


@pytest.mark.parametrize("mode", ["baseline", "current", "context_only"])
async def test_chat_discovers_skill_generates_and_downloads_office_files(
    chat: Any, env: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    client, runtime = chat
    if mode != "baseline":
        server = env[0]
        await runtime.ov_http.aclose()

        async def transport(request: httpx.Request) -> httpx.Response:
            response = await server.handle(request)
            if request.url.path.endswith("/context"):
                body = response.json()
                body["result"]["latest_finished_archive_id"] = server.archive_id
                return httpx.Response(200, json=body)
            return response

        runtime.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    # The generation contract uses actual Python packages; renderer availability
    # is separately checked by the deployment test, independent of this fixture.
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )
    runtime.settings = replace(runtime.settings, office_skills=True)
    runtime.model = OfficeModel()
    response = await client.post(
        "/api/sessions", json={"execution": {"mode": mode, "recursion_limit": 100}}
    )
    sid = response.json()["session_id"]
    if mode != "baseline":
        env[0].sessions.add(sid)
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "Prepare sales files"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "done", events[-1]
    artifacts = events[-1][1]["artifacts"]
    assert len(artifacts) == 3
    for item in artifacts:
        downloaded = await client.get(item["url"])
        assert downloaded.status_code == 200
        assert len(downloaded.content) == item["bytes"]
        data = io.BytesIO(downloaded.content)
        if item["kind"] == "document":
            assert "Total: 120 CNY" in "\n".join(p.text for p in Document(data).paragraphs)
            assert "wordprocessingml.document" in downloaded.headers["content-type"]
        elif item["kind"] == "spreadsheet":
            book = load_workbook(data)
            assert book.active["D4"].value == "=SUM(D2:D3)"
            assert "spreadsheetml.sheet" in downloaded.headers["content-type"]
            book.close()
        else:
            assert "Total: 120 CNY" in PdfReader(data).pages[0].extract_text()
            assert downloaded.headers["content-type"] == "application/pdf"
    # Refresh uses persisted manifests rather than model response text.
    assert len((await client.get(f"/api/sessions/{sid}/artifacts")).json()) == 3
    state = await runtime.state(runtime.store.get(sid))
    assert state["todos"] == [{"content": "Prepare sales files", "status": "completed"}]
    health = (await client.get("/api/health")).json()
    assert health["skills"] == ["data-analysis", "office-documents"]
    assert "browser" not in json.dumps(health["skills"])


@pytest.mark.parametrize("mode", ["baseline", "current", "context_only"])
@pytest.mark.parametrize(
    "filename,error_type", [("report.pptx", "BadZipFile"), ("report.csv", "Error")]
)
async def test_artifact_failure_preserves_history_and_allows_regeneration(
    chat: Any,
    env: Any,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    filename: str,
    error_type: str,
) -> None:
    client, runtime = chat
    if mode != "baseline":
        server = env[0]
        await runtime.ov_http.aclose()

        async def transport(request: httpx.Request) -> httpx.Response:
            response = await server.handle(request)
            if request.url.path.endswith("/context"):
                body = response.json()
                body["result"]["latest_finished_archive_id"] = server.archive_id
                return httpx.Response(200, json=body)
            return response

        runtime.ov_http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )
    runtime.settings = replace(runtime.settings, office_skills=True)
    runtime.model = NoToolModel()
    publish = OfficeSkillRun.publish
    attempts = 0

    def generated_output(run: OfficeSkillRun):
        nonlocal attempts
        attempts += 1
        path = run.folder / "output" / filename
        if attempts == 1:
            path.write_text(
                "broken presentation" if path.suffix == ".pptx" else 'city,aqi\n"Hangzhou,92\n'
            )
        elif path.suffix == ".pptx":
            from pptx import Presentation

            deck = Presentation()
            deck.slides.add_slide(deck.slide_layouts[6])
            deck.save(path)
        else:
            path.write_text("city,aqi\nHangzhou,92\n")
        return publish(run)

    monkeypatch.setattr(OfficeSkillRun, "publish", generated_output)
    created = await client.post("/api/sessions", json={"execution": {"mode": mode}})
    sid = created.json()["session_id"]
    if mode != "baseline":
        env[0].sessions.add(sid)
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "Generate a report"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    error = events[-1][1]
    assert error["stage"] == "artifact_publish" and error["can_continue"]
    assert error["error_type"] == error_type
    assert "会话已保存" in error["message"] and "恢复" not in error["message"]
    session = runtime.store.get(sid)
    assert not session.needs_restore
    state = await runtime.state(session)
    original_ids = [message.id for message in state["messages"]]
    assert len(original_ids) == 2
    if mode != "baseline":
        assert state["ov_context"]["run_id"] is None
        assert state["ov_context"]["write_index"] == 2
    assert not (await client.get(f"/api/sessions/{sid}/artifacts")).json()

    # A fresh graph continues the same checkpoint; no restore/reset is needed.
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "Regenerate the report correctly"}
    )
    assert response.status_code == 200
    events = parse_events(response.text)
    assert events[-1][0] == "done", events[-1]
    artifact = events[-1][1]["artifacts"][0]
    assert artifact["filename"] == filename
    assert (await client.get(artifact["url"])).status_code == 200
    current = await runtime.state(runtime.store.get(sid))
    assert len(current["messages"]) == 4
    assert [message.id for message in current["messages"][:2]] == original_ids
    assert not runtime.store.get(sid).needs_restore


async def test_checkpoint_skill_metadata_refreshes_after_agent_rebuild(
    chat: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, runtime = chat
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )
    runtime.settings = replace(runtime.settings, office_skills=True)
    runtime.model = NoToolModel()
    created = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    sid = created.json()["session_id"]
    session = runtime.store.get(sid)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    assert parse_events(response.text)[-1][0] == "done"
    sdk = runtime.sdk()
    skills = OfficeSkillRun(
        {"office-documents": BUNDLED_SKILLS / "office-documents"},
        None,
        tmp_path / "inspect",
        sid,
        "inspect",
    )
    try:
        graph, _ = runtime.session_graph(session, sdk, ActiveRun(), skill_run=skills)
        await graph.aupdate_state(
            session.graph_config,
            {"skills_metadata": [], "skills_load_errors": ["obsolete deployment error"]},
            as_node="SkillsMiddleware.before_agent",
        )
    finally:
        await sdk.close()
    # A fresh Agent / workspace must discover enabled Skills, even if an older
    # checkpoint has a cached empty list and an earlier deployment error.
    response = await client.post("/api/chat", json={"session_id": sid, "message": "hello again"})
    assert parse_events(response.text)[-1][0] == "done"
    saved = await runtime.saver.aget_tuple(session.graph_config)
    assert saved is not None
    state = saved.checkpoint["channel_values"]
    assert {item["name"] for item in state["skills_metadata"]} == {
        "office-documents",
        "data-analysis",
    }
    assert state["skills_load_errors"] == []


def test_official_browser_skill_keeps_upstream_bytes_and_native_discovery(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace

    from examples.deepagents_ov_context.chat.office_skills import RunSkillsMiddleware

    source = BUNDLED_SKILLS / "agent-browser"
    manifest = json.loads((source / "upstream.json").read_text())
    for name, item in manifest["files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == item["sha256"]
    run = OfficeSkillRun(
        {"agent-browser": source}, None, tmp_path, "skill-session", "skill-run", shell_enabled=False
    )
    middleware = RunSkillsMiddleware(backend=run.backend, sources=["/skills/"])
    update = middleware.before_agent({}, SimpleNamespace(context=None), {})
    assert update and not update.get("skills_load_errors")
    assert update["skills_metadata"][0]["name"] == "agent-browser"
    assert "filling forms" in update["skills_metadata"][0]["description"]
