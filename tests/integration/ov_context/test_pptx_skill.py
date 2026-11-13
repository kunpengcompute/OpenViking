"""Native Skill discovery/execution, artifact publication and cancellation."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pptx import Presentation

from examples.deepagents_ov_context.chat.artifacts import (
    FileArtifact,
    artifact_directory,
    list_artifacts,
)
from examples.deepagents_ov_context.chat.office_skills import OfficeSkillRun, validate_skill

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    path = tmp_path / "official-skill"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: pptx\ndescription: Create editable presentations with scripts.\n---\n"
        "Read sources, generate a deck, validate its contents and render it."
    )
    return path


def test_skill_requires_named_frontmatter_and_regular_files(skill: Path, tmp_path: Path) -> None:
    validate_skill(skill)
    with pytest.raises(ValueError, match="missing SKILL"):
        validate_skill(tmp_path / "missing")
    (skill / "SKILL.md").write_text("---\nname: other\n---\n")
    with pytest.raises(ValueError, match="name: pptx"):
        validate_skill(skill)


@pytest.mark.parametrize("available", [True, False])
def test_skill_deployment_check_reports_node_dependencies(
    skill: Path, monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    from examples.deepagents_ov_context.scripts import check as deploy_check

    monkeypatch.setattr(deploy_check.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(deploy_check.shutil, "which", lambda name: f"/usr/bin/{name}")

    def version(command: list[str]) -> str:
        if not available:
            raise subprocess.CalledProcessError(1, command)
        return "PPTX Skill dependencies OK"

    monkeypatch.setattr(deploy_check, "version", version)
    if available:
        assert deploy_check.check_pptx_skill(skill, skill)["node"] == "PPTX Skill dependencies OK"
    else:
        with pytest.raises(deploy_check.CheckError, match="rerun install-pptx-skill"):
            deploy_check.check_pptx_skill(skill, skill)


def test_artifacts_are_scoped_and_reject_symlinks(tmp_path: Path) -> None:
    folder = artifact_directory(tmp_path, "session-a", "run-a")
    folder.mkdir(parents=True)
    (folder / "intro.pptx").write_bytes(b"file")
    artifact = FileArtifact(
        filename="intro.pptx", run_id="run-a", url="https://untrusted.invalid/", slides=10, bytes=4
    )
    (folder / "intro.artifact.json").write_text(artifact.model_dump_json())
    assert list_artifacts(tmp_path, "session-b") == []
    assert (
        list_artifacts(tmp_path, "session-a")[0].url
        == "/api/sessions/session-a/artifacts/run-a/intro.pptx"
    )
    (folder / "intro.pptx").unlink()
    (folder / "intro.pptx").symlink_to(folder / "intro.artifact.json")
    assert list_artifacts(tmp_path, "session-a") == []
    (tmp_path / "session-b").symlink_to(tmp_path / "session-a")
    with pytest.raises(ValueError):
        artifact_directory(tmp_path, "session-b", "run-a")
    with pytest.raises(ValueError):
        artifact_directory(tmp_path, "../outside", "run-a")


async def test_execute_generates_real_deck_and_publish_preserves_owner(
    skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "must-not-reach-skill")
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "session-a", "run-a")
    script = (
        "import os; from pptx import Presentation; "
        "assert 'MODEL_API_KEY' not in os.environ; "
        "p=Presentation(); p.slides.add_slide(p.slide_layouts[0]); p.save('output/intro.pptx')"
    )
    result = await run.backend.aexecute(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    assert result.exit_code == 0, result.output
    assert list_artifacts(tmp_path / "artifacts", "session-a") == []
    published = run.publish()
    assert len(published) == 1 and published[0].slides == 1
    assert len(Presentation(run.folder / "intro.pptx").slides) == 1
    assert list_artifacts(tmp_path / "artifacts", "session-a") == published
    assert list_artifacts(tmp_path / "artifacts", "session-b") == []
    assert run.publish() == published
    assert run.last_published == []


def test_root_output_is_validated_and_published(skill: Path, tmp_path: Path) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[0])
    deck.save(run.folder / "intro.pptx")
    assert list_artifacts(tmp_path / "artifacts", "s") == []
    published = run.publish()
    assert len(published) == 1 and published[0].slides == 1
    assert list_artifacts(tmp_path / "artifacts", "s") == published


async def test_shell_paths_and_virtual_file_paths_share_the_workspace(
    skill: Path, tmp_path: Path
) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    script = "from pathlib import Path; Path('output/proof.txt').write_text('same workspace')"
    result = run.backend.write(str(run.folder / "generate.py"), script)
    assert result.error is None
    assert (run.folder / "generate.py").read_text() == script
    executed = await run.backend.aexecute(f"{shlex.quote(sys.executable)} generate.py")
    assert executed.exit_code == 0
    assert run.backend.read("/output/proof.txt") == run.backend.read(
        str(run.folder / "output/proof.txt")
    )
    with pytest.raises(ValueError, match="traversal"):
        run.backend._resolve_path(str(run.folder) + "/../outside.txt")


def test_duplicate_output_names_are_rejected(skill: Path, tmp_path: Path) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    for directory in [run.folder, run.folder / "output"]:
        deck = Presentation()
        deck.slides.add_slide(deck.slide_layouts[0])
        deck.save(directory / "intro.pptx")
    with pytest.raises(ValueError, match="filenames must be unique"):
        run.publish()
    assert list_artifacts(tmp_path / "artifacts", "s") == []


@pytest.mark.parametrize("bad", ["corrupt", "symlink", "directory-symlink"])
def test_invalid_outputs_are_never_published(skill: Path, tmp_path: Path, bad: str) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    output = run.folder / "output"
    if bad == "directory-symlink":
        output.rmdir()
        output.symlink_to(tmp_path)
    elif bad == "symlink":
        (output / "intro.pptx").symlink_to(skill / "SKILL.md")
    else:
        (output / "intro.pptx").write_bytes(b"not a presentation")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        run.publish()
    assert list_artifacts(tmp_path / "artifacts", "s") == []


async def test_cancel_stops_command_process_group(skill: Path, tmp_path: Path) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    script = "import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(60)"
    task = asyncio.create_task(
        run.backend.aexecute(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    )
    for _ in range(200):
        if (run.folder / "pid").exists():
            break
        await asyncio.sleep(0.01)
    assert (run.folder / "pid").exists()
    pid = int((run.folder / "pid").read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Skill command survived cancellation")
    assert list_artifacts(tmp_path / "artifacts", "s") == []


async def test_execute_timeout_and_output_limit(skill: Path, tmp_path: Path) -> None:
    run = OfficeSkillRun({"pptx": skill}, None, tmp_path / "artifacts", "s", "r")
    timed = await run.backend.aexecute("sleep 30", timeout=1)
    assert timed.exit_code == 124
    command = f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 200000)'"
    long = await run.backend.aexecute(command)
    assert long.exit_code == 0 and long.truncated and len(long.output) == 100000


class SkillModel(NoToolModel):
    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        tools = [message for message in messages if isinstance(message, ToolMessage)]
        if len(tools) == 0:
            assert "pptx" in str(messages[0].content)
            name, args = "read_file", {"file_path": "/skills/pptx/SKILL.md"}
        elif len(tools) == 1:
            assert "Create editable presentations" in str(tools[-1].content)
            script = "from pptx import Presentation\np=Presentation()\np.slides.add_slide(p.slide_layouts[0])\np.save('output/intro.pptx')\n"
            name, args = "write_file", {"file_path": "/generate.py", "content": script}
        elif len(tools) == 2:
            name, args = "execute", {"command": f"{shlex.quote(sys.executable)} generate.py"}
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="PPT ready"))])
        message = AIMessage(
            content="", tool_calls=[{"id": f"tool-{len(tools)}", "name": name, "args": args}]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.parametrize("mode", ["baseline", "current", "context_only"])
async def test_chat_native_skill_executes_and_downloads_without_ppt_tool(
    chat: Any, skill: Path, env: Any, mode: str
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
    # Four model turns plus the OV hooks exceed the framework's 24-step default.
    runtime.settings = replace(runtime.settings, pptx_skill=skill, recursion_limit=64)
    runtime.model = SkillModel()
    created = await client.post("/api/sessions", json={"execution": {"mode": mode}})
    sid = created.json()["session_id"]
    if mode != "baseline":
        env[0].sessions.add(sid)
    response = await client.post("/api/chat", json={"session_id": sid, "message": "make a PPT"})
    events = parse_events(response.text)
    assert events[-1][0] == "done", events[-1]
    artifacts = events[-1][1]["artifacts"]
    assert len(artifacts) == 1 and artifacts[0]["slides"] == 1
    download = await client.get(artifacts[0]["url"])
    assert download.status_code == 200 and download.content[:2] == b"PK"
    names = [event[1]["message"].get("name") for event in events if event[0] == "message"]
    assert "execute" in names and "create_presentation" not in names
    assert len((await client.get(f"/api/sessions/{sid}/artifacts")).json()) == 1
