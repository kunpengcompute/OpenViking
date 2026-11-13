# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run office Skills through the native DeepAgents filesystem and shell backend."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.middleware.skills import SkillsMiddleware, SkillsState, SkillsStateUpdate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from pptx import Presentation

from examples.deepagents_ov_context.chat.artifacts import (
    ARTIFACT_TYPES,
    FileArtifact,
    artifact_directory,
    list_artifacts,
    valid_artifact_name,
    valid_input_name,
)

BUNDLED_SKILLS = Path(__file__).with_name("skills")


class RunSkillsMiddleware(SkillsMiddleware):
    """Refresh native Skill metadata for the current per-Run workspace."""

    @property
    def name(self) -> str:
        # Override the native slot through DeepAgents' public middleware mechanism.
        return "SkillsMiddleware"

    def before_agent(
        self, state: SkillsState, runtime: Runtime[Any], config: RunnableConfig
    ) -> SkillsStateUpdate | None:
        fresh = state.copy()
        fresh.pop("skills_metadata", None)
        fresh.pop("skills_load_errors", None)
        update = super().before_agent(fresh, runtime, config)
        if update is not None:
            update.setdefault("skills_load_errors", [])
        return update

    async def abefore_agent(
        self, state: SkillsState, runtime: Runtime[Any], config: RunnableConfig
    ) -> SkillsStateUpdate | None:
        fresh = state.copy()
        fresh.pop("skills_metadata", None)
        fresh.pop("skills_load_errors", None)
        update = await super().abefore_agent(fresh, runtime, config)
        if update is not None:
            update.setdefault("skills_load_errors", [])
        return update


def validate_skill(path: Path, name: str = "pptx") -> None:
    """Require an installed, named Skill rather than a generation tool."""
    if not (path / "SKILL.md").is_file():
        raise ValueError(f"{name} Skill is missing SKILL.md")
    header = (path / "SKILL.md").read_text(encoding="utf-8").split("---", 2)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Invalid Skill name")
    if len(header) < 3 or not re.search(rf"^name:\s*{re.escape(name)}\s*$", header[1], re.M):
        raise ValueError(f"Expected a Skill with name: {name}")
    if any(item.is_symlink() for item in [path, *path.rglob("*")]):
        raise ValueError("Skill files must be ordinary files and directories")


class SkillBackend(LocalShellBackend):
    """Local test-container execution, with cancellable process groups.

    A per-Run cwd is a workspace, not an OS security boundary. Deploy this
    optional capability only in the dedicated Chat test container.
    """

    def _resolve_path(self, key: str) -> Path:
        """Accept shell-reported paths inside this workspace as file-tool paths."""
        root = str(self.cwd)
        if key == root or key.startswith(root + "/"):
            key = key[len(root) :] or "/"
        return super()._resolve_path(key)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.cwd),
            env=self._env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        chunks: list[bytes] = []
        captured = 0
        truncated = False

        async def drain() -> None:
            nonlocal captured, truncated
            assert process.stdout is not None
            while chunk := await process.stdout.read(8192):
                remaining = max(0, self._max_output_bytes - captured)
                if remaining:
                    chunks.append(chunk[:remaining])
                captured += min(len(chunk), remaining)
                truncated |= len(chunk) > remaining
            await process.wait()

        try:
            await asyncio.wait_for(drain(), effective_timeout)
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ExecuteResponse(output="Skill command timed out", exit_code=124)
        return ExecuteResponse(
            output=b"".join(chunks).decode("utf-8", errors="replace"),
            exit_code=process.returncode,
            truncated=truncated,
        )


class OfficeSkillRun:
    def __init__(
        self,
        skills: Mapping[str, Path],
        node_modules: Path | None,
        root: Path,
        session_id: str,
        run_id: str,
        *,
        shell_enabled: bool = True,
        input_dir: Path | None = None,
    ) -> None:
        if not skills:
            raise ValueError("At least one Skill is required")
        for name, path in skills.items():
            validate_skill(path, name)
        self.skill_names = frozenset(skills)
        self.shell_enabled = shell_enabled
        self.folder = artifact_directory(root, session_id, run_id)
        self.root = root
        self.session_id, self.run_id = session_id, run_id
        self._published_hashes: dict[str, str] = {}
        self.last_published: list[dict[str, Any]] = []
        self.folder.mkdir(parents=True, exist_ok=False)
        for name, path in skills.items():
            shutil.copytree(path, self.folder / "skills" / name)
        (self.folder / "output").mkdir()
        (self.folder / "tmp").mkdir()
        if input_dir is not None and input_dir.exists():
            if input_dir.is_symlink():
                raise ValueError("Input directory must not be a symlink")
            (self.folder / "inputs").mkdir()
            for source in sorted(input_dir.iterdir()):
                if source.is_symlink() or not source.is_file() or not valid_input_name(source.name):
                    raise ValueError("Invalid input file")
                shutil.copyfile(source, self.folder / "inputs" / source.name)
        env = {
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.environ.get("PATH", os.defpath)]
            ),
            "TMPDIR": str((self.folder / "tmp").resolve()),
            "LANG": "C.UTF-8",
            "XDG_CACHE_HOME": str((self.folder / "tmp/cache").resolve()),
            "XDG_CONFIG_HOME": str((self.folder / "tmp/config").resolve()),
        }
        if node_modules is not None:
            env["NODE_PATH"] = str(node_modules.resolve())
        self.backend = (
            SkillBackend(root_dir=self.folder, env=env, inherit_env=False)
            if shell_enabled
            else FilesystemBackend(root_dir=self.folder, virtual_mode=True)
        )

    @property
    def instructions(self) -> str:
        if not self.shell_enabled:
            return (
                f"本轮已加载 Skill：{', '.join(sorted(self.skill_names))}。"
                "任务匹配时先读取 /skills/技能名/SKILL.md。"
                "工作区虚拟根目录为 /，文件通过原生文件工具读写；"
                "网页任务通过 browser 工具完成，交付文件保存到 /output/。"
            )
        return (
            f"本轮已加载 Skill：{', '.join(sorted(self.skill_names))}。"
            "任务匹配时先读取 /skills/技能名/SKILL.md，"
            "按需使用原生文件工具与 execute 完成生成及质量检查。"
            f"本轮工作区的物理根目录是 {self.folder.resolve()}。"
            "文件工具中的虚拟路径 / 对应该物理根目录；"
            "例如 write_file('/generate.js') 创建的文件，"
            "在 execute 中用 node generate.js 执行。"
            "execute 从此物理根目录执行；"
            "脚本路径使用 skills/技能名/scripts/...，最终交付文件写到 output/，"
            "中间文件和预览放到 tmp/。"
            "本会话上传的文件已复制到本轮 inputs/。"
            "文件工具也接受本轮工作区内的物理绝对路径，"
            "可直接用渲染脚本返回的路径调用 read_file 查看图片。"
            "最终文件名使用英文字母、数字、下划线或连字符，"
            "支持 pptx、docx、xlsx、pdf、csv、json、md、txt、png 扩展名。"
            "完成后交付下载链接："
            f"/api/sessions/{self.session_id}/artifacts/{self.run_id}/文件名。"
            "下载链接在本轮成功完成并发布文件后可用。"
            "修改本会话以前生成的文件时，通过 session_files(action='list') 列出已发布文件，"
            "再用 action='load' 和返回的 run_id、filename 将所需文件加载到本轮 inputs/，"
            "读取并修改副本后交付新的 output/ 文件。"
            "网页引用和评测条件由本次查证取得。"
        )

    def file_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.session_files,
            name="session_files",
            description=(
                "列出或加载本会话已发布的交付文件。action='list' 返回 run_id、filename 和下载信息；"
                "action='load' 需提供 run_id、filename，将文件副本放到本轮 inputs/，返回文件工具可读的路径。"
            ),
        )

    def session_files(
        self,
        action: Literal["list", "load"],
        run_id: str | None = None,
        filename: str | None = None,
    ) -> dict[str, object]:
        items = list_artifacts(self.root, self.session_id)
        if action == "list":
            return {"files": [item.model_dump() for item in items]}
        if action != "load" or not run_id or not filename:
            raise ValueError("Use list, or load with run_id and filename")
        if not any(item.run_id == run_id and item.filename == filename for item in items):
            raise FileNotFoundError("Published file not found in this session")
        source = artifact_directory(self.root, self.session_id, run_id) / filename
        directory = self.folder / "inputs" / run_id
        target = directory / filename
        if any(path.is_symlink() for path in (directory.parent, directory, target)):
            raise ValueError("Invalid input path")
        directory.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with source.open("rb") as original, target.open("xb") as destination:
                shutil.copyfileobj(original, destination)
        return {"path": f"/inputs/{run_id}/{filename}", "bytes": target.stat().st_size}

    def publish(self) -> list[FileArtifact]:
        """Publish complete output snapshots, including progress before Run completion."""
        self.last_published = []
        output_dir = self.folder / "output"
        if output_dir.is_symlink():
            raise ValueError("Invalid output directory")
        if any(self.folder.glob("*.artifact.json")) and not self._published_hashes:
            raise FileExistsError("This Run's artifacts have already been published")
        artifacts: list[FileArtifact] = []
        files: list[Path] = []
        # Only output/ is published, except the pre-existing root PPTX convention.
        candidates = sorted(
            [p for p in output_dir.iterdir() if p.suffix in ARTIFACT_TYPES]
            + [
                p
                for p in self.folder.glob("*.pptx")
                if not (p.name in self._published_hashes and (output_dir / p.name).exists())
            ]
        )
        names = [path.name for path in candidates]
        if len(names) != len(set(names)):
            raise ValueError("Artifact output filenames must be unique")
        for path in candidates:
            if path.is_symlink() or not path.is_file() or not valid_artifact_name(path.name):
                raise ValueError("Invalid artifact output path")
            if (
                path.parent == output_dir
                and (self.folder / path.name).exists()
                and path.name not in self._published_hashes
            ):
                raise FileExistsError("Artifact target already exists")
            slides = validate_output(path)
            files.append(path)
            artifacts.append(
                FileArtifact(
                    filename=path.name,
                    run_id=self.run_id,
                    url=f"/api/sessions/{self.session_id}/artifacts/{self.run_id}/{path.name}",
                    slides=slides,
                    bytes=path.stat().st_size,
                    kind=ARTIFACT_TYPES[path.suffix][0],
                )
            )
        for path, artifact in zip(files, artifacts, strict=True):
            target = self.folder / path.name
            manifest = target.with_name(target.name + ".artifact.json")
            if target.is_symlink() or manifest.is_symlink():
                raise ValueError("Invalid published artifact path")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name in self._published_hashes:
                if not target.is_file() or not manifest.is_file():
                    raise ValueError("Published artifact snapshot is missing")
                if (
                    path != target
                    and hashlib.sha256(target.read_bytes()).hexdigest()
                    != self._published_hashes[path.name]
                ):
                    raise ValueError("Published artifact snapshot was changed outside output/")
                if self._published_hashes[path.name] == digest:
                    continue
            if path != target:
                temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex)
                try:
                    with temporary.open("xb") as destination, path.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            temporary_manifest = manifest.with_name("." + manifest.name + "." + uuid.uuid4().hex)
            try:
                temporary_manifest.write_text(artifact.model_dump_json(), encoding="utf-8")
                os.replace(temporary_manifest, manifest)
            finally:
                temporary_manifest.unlink(missing_ok=True)
            self._published_hashes[path.name] = digest
            self.last_published.append({**artifact.model_dump(), "sha256": digest})
        return artifacts


def validate_output(path: Path) -> int | None:
    """Parse actual files before publishing; content/layout review belongs to the Skill."""
    if not path.stat().st_size:
        raise ValueError("Artifact must not be empty")
    suffix = path.suffix
    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Expected PNG image")
            image.verify()
        return None
    if suffix in {".pptx", ".docx", ".xlsx"}:
        expected = {
            ".pptx": "ppt/presentation.xml",
            ".docx": "word/document.xml",
            ".xlsx": "xl/workbook.xml",
        }[suffix]
        with zipfile.ZipFile(path) as package:
            if expected not in package.namelist() or package.testzip():
                raise ValueError("Invalid Office package")
        if suffix == ".pptx":
            slides = len(Presentation(path).slides)
            if slides < 1:
                raise ValueError("Presentation must contain slides")
            return slides
        if suffix == ".docx":
            from docx import Document

            Document(path)
        else:
            from openpyxl import load_workbook

            book = load_workbook(path, read_only=True)
            try:
                if not book.sheetnames:
                    raise ValueError("Spreadsheet must contain a sheet")
            finally:
                book.close()
    elif suffix == ".pdf":
        from pypdf import PdfReader

        if not PdfReader(path, strict=True).pages:
            raise ValueError("PDF must contain pages")
    elif suffix == ".json":

        def invalid_constant(value: str) -> None:
            raise ValueError(f"Invalid JSON constant: {value}")

        json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = csv.reader(stream, strict=True)
            if next(records, None) is None:
                raise ValueError("CSV must contain a row")
            for _ in records:
                pass
    else:
        path.read_text(encoding="utf-8")
    return None
