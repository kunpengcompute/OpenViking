# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run the optional agent-browser CLI in one isolated browser session per Run."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import signal
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.tools import StructuredTool

from examples.deepagents_ov_context.chat.browser_guides import correct_browser_guide


class AgentBrowser:
    def __init__(
        self,
        executable: Path,
        session: str,
        *,
        chrome: Path | None = None,
        proxy: str | None = None,
        timeout: float = 50,
        download_dir: Path | None = None,
        screenshot_dir: Path | None = None,
    ) -> None:
        self.prefix = [str(executable.resolve()), "--session", session, "--content-boundaries"]
        self.launch_args: list[str] = []
        # The CLI resolves context options on every command. Keep them identical
        # after startup as well; a changed proxy/UA can recreate the browser.
        self._environment: dict[str, str] = os.environ.copy()
        if chrome is not None:
            self.launch_args += ["--executable-path", str(chrome.resolve())]
            self._environment["AGENT_BROWSER_EXECUTABLE_PATH"] = str(chrome.resolve())
        if proxy is not None:
            self.launch_args += ["--proxy", proxy]
            self._environment["AGENT_BROWSER_PROXY"] = proxy
        self.timeout = timeout
        self.legacy_file_aliases = False
        self.lock = asyncio.Lock()
        self.used = False
        self.download_dir = download_dir.resolve() if download_dir is not None else None
        self.downloads: set[str] = set()
        self.screenshot_dir = screenshot_dir.resolve() if screenshot_dir is not None else None

    def tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=self.command,
            name="browser",
            description="Browser automation CLI for AI agents.",
        )

    def file_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=self.file_command,
            name="browser_files",
            description=(
                "Chat 文件交付扩展。download_pdf 按 target URL 或页面元素下载到本轮附件目录；"
                "pdf_text 提取本轮已下载 PDF 的正文；screenshot 将当前页面截图保存到"
                "本轮 /output/filename，可用 full=true 截取整页。filename 为单个文件名。"
                "此工具独立于官方 agent-browser 命令。"
            ),
        )

    async def file_command(
        self,
        action: Literal["download_pdf", "pdf_text", "screenshot"],
        filename: str,
        target: str | None = None,
        full: bool = False,
    ) -> str:
        if action == "screenshot":
            return await self._screenshot(["screenshot", filename, *(["--full"] if full else [])])
        if action == "pdf_text":
            return await self._attachment(["pdf_text", filename])
        if action == "download_pdf" and target:
            return await self._attachment(["download", target, filename])
        return json.dumps({"error": "download_pdf requires target"})

    @staticmethod
    def _scope_error(argv: list[str]) -> str | None:
        # Command semantics belong to the CLI. Only the application-owned Run
        # connection and host lifecycle remain outside model control.
        if not argv:
            return "argv must contain an agent-browser command"
        reserved = {
            "--session",
            "--session-name",
            "--profile",
            "--cdp",
            "--auto-connect",
            "--config",
            "--proxy",
            "--executable-path",
            "--allow-file-access",
        }
        if any(arg.split("=", 1)[0] in reserved for arg in argv) or (
            argv[0] != "wait" and any(arg.split("=", 1)[0] == "--state" for arg in argv)
        ):
            return "Browser session and launch configuration are managed by Chat"
        if argv[0] in {"connect", "attach", "session", "install", "upgrade"} or (
            argv[0] == "close" and "--all" in argv
        ):
            return "Browser connection and host lifecycle are managed by Chat"
        if argv[0] in {"open", "goto", "navigate"}:
            urls = [
                arg
                for arg in argv[1:]
                if "://" in arg or arg.startswith(("file:", "data:", "about:"))
            ]
            if any(urlsplit(url).scheme not in {"http", "https"} for url in urls):
                return "Chat browser navigation uses HTTP(S) URLs"
        if argv[0] == "batch":
            for command in argv[1:]:
                if command in {"--bail", "--json"}:
                    continue
                try:
                    error = AgentBrowser._scope_error(shlex.split(command))
                except ValueError:
                    return "Invalid quoted batch command"
                if error:
                    return error
        return None

    async def command(self, argv: list[str], stdin: str | None = None) -> str:
        if self.legacy_file_aliases and stdin is None:
            if len(argv) == 2 and argv[0] == "pdf_text":
                return await self._attachment(argv)
            if len(argv) == 3 and argv[0] == "download" and argv[-1].lower().endswith(".pdf"):
                return await self._attachment(argv)
            if len(argv) in (2, 3) and argv[0] == "screenshot" and argv[1].endswith(".png"):
                return await self._screenshot(argv)
        error = self._scope_error(argv)
        if not error and argv and argv[0] == "batch" and stdin is not None:
            try:
                commands = json.loads(stdin)
                if not isinstance(commands, list) or not all(
                    isinstance(command, list) and all(isinstance(arg, str) for arg in command)
                    for command in commands
                ):
                    raise ValueError("Expected an array of command arrays")
                error = next(
                    (issue for command in commands if (issue := self._scope_error(command))), None
                )
            except (TypeError, ValueError):
                error = "batch stdin must be a JSON array of command arrays"
        if error:
            return json.dumps({"error": error})
        metadata = (
            argv[0] in {"skills", "help", "--help", "--version", "version"} or "--help" in argv
        )
        async with self.lock:
            launch = not metadata and not self.used
            if not metadata:
                self.used = True
            result = await self._run(argv, launch=launch, stdin=stdin)
            if argv[0] == "close" and json.loads(result).get("exit_code") == 0:
                self.used = False
            return result

    async def _screenshot(self, argv: list[str]) -> str:
        from examples.deepagents_ov_context.chat.artifacts import valid_artifact_name
        from examples.deepagents_ov_context.chat.office_skills import validate_output

        if (
            self.screenshot_dir is None
            or len(argv) not in {2, 3}
            or (len(argv) == 3 and argv[2] != "--full")
            or not valid_artifact_name(argv[1])
            or Path(argv[1]).suffix != ".png"
        ):
            return json.dumps({"error": "Use screenshot name.png [--full] in an enabled Run"})
        directory = self.screenshot_dir
        path = directory / argv[1]
        async with self.lock:
            if directory.is_symlink() or directory.resolve() != directory:
                return json.dumps({"error": "Screenshot directory is outside this Run"})
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.exists() or path.is_symlink():
                return json.dumps({"error": "Screenshot already exists; choose another filename"})
            launch = not self.used
            self.used = True
            result = json.loads(
                await self._run(["screenshot", str(path), *argv[2:]], launch=launch)
            )
            if result.get("exit_code") != 0:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
                return json.dumps(result)
            try:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > 20 * 1024 * 1024
                ):
                    raise ValueError("Expected PNG screenshot smaller than 20 MiB")
                await asyncio.to_thread(validate_output, path)
            except (OSError, ValueError) as exc:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
                return json.dumps({"exit_code": 1, "error": f"Screenshot failed: {exc}"})
            return json.dumps(
                {
                    "exit_code": 0,
                    "path": f"/output/{path.name}",
                    "bytes": path.stat().st_size,
                    "stdout": f"Saved screenshot to /output/{path.name}",
                    "source": "untrusted browser screenshot",
                },
                ensure_ascii=False,
            )

    async def _attachment(self, argv: list[str]) -> str:
        expected = 3 if argv[0] == "download" else 2
        name = argv[-1]
        if (
            self.download_dir is None
            or len(argv) != expected
            or Path(name).name != name
            or not name.lower().endswith(".pdf")
            or name.startswith((".", "-"))
            or any(char in name for char in ("\\", "\x00"))
            or (argv[0] == "download" and argv[1].startswith("-"))
        ):
            return json.dumps(
                {"error": "Use download URL_OR_SELECTOR name.pdf or pdf_text name.pdf"}
            )
        path = self.download_dir / name
        async with self.lock:
            if path.is_symlink() or path.resolve().parent != self.download_dir:
                return json.dumps({"error": "Attachment path is outside this Run"})
            if argv[0] == "pdf_text":
                if name not in self.downloads:
                    return json.dumps({"error": "Download this PDF in the current Run first"})
                return await asyncio.to_thread(self._pdf_text, path)
            self.download_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.downloads.discard(name)
            launch = not self.used
            self.used = True
            return await self._download(argv[1], path, launch=launch)

    async def _download(self, target: str, path: Path, *, launch: bool) -> str:
        # All network access stays inside the browser CLI. Python only decodes bytes.
        if target.startswith("@"):
            result = json.loads(await self._run(["get", "attr", target, "href"], launch=launch))
            launch = False
            if result.get("exit_code") != 0:
                return json.dumps(result)
            target = result.get("stdout", "").strip()
        expression = r"""(async () => {
            const target = TARGET;
            let href = target;
            if (!/^(https?:|\/|\.\.?\/)/i.test(target)) {
                const element = document.querySelector(target);
                if (!element) throw new Error('Attachment element not found; pass its observed URL');
                href = element.href || element.src;
            }
            let url = new URL(href, location.href);
            if (url.searchParams.has('file')) url = new URL(url.searchParams.get('file'), url);
            if (!['http:', 'https:'].includes(url.protocol)) throw new Error('HTTP(S) PDF required');
            const response = await fetch(url.href, {credentials: 'omit'});
            if (!response.ok) throw new Error('PDF HTTP ' + response.status);
            const reader = response.body.getReader(), chunks = [];
            let length = 0;
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                length += value.length;
                if (length > 20 * 1024 * 1024) {await reader.cancel(); throw new Error('PDF exceeds 20 MiB');}
                chunks.push(value);
            }
            let binary = '';
            for (const chunk of chunks) for (let i = 0; i < chunk.length; i += 8192)
                binary += String.fromCharCode(...chunk.subarray(i, i + 8192));
            return btoa(binary);
        })()""".replace("TARGET", json.dumps(target))
        result = json.loads(await self._run(["--json", "eval", expression], launch=launch))
        if result.get("exit_code") != 0:
            return json.dumps(result)
        try:
            output = json.loads(result["stdout"])
            if not output.get("success"):
                raise ValueError(str(output.get("error", "Browser download failed")))
            content = base64.b64decode(output["data"]["result"], validate=True)
            if len(content) > 20 * 1024 * 1024 or not content.startswith(b"%PDF-"):
                raise ValueError("Expected PDF smaller than 20 MiB")
            path.write_bytes(content)
            self.downloads.add(path.name)
            return json.dumps(
                {
                    "exit_code": 0,
                    "stdout": f"Downloaded {path.name}: {len(content)} bytes",
                    "source": "untrusted browser download",
                }
            )
        except (ValueError, KeyError, TypeError) as exc:
            return json.dumps({"exit_code": 1, "error": f"PDF download failed: {exc}"})

    @staticmethod
    def _pdf_text(path: Path) -> str:
        import pdfplumber

        try:
            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("PDF exceeds 20 MiB")
            with path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise ValueError("Downloaded file is not a PDF")
            with pdfplumber.open(path) as document:
                if len(document.pages) > 50:
                    raise ValueError("PDF exceeds 50 pages")
                pages = [page.extract_text() or "" for page in document.pages]
            text = "\n\n".join(f"[Page {i + 1}]\n{page}" for i, page in enumerate(pages))
            return json.dumps(
                {
                    "exit_code": 0,
                    "stdout": text[:60_000],
                    "pages": len(pages),
                    "truncated": len(text) > 60_000,
                    "source": "untrusted downloaded PDF",
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {"exit_code": 1, "error": f"PDF extraction failed: {type(exc).__name__}: {exc}"}
            )

    async def _run(self, argv: list[str], *, launch: bool = False, stdin: str | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            *self.prefix,
            *(self.launch_args if launch else []),
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            # Loading storage state recreates the context; initialize it once.
            env=self._environment
            if launch
            else {k: v for k, v in self._environment.items() if k != "AGENT_BROWSER_STATE"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin.encode() if stdin is not None else None), self.timeout
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.communicate()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return json.dumps({"exit_code": 124, "error": "Browser command timed out"})
        output = stdout.decode(errors="replace")
        patch = None
        if process.returncode == 0:
            output, patch = correct_browser_guide(argv, output)
        return json.dumps(
            {
                "exit_code": process.returncode,
                "stdout": output,
                "stderr": stderr.decode(errors="replace"),
                "source": "untrusted browser output",
                **({"documentation_patch": patch} if patch else {}),
            },
            ensure_ascii=False,
        )

    async def close(self) -> None:
        if self.used:
            async with self.lock:
                await self._run(["close"])
                self.used = False
