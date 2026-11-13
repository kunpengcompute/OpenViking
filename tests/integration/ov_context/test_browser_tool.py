"""Check CLI arguments and cancellation against a real local child process."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import pytest

from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser


@pytest.fixture
def cli(tmp_path: Path) -> Path:
    executable = tmp_path / "browser"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "if sys.argv[-1] == 'slow':\n"
        "    Path(__file__ + '.pid').write_text(str(os.getpid()))\n"
        "    time.sleep(10)\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o700)
    return executable


async def test_literal_arguments_and_session_isolation(cli: Path, tmp_path: Path) -> None:
    first = AgentBrowser(cli, "run-a")
    second = AgentBrowser(cli, "run-b")
    expression = f"$(touch {tmp_path / 'must-not-exist'})"
    results = await asyncio.gather(
        first.command(["eval", expression]), second.command(["snapshot", "-i"])
    )
    arguments = [json.loads(json.loads(result)["stdout"]) for result in results]
    assert arguments[0][:2] == ["--session", "run-a"]
    assert arguments[1][:2] == ["--session", "run-b"]
    assert arguments[0][-1] == expression
    assert not (tmp_path / "must-not-exist").exists()
    await first.close()
    await second.close()
    assert not first.used and not second.used


@pytest.mark.parametrize(
    "argv",
    [
        ["--session=other-user", "snapshot"],
        ["open", "file:///etc/passwd"],
        ["snapshot", "--session", "other-user"],
        ["batch", "open https://example.org", "close --all"],
    ],
)
async def test_rejected_commands_do_not_launch(cli: Path, argv: list[str]) -> None:
    browser = AgentBrowser(cli, "test")
    result = json.loads(await browser.command(argv))
    assert "error" in result and not browser.used


async def test_timeout_stops_child(cli: Path) -> None:
    browser = AgentBrowser(cli, "timeout", timeout=2)
    result = json.loads(await browser.command(["get", "text", "slow"]))
    assert result["exit_code"] == 124
    pid = int(Path(str(cli) + ".pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await browser.close()


async def test_cancellation_stops_child(cli: Path) -> None:
    browser = AgentBrowser(cli, "cancel")
    task = asyncio.create_task(browser.command(["get", "text", "slow"]))
    pid_file = Path(str(cli) + ".pid")
    async with asyncio.timeout(5):
        while not pid_file.exists():
            if task.done():
                pytest.fail(f"Child exited before readiness: {task.result()}")
            await asyncio.sleep(0.01)
    assert pid_file.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    await browser.close()


async def test_official_command_transport_and_tool_schema(cli: Path) -> None:
    browser = AgentBrowser(cli, "helpers")
    tool = browser.tool()
    assert set(tool.args) == {"argv", "stdin"}
    result = json.loads(
        await tool.ainvoke({"argv": ["find", "label", "Email", "fill", "user@example.org"]})
    )
    assert json.loads(result["stdout"])[-5:] == [
        "find",
        "label",
        "Email",
        "fill",
        "user@example.org",
    ]
    await browser.close()


async def test_launch_options_are_not_repeated_for_existing_daemon(cli: Path) -> None:
    browser = AgentBrowser(cli, "options", chrome=cli, proxy="http://127.0.0.1:11082")
    initial = json.loads(await browser.command(["open", "https://example.org"]))
    later = json.loads(await browser.command(["snapshot", "-i"]))
    assert "--proxy" in json.loads(initial["stdout"])
    assert "--executable-path" not in json.loads(later["stdout"])
    assert "--proxy" not in json.loads(later["stdout"])
    await browser.command(["close"])
    restarted = json.loads(await browser.command(["open", "https://example.org"]))
    assert "--proxy" in json.loads(restarted["stdout"])
    await browser.close()


async def test_navigation_preserves_official_defaults_and_explicit_wait(cli: Path) -> None:
    browser = AgentBrowser(cli, "dom-ready")
    argv = ["open", "https://example.org"]
    result = json.loads(await browser.command(argv))
    assert json.loads(result["stdout"])[-2:] == argv
    explicit = await browser.command([*argv, "--wait-until", "domcontentloaded"])
    assert json.loads(json.loads(explicit)["stdout"])[-4:] == [
        *argv,
        "--wait-until",
        "domcontentloaded",
    ]
    assert argv == ["open", "https://example.org"]
    result = json.loads(await browser.command(["snapshot", "-i"]))
    assert "--wait-until" not in json.loads(result["stdout"])
    await browser.close()


async def test_context_options_survive_commands_and_restarts(
    cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "print(json.dumps({key: os.environ.get(key) for key in "
        "('AGENT_BROWSER_PROXY', 'AGENT_BROWSER_EXECUTABLE_PATH', 'AGENT_BROWSER_USER_AGENT')}))\n"
    )
    monkeypatch.setenv("AGENT_BROWSER_USER_AGENT", "browser-test-UA")
    monkeypatch.setenv("AGENT_BROWSER_PROXY", "http://ambient-proxy:8080")
    browser = AgentBrowser(cli, "stable-options", chrome=cli, proxy="socks5://127.0.0.1:11082")
    # One Run retains its configuration even if another request changes the environment.
    monkeypatch.setenv("AGENT_BROWSER_USER_AGENT", "later-UA")
    expected = {
        "AGENT_BROWSER_PROXY": "socks5://127.0.0.1:11082",
        "AGENT_BROWSER_EXECUTABLE_PATH": str(cli.resolve()),
        "AGENT_BROWSER_USER_AGENT": "browser-test-UA",
    }
    for argv in (
        ["open", "https://example.org"],
        ["snapshot", "-i"],
        ["eval", "location.href"],
        ["close"],
        ["open", "https://example.org"],
    ):
        result = json.loads(await browser.command(argv))
        assert json.loads(result["stdout"]) == expected
    await browser.close()
    assert os.environ["AGENT_BROWSER_PROXY"] == "http://ambient-proxy:8080"
    assert os.environ["AGENT_BROWSER_USER_AGENT"] == "later-UA"


async def test_browser_instances_keep_separate_proxy_settings(
    cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.write_text(f"#!{sys.executable}\nimport os\nprint(os.environ['AGENT_BROWSER_PROXY'])\n")
    monkeypatch.delenv("AGENT_BROWSER_PROXY", raising=False)
    first = AgentBrowser(cli, "proxy-a", proxy="http://127.0.0.1:11081")
    second = AgentBrowser(cli, "proxy-b", proxy="socks5://127.0.0.1:11082")
    for _ in range(2):
        values = await asyncio.gather(first.command(["get", "url"]), second.command(["get", "url"]))
        assert [json.loads(v)["stdout"].strip() for v in values] == [
            "http://127.0.0.1:11081",
            "socks5://127.0.0.1:11082",
        ]
    await first.close()
    await second.close()
    assert "AGENT_BROWSER_PROXY" not in os.environ


def pdf_bytes() -> bytes:
    content = b"BT /F1 12 Tf 30 150 Td (PDF local extraction proof) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    result = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(result)
    result += b"xref\n0 6\n0000000000 65535 f \n"
    result += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return result + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


async def test_download_and_extract_real_pdf(cli: Path, tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(pdf_bytes())
    payload = json.dumps(
        {"success": True, "data": {"result": base64.b64encode(source.read_bytes()).decode()}}
    )
    cli.write_text(f"#!{sys.executable}\nprint({payload!r})\n")
    browser = AgentBrowser(cli, "pdf", download_dir=tmp_path / "downloads")
    assert "error" in json.loads(await browser.file_command("pdf_text", "notice.pdf"))
    result = json.loads(
        await browser.file_command(
            "download_pdf", "notice.pdf", target="https://example.org/notice.pdf"
        )
    )
    assert result["exit_code"] == 0
    extracted = json.loads(await browser.file_command("pdf_text", "notice.pdf"))
    assert extracted["exit_code"] == 0
    assert "PDF local extraction proof" in extracted["stdout"]
    assert extracted["pages"] == 1 and extracted["truncated"] is False
    other = AgentBrowser(cli, "other", download_dir=tmp_path / "downloads")
    assert "error" in json.loads(await other.file_command("pdf_text", "notice.pdf"))
    await browser.close()


@pytest.mark.parametrize("name", ["../outside.pdf", "/tmp/outside.pdf", "a\\b.pdf", ".hidden.pdf"])
async def test_attachment_paths_cannot_escape(cli: Path, tmp_path: Path, name: str) -> None:
    browser = AgentBrowser(cli, "paths", download_dir=tmp_path / "downloads")
    result = json.loads(await browser.file_command("download_pdf", name, target="@e1"))
    assert "error" in result and not browser.used


async def test_attachment_symlink_is_rejected(cli: Path, tmp_path: Path) -> None:
    directory = tmp_path / "downloads"
    directory.mkdir()
    (directory / "notice.pdf").symlink_to(tmp_path / "outside.pdf")
    browser = AgentBrowser(cli, "symlink", download_dir=directory)
    assert "error" in json.loads(
        await browser.file_command("download_pdf", "notice.pdf", target="@e1")
    )
    assert not browser.used


def test_html_download_is_not_parsed_as_pdf(tmp_path: Path) -> None:
    path = tmp_path / "error.pdf"
    path.write_text("<html>upstream error</html>")
    assert json.loads(AgentBrowser._pdf_text(path))["exit_code"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "error": "PDF HTTP 503"},
        {"success": True, "data": {"result": "not base64"}},
        {"success": True, "data": {"result": base64.b64encode(b"<html>error</html>").decode()}},
    ],
)
async def test_failed_browser_pdf_download_is_not_published(
    cli: Path, tmp_path: Path, payload: dict[str, object]
) -> None:
    cli.write_text(f"#!{sys.executable}\nprint({json.dumps(payload)!r})\n")
    browser = AgentBrowser(cli, "invalid-pdf", download_dir=tmp_path / "downloads")
    result = json.loads(
        await browser.file_command("download_pdf", "a.pdf", target="https://example.org/a.pdf")
    )
    assert result["exit_code"] == 1
    assert not (tmp_path / "downloads" / "a.pdf").exists()
    assert "error" in json.loads(await browser.file_command("pdf_text", "a.pdf"))
    await browser.close()


async def test_storage_state_is_loaded_only_when_starting_browser(
    cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.write_text(
        f"#!{sys.executable}\nimport json, os\n"
        "print(json.dumps({'state': os.environ.get('AGENT_BROWSER_STATE'), "
        "'proxy': os.environ.get('AGENT_BROWSER_PROXY')}))\n"
    )
    monkeypatch.setenv("AGENT_BROWSER_STATE", "synthetic-state.json")
    browser = AgentBrowser(cli, "state-once", proxy="http://localhost:1234")
    for argv, expected_state in (
        (["open", "https://example.org"], "synthetic-state.json"),
        (["snapshot", "-i"], None),
        (["tab", "new", "https://example.org/calendar"], None),
        (["tab", "close", "t2"], None),
        (["close"], None),
        (["open", "https://example.org"], "synthetic-state.json"),
    ):
        result = json.loads(await browser.command(argv))
        assert json.loads(result["stdout"]) == {
            "state": expected_state,
            "proxy": "http://localhost:1234",
        }
    await browser.close()
    assert os.environ["AGENT_BROWSER_STATE"] == "synthetic-state.json"


@pytest.mark.parametrize(
    "argv",
    [
        ["find", "role", "button", "click", "--name", "Save"],
        ["fill", "#title", "--full is literal input"],
        ["wait", "--text", "Saved"],
        ["wait", "--url", "**/dashboard"],
        ["wait", "--fn", "window.ready"],
        ["snapshot", "--json", "-u"],
        ["eval", "-b", "ZG9jdW1lbnQudGl0bGU="],
        ["batch", "find label Title fill Example", "click #save"],
        ["skills", "get", "core", "--full"],
        ["screenshot", "--full", "native.png"],
        ["download", "#attachment", "native.pdf"],
    ],
)
async def test_official_arguments_reach_cli_unchanged(cli: Path, argv: list[str]) -> None:
    browser = AgentBrowser(cli, "official")
    result = json.loads(await browser.command(argv))
    assert json.loads(result["stdout"])[-len(argv) :] == argv
    await browser.close()


async def test_standard_input_and_skill_lookup_do_not_consume_launch_state(
    cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.write_text(
        f"#!{sys.executable}\nimport sys,json,os\n"
        "print(json.dumps({'argv':sys.argv[1:],'stdin':sys.stdin.read(),"
        "'state':os.environ.get('AGENT_BROWSER_STATE')}))\n"
    )
    monkeypatch.setenv("AGENT_BROWSER_STATE", "initial-state.json")
    browser = AgentBrowser(cli, "docs-first")
    docs = json.loads(await browser.command(["skills", "get", "core"]))
    assert not browser.used and json.loads(docs["stdout"])["state"] is None
    code = 'document.querySelector("#title").value = "literal $()"'
    result = json.loads(await browser.command(["eval", "--stdin"], stdin=code))
    data = json.loads(result["stdout"])
    assert data["stdin"] == code and data["state"] == "initial-state.json"
    assert browser.used
    await browser.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["batch", "snapshot -i", "--session=another open https://example.org"],
        ["batch", "close --all"],
        ["connect", "9222"],
        ["goto", "file:///private/data"],
    ],
)
async def test_official_transport_keeps_application_owned_session(
    cli: Path, argv: list[str]
) -> None:
    browser = AgentBrowser(cli, "owned")
    assert "error" in json.loads(await browser.command(argv))
    assert not browser.used


async def test_native_wait_state_and_batch_stdin_preserve_scope(cli: Path) -> None:
    browser = AgentBrowser(cli, "native-state")
    argv = ["wait", "#spinner", "--state", "hidden"]
    result = json.loads(await browser.command(argv))
    assert json.loads(result["stdout"])[-len(argv) :] == argv
    assert "error" in json.loads(await browser.command(["batch"], stdin='[["close", "--all"]]'))
    await browser.close()


async def test_legacy_file_aliases_are_scoped_to_old_sessions(cli: Path, tmp_path: Path) -> None:
    current = AgentBrowser(cli, "modern", download_dir=tmp_path / "new")
    # Current official download retains the CLI's arguments, including PDF names.
    args = ["download", "#attachment", "notice.pdf"]
    result = json.loads(await current.command(args))
    assert json.loads(result["stdout"])[-3:] == args
    await current.close()
    payload = json.dumps(
        {"success": True, "data": {"result": base64.b64encode(pdf_bytes()).decode()}}
    )
    cli.write_text(f"#!{sys.executable}\nprint({payload!r})\n")
    legacy = AgentBrowser(cli, "legacy", download_dir=tmp_path / "old")
    legacy.legacy_file_aliases = True
    downloaded = json.loads(
        await legacy.command(["download", "https://example.org/a.pdf", "notice.pdf"])
    )
    assert downloaded["exit_code"] == 0
    parsed = json.loads(await legacy.command(["pdf_text", "notice.pdf"]))
    assert parsed["exit_code"] == 0 and "PDF local extraction proof" in parsed["stdout"]
    await legacy.close()
