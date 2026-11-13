"""Screenshot routing, genuine PNG publication and per-Run path isolation."""

import base64
import io
import json
import sys

import pytest
from PIL import Image

from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser
from examples.deepagents_ov_context.chat.office_skills import BUNDLED_SKILLS, OfficeSkillRun


@pytest.fixture
def screenshot_cli(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode()
    path = tmp_path / "browser"
    path.write_text(
        f"#!{sys.executable}\nimport sys,json,base64\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if 'screenshot' in args:\n"
        " target=Path(args[args.index('screenshot')+1])\n"
        f" target.write_bytes(base64.b64decode({payload!r}))\n"
        "print(json.dumps(args))\n"
    )
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("full", [False, True])
async def test_screenshot_saves_verified_png_and_preserves_session(screenshot_cli, tmp_path, full):
    browser = AgentBrowser(screenshot_cli, "run-a", screenshot_dir=tmp_path / "a/output")
    await browser.command(["open", "https://example.test"])
    argv = ["screenshot", "verification.png", *(["--full"] if full else [])]
    result = json.loads(await browser.file_command("screenshot", "verification.png", full=full))
    assert result["exit_code"] == 0 and result["path"] == "/output/verification.png"
    with Image.open(tmp_path / "a/output/verification.png") as image:
        assert image.size == (100, 80)
    assert browser.used and argv[1] == "verification.png"
    assert "error" in json.loads(
        await browser.file_command("screenshot", "verification.png", full=full)
    )
    second = AgentBrowser(screenshot_cli, "run-b", screenshot_dir=tmp_path / "b/output")
    assert (
        json.loads(await second.file_command("screenshot", "verification.png", full=full))[
            "exit_code"
        ]
        == 0
    )
    await browser.close()
    await second.close()


@pytest.mark.parametrize(
    "name", ["../escape.png", "/tmp/escape.png", "x.jpg", "--session", "x\\y.png"]
)
async def test_screenshot_rejects_paths_before_starting(screenshot_cli, tmp_path, name):
    browser = AgentBrowser(screenshot_cli, "a", screenshot_dir=tmp_path / "output")
    assert "error" in json.loads(await browser.file_command("screenshot", name))
    assert not browser.used and not (tmp_path / "output").exists()


async def test_screenshot_rejects_options_disabled_output_and_symlinks(screenshot_cli, tmp_path):
    disabled = AgentBrowser(screenshot_cli, "a")
    assert "error" in json.loads(await disabled.file_command("screenshot", "x.png"))
    browser = AgentBrowser(screenshot_cli, "b", screenshot_dir=tmp_path / "output")
    assert "error" in json.loads(
        await browser.file_tool().ainvoke({"action": "screenshot", "filename": "--session"})
    )
    directory = tmp_path / "output"
    directory.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"original")
    (directory / "x.png").symlink_to(outside)
    assert "error" in json.loads(await browser.file_command("screenshot", "x.png"))
    assert outside.read_bytes() == b"original" and not browser.used
    directory.rename(tmp_path / "saved")
    directory.symlink_to(tmp_path / "saved", target_is_directory=True)
    assert "error" in json.loads(await browser.file_command("screenshot", "other.png"))
    assert not browser.used


async def test_invalid_screenshot_is_not_delivered(screenshot_cli, tmp_path):
    screenshot_cli.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if 'screenshot' in args:Path(args[args.index('screenshot')+1]).write_bytes(b'not PNG')\n"
    )
    browser = AgentBrowser(screenshot_cli, "a", screenshot_dir=tmp_path / "output")
    result = json.loads(await browser.file_command("screenshot", "x.png"))
    assert result["exit_code"] == 1 and not (tmp_path / "output/x.png").exists()
    await browser.close()


@pytest.mark.parametrize("mode", ["baseline", "current"])
async def test_chat_publishes_png_download_in_both_modes(chat, mode):
    client, runtime = chat
    session = (
        await client.post("/api/sessions", json={"user_id": "u", "execution": {"mode": mode}})
    ).json()
    sid = session["session_id"]
    run = OfficeSkillRun(
        {"agent-browser": BUNDLED_SKILLS / "agent-browser"},
        None,
        runtime.settings.data_dir / "artifacts",
        sid,
        "image-run",
    )
    Image.new("RGB", (160, 90), "white").save(run.folder / "output/verification.png")
    assert run.publish()[0].kind == "image"
    items = (await client.get(f"/api/sessions/{sid}/artifacts")).json()
    response = await client.get(items[0]["url"])
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (160, 90)
