"""The corrected guide must round-trip executable batch commands through Chat."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from examples.deepagents_ov_context.chat.browser_guides import (
    _CORRECT_BATCH_EXAMPLE,
    _OLD_BATCH_EXAMPLE,
    BATCH_GUIDE_PATCH,
    correct_browser_guide,
)
from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser


def test_only_known_guide_example_changes() -> None:
    text = "prefix\n```bash\n" + _OLD_BATCH_EXAMPLE + "\n```\nsuffix"
    fixed, patch = correct_browser_guide(["skills", "get", "core", "--full"], text)
    assert patch == BATCH_GUIDE_PATCH
    assert fixed == text.replace(_OLD_BATCH_EXAMPLE, _CORRECT_BATCH_EXAMPLE)
    assert correct_browser_guide(["snapshot"], text) == (text, None)
    assert correct_browser_guide(["skills", "get", "core", "--full"], fixed) == (fixed, None)
    assert correct_browser_guide(["batch", "--help"], text) == (text, None)


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_guide_patch_and_stdin_transport(tmp_path: Path, exit_code: int) -> None:
    executable = tmp_path / "browser"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys,json,shlex\n"
        "args=sys.argv[4:]\n"
        "if args[:3]==['skills','get','core']:\n"
        f" print({_OLD_BATCH_EXAMPLE!r})\n"
        f" raise SystemExit({exit_code})\n"
        "if args and args[0]=='batch':\n"
        " text=sys.stdin.read()\n"
        " commands=json.loads(text) if text else [shlex.split(a) for a in args[1:] if not a.startswith('--')]\n"
        " print(json.dumps(commands,ensure_ascii=False))\n"
        "else: print('closed')\n"
    )
    executable.chmod(0o700)
    browser = AgentBrowser(executable, "guide-contract")
    guide = json.loads(await browser.command(["skills", "get", "core", "--full"]))
    assert guide["exit_code"] == exit_code
    assert not browser.used
    if exit_code:
        assert "documentation_patch" not in guide
        assert guide["stdout"] == _OLD_BATCH_EXAMPLE + "\n"
    else:
        assert guide["documentation_patch"] == BATCH_GUIDE_PATCH
        assert guide["stdout"] == _CORRECT_BATCH_EXAMPLE + "\n"
        # Execute the guide's actual JSON body through the same StructuredTool
        # stdin transport. Incorrect argument-mode JSON would not decode here.
        payload = guide["stdout"].split("\n", 1)[1].rsplit("\nJSON", 1)[0]
        result = json.loads(
            await browser.tool().ainvoke({"argv": ["batch", "--bail", "--json"], "stdin": payload})
        )
        assert json.loads(result["stdout"]) == json.loads(payload)
        command = json.loads(await browser.command(["batch", "fill @e2 标题"]))
        assert json.loads(command["stdout"]) == [["fill", "@e2", "标题"]]
    await browser.close()
