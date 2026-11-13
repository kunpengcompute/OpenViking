# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One business prompt source for Chat and comparison runs."""

import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

_PROMPTS = Path(__file__).with_name("prompts")
_TOOL_PROMPTS = {"browser": "browser.md", "write_todos": "planning.md"}


def chat_base_prompt() -> str:
    """Load shared execution and office rules without optional capabilities."""
    return "\n\n".join(
        (_PROMPTS / filename).read_text(encoding="utf-8").strip()
        for filename in ("assistant.md", "office.md")
    )


def build_chat_system_prompt(
    *,
    tool_names: Collection[str],
    custom_prompt: str | None = None,
    skill_names: Collection[str] = (),
) -> str:
    """Keep shared rules and append caller-authored session requirements."""
    parts = [chat_base_prompt()]
    if custom_prompt is not None:
        parts.append("[本次会话要求]\n" + custom_prompt)
    parts.extend(
        (_PROMPTS / filename).read_text(encoding="utf-8").strip()
        for name, filename in _TOOL_PROMPTS.items()
        if name in tool_names
    )
    if "pptx" in skill_names:
        parts.append((_PROMPTS / "presentation.md").read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


BROWSER_CONTRACT_VERSION = 2


def browser_contract(execution: dict[str, Any]) -> int:
    """Identify frozen pre-wrapper and current official-CLI session contracts."""
    if "browser_contract_version" in execution:
        version = execution["browser_contract_version"]
        if version not in (1, BROWSER_CONTRACT_VERSION):
            raise ValueError("Unsupported browser contract version")
        return version
    prompt = execution.get("system_prompt", "")
    return 2 if "[Browser 接入说明]" in prompt and "browser_files" in prompt else 1


def current_browser_prompt(stored_prompt: str, *, legacy: bool) -> str:
    """Refresh only application-owned browser guidance, leaving the snapshot intact."""
    guide = (_PROMPTS / "browser.md").read_text(encoding="utf-8").strip()
    pattern = re.compile(r"^\[Browser (?:工具|接入)说明\]\n.*?(?=^\[[^\n]+\]\n|\Z)", re.M | re.S)
    matches = list(pattern.finditer(stored_prompt))
    if matches:
        # Custom session instructions precede the generated browser section.
        match = matches[-1]
        result = stored_prompt[: match.start()] + guide + "\n\n" + stored_prompt[match.end() :]
    else:
        result = stored_prompt + "\n\n" + guide
    if legacy:
        result += (
            "\n\n[Browser 历史会话兼容]\n"
            "本会话历史中的 browser download/pdf_text 和旧截图调用由接入层兼容。"
            "本轮新文件操作使用 browser_files，普通网页操作使用官方 browser CLI。"
        )
    return result.rstrip()
