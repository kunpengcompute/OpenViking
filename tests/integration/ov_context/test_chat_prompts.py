"""Business instructions are shared across modes and independent of OV rules."""

from types import SimpleNamespace
from typing import Any

import pytest

from examples.deepagents_ov_context.chat.prompts import (
    build_chat_system_prompt,
    chat_base_prompt,
)
from examples.deepagents_ov_context.chat.server import ActiveRun


def test_batch_base_matches_chat_without_unavailable_tool_instructions() -> None:
    prompt = build_chat_system_prompt(tool_names={"debug_echo"})
    assert prompt == chat_base_prompt()
    assert "本地 OVContext 调试助手" not in prompt
    assert "browser" not in prompt and "create_presentation" not in prompt
    assert "OpenViking Context Guide" not in prompt


def test_tool_guidance_is_independent_and_follows_custom_business_prompt() -> None:
    custom = "你是资料整理助手，使用英文作答。"
    assert build_chat_system_prompt(tool_names=(), custom_prompt=custom) == (
        chat_base_prompt() + "\n\n[本次会话要求]\n" + custom
    )
    browser = build_chat_system_prompt(tool_names={"browser"}, custom_prompt=custom)
    assert browser.startswith(chat_base_prompt() + "\n\n[本次会话要求]\n" + custom)
    assert "pdf_text" in browser and "browser_files" in browser
    assert "create_presentation" not in browser
    presentation = build_chat_system_prompt(tool_names=(), skill_names={"pptx"})
    assert "pptx Skill" in presentation and "create_presentation" not in presentation
    both = build_chat_system_prompt(tool_names={"browser"}, skill_names={"pptx"})
    assert both.count("[Browser 接入说明]") == both.count("[PPT Skill 说明]") == 1


@pytest.mark.parametrize("mode", ["current", "baseline", "context_only", "ordinary", "experience"])
@pytest.mark.parametrize("custom_prompt", [None, "你是资料整理助手，使用英文作答。"])
async def test_graph_modes_pass_the_same_authored_business_prompt(
    chat: Any, monkeypatch: Any, mode: str, custom_prompt: str | None
) -> None:
    client, runtime = chat
    captured: dict[str, Any] = {}

    def create_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.create_deep_agent", create_graph
    )
    created = await client.post(
        "/api/sessions", json={"execution": {"mode": mode, "system_prompt": custom_prompt}}
    )
    assert created.status_code == 201
    session = runtime.store.get(created.json()["session_id"])
    sdk = runtime.sdk()
    middleware = None
    try:
        _, middleware = runtime.session_graph(session, sdk, ActiveRun())
        assert captured["system_prompt"] == build_chat_system_prompt(
            tool_names=(), custom_prompt=custom_prompt
        )
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()


async def test_browser_cli_and_chat_file_delivery_are_separate_graph_tools(
    chat, monkeypatch
) -> None:
    from pathlib import Path

    from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser

    client, runtime = chat
    captured = {}
    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    response = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    session = runtime.store.get(response.json()["session_id"])
    sdk = runtime.sdk()
    try:
        runtime.session_graph(
            session, sdk, ActiveRun(), browser=AgentBrowser(Path("/bin/true"), "tools")
        )
        tools = {tool.name: tool for tool in captured["tools"]}
        assert tools["browser"].description == "Browser automation CLI for AI agents."
        assert set(tools["browser"].args) == {"argv", "stdin"}
        assert "action" in tools["browser_files"].args
    finally:
        await sdk.close()


def test_old_browser_guidance_refresh_preserves_business_and_custom_requirements() -> None:
    from examples.deepagents_ov_context.chat.prompts import browser_contract, current_browser_prompt

    stored = (
        "[执行规则]\n固定业务规则。\n\n[本次会话要求]\n只查本工作区，禁止发送邮件。\n\n"
        "[Browser 工具说明]\neval 只用于读取页面。browser download URL name.pdf。\n\n"
        "[任务计划]\n保留原计划规则。"
    )
    assert browser_contract({"system_prompt": stored}) == 1
    current = current_browser_prompt(stored, legacy=True)
    assert "固定业务规则。" in current and "禁止发送邮件。" in current
    assert "[任务计划]\n保留原计划规则。" in current
    assert "eval 只用于读取页面" not in current
    assert current.count("[Browser 接入说明]") == 1
    assert "[Browser 历史会话兼容]" in current
    assert browser_contract({"system_prompt": current}) == 2
    assert current_browser_prompt(
        current_browser_prompt(stored, legacy=False), legacy=False
    ) == current_browser_prompt(stored, legacy=False)


async def test_legacy_graph_compatibility_keeps_persisted_snapshot_frozen(
    chat, monkeypatch
) -> None:
    from pathlib import Path

    from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser

    client, runtime = chat
    sid = (await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})).json()[
        "session_id"
    ]
    session = runtime.store.get(sid)
    payload = session.model_dump(mode="json")
    payload["config_snapshot"]["execution"].pop("browser_contract_version", None)
    payload["config_snapshot"]["execution"]["system_prompt"] += (
        "\n\n[Browser 工具说明]\neval 只用于读取页面。"
    )
    with runtime.store.db:
        runtime.store.db.execute(
            "UPDATE sessions SET data=? WHERE id=?", (__import__("json").dumps(payload), sid)
        )
    session = runtime.store.get(sid)
    original = session.model_dump_json()
    captured = {}
    monkeypatch.setattr(
        "examples.deepagents_ov_context.chat.server.create_deep_agent",
        lambda **kw: captured.update(kw) or SimpleNamespace(),
    )
    browser = AgentBrowser(Path("/bin/true"), "legacy-probe")
    sdk = runtime.sdk()
    try:
        runtime.session_graph(session, sdk, ActiveRun(), browser=browser)
        assert browser.legacy_file_aliases
        assert "[Browser 接入说明]" in captured["system_prompt"]
        assert "eval 只用于读取页面" not in captured["system_prompt"]
        assert runtime.store.get(sid).model_dump_json() == original
        runtime.store.save(session)
    finally:
        await sdk.close()
