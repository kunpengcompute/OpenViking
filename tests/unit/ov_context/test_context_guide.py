"""Original system instructions survive rule injection and repeated assembly."""

from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from openviking.integrations.langchain.ov_context import OVContext, OVContextConfig
from openviking.integrations.langchain.ov_context.assembler import GUIDE, RULES, ContextAssembler
from openviking.integrations.langchain.ov_context.context_guide import build_context_guide

SystemContent = str | list[str | dict[str, Any]]


@tool
def openviking_tool_result_read() -> str:
    """Read the original tool result."""
    return "original result"


@pytest.mark.parametrize(
    "content",
    [
        f"Original instructions. {GUIDE} Keep this original rule.",
        RULES,
        "Original instructions.\n\n  ",
        "",
        [],
        [{"type": "text", "text": RULES}],
        [
            {"type": "text", "text": f"{GUIDE} Keep this original rule."},
            "Keep this string block too.",
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        ],
    ],
    ids=[
        "literal-marker",
        "literal-rules",
        "whitespace",
        "empty-string",
        "empty-list",
        "rules-block",
        "multimodal",
    ],
)
def test_original_system_and_metadata_survive_repeated_assembly(
    env: tuple[Any, ...], content: SystemContent
) -> None:
    assembler = ContextAssembler(OVContextConfig(context_window=100000), lambda request: 10)
    context = OVContext(session_id="s", messages=env[4].messages, owner=env[3].session_key)
    system = SystemMessage(
        id="original-system",
        name="instructions",
        content=deepcopy(content),
        additional_kwargs={"cache": {"enabled": True}},
        response_metadata={"source": "application"},
    )
    before = deepcopy(system.model_dump())
    first = assembler.prepare(env[4].override(system_message=system), context).request
    assert first.system_message is not None
    expected = (
        content + ("\n\n" if content else "") + RULES
        if isinstance(content, str)
        else [*content, {"type": "text", "text": RULES}]
    )
    assert first.system_message.content == expected
    # The explicit marker remains usable after native serialization and copying.
    restored = SystemMessage.model_validate(first.system_message.model_dump())
    second = assembler.prepare(first.override(system_message=restored), context).request
    assert second.system_message is not None
    assert second.system_message.model_dump() == first.system_message.model_dump()
    assert second.system_message.id == system.id
    assert second.system_message.name == system.name
    assert second.system_message.response_metadata == system.response_metadata
    assert second.system_message.additional_kwargs["cache"] == {"enabled": True}
    assert system.model_dump() == before


@pytest.mark.parametrize("blocks", [False, True], ids=["string", "blocks"])
def test_guide_updates_with_tool_configuration_without_removing_original_rules(
    env: tuple[Any, ...], blocks: bool
) -> None:
    content: SystemContent = [{"type": "text", "text": RULES}] if blocks else RULES
    assembler = ContextAssembler(OVContextConfig(context_window=100000), lambda request: 10)
    context = OVContext(session_id="s", messages=env[4].messages, owner=env[3].session_key)
    first = assembler.prepare(
        env[4].override(system_message=SystemMessage(content=content)), context
    ).request
    assert first.system_message is not None
    with_tool = assembler.prepare(
        first.override(tools=[openviking_tool_result_read]), context
    ).request
    assert with_tool.system_message is not None
    assert str(with_tool.system_message.content).count("openviking_tool_result_read(") == 1
    own_guide = with_tool.system_message.additional_kwargs["ov_context_system_guide"]["text"]
    assert "本次未开放 OV 工具原文回读能力" not in own_guide
    without_tool = assembler.prepare(with_tool.override(tools=[]), context).request
    assert without_tool.system_message is not None
    assert without_tool.system_message.content == first.system_message.content
    assert "openviking_tool_result_read(" not in str(without_tool.system_message.content)


@pytest.mark.parametrize("blocks", [False, True], ids=["string", "blocks"])
def test_assembly_preserves_instructions_appended_after_the_injected_guide(
    env: tuple[Any, ...], blocks: bool
) -> None:
    content: SystemContent = [{"type": "text", "text": "Original"}] if blocks else "Original"
    assembler = ContextAssembler(OVContextConfig(context_window=100000), lambda request: 10)
    context = OVContext(session_id="s", messages=env[4].messages, owner=env[3].session_key)
    first = assembler.prepare(
        env[4].override(system_message=SystemMessage(content=content)), context
    ).request
    assert first.system_message is not None
    previous = deepcopy(first.system_message.model_dump())
    edited: SystemContent = (
        [*first.system_message.content, {"type": "text", "text": "Later instructions"}]
        if isinstance(first.system_message.content, list)
        else first.system_message.content + "\nLater instructions"
    )
    second = assembler.prepare(
        first.override(system_message=first.system_message.model_copy(update={"content": edited})),
        context,
    ).request
    assert second.system_message is not None
    assert "Later instructions" in str(second.system_message.content)
    assert str(second.system_message.content).count(GUIDE) == 1
    assert first.system_message.model_dump() == previous


def test_missing_system_message_gets_one_guide(env: tuple[Any, ...]) -> None:
    assembler = ContextAssembler(OVContextConfig(context_window=100000), lambda request: 10)
    context = OVContext(session_id="s", messages=env[4].messages, owner=env[3].session_key)
    first = assembler.prepare(env[4].override(system_message=None), context).request
    second = assembler.prepare(first, context).request
    assert second.system_message is not None
    assert second.system_message.content == RULES


@pytest.mark.parametrize("mask", range(16))
def test_guide_only_describes_available_backtracking_capabilities(mask: int) -> None:
    names = (
        "openviking_tool_result_read",
        "openviking_tool_result_search",
        "openviking_tool_result_list",
        "openviking_tool_result_search_many",
    )
    available = {name for index, name in enumerate(names) if mask & (1 << index)}
    guide = build_context_guide(available)
    for name in names:
        assert (name + "(" in guide) == (name in available)
    assert guide.count(GUIDE) == 1
    assert guide.endswith("[/OpenViking Context Guide]")
    assert ("本次未开放 OV 工具原文回读能力" in guide) == (not available)
    if names[0] in available:
        assert "非负字符偏移" in guide and "1–20000" in guide
    if names[1] in available:
        assert "命中条数" in guide and "1–20。" in guide
    if names[2] in available:
        assert "1–50。" in guide

    if names[3] in available:
        assert "1–4" in guide and "1–20。" in guide
