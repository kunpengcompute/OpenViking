"""Exercise the production OVContext contract and its reader output boundary."""

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context.assembler import INJECTED
from openviking.integrations.langchain.ov_context.state import OVContext
from openviking.integrations.ov_memoryRetrieve import (
    ContextReadError,
    MemoryReaderConfig,
    OVContextReader,
)
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext


@pytest.fixture
def context():
    user = HumanMessage(
        id="u1", content="原始问题", name="user", additional_kwargs={"source": "app"}
    )
    cache = OVContext(
        "s1", messages=[user], archive_id="archive_1", write_index=12, completed_runs=5
    )
    cache.begin_run("run1", user)
    return cache


def assert_unchanged(context, previous, state):
    assert context.messages is previous
    assert context.to_state() == state


@pytest.mark.parametrize("status", ["success", "empty", "timeout", "error"])
def test_recall_status_controls_injection_and_attempt_is_reused(context, status):
    user = context.original_user
    # Non-success results must not inject even if a caller supplies a body.
    context.inject_recall("参考内容", status)
    view = context.messages[0]
    assert context.recall_done and context.recall_status == status
    if status == "success":
        assert view.content == "[参考记忆]\n参考内容\n\n[用户原文]\n原始问题"
        assert view.id == user.id and view.name == user.name
        assert view.additional_kwargs == {"source": "app", INJECTED: True}
        assert view is not user
    else:
        assert view is user
    assert user.content == "原始问题" and user.additional_kwargs == {"source": "app"}
    context.begin_run("run1", user)
    context.inject_recall("第二次召回", "success")
    assert context.messages[0] is view and context.recall_status == status
    context.end_run()
    assert context.messages[0] is user
    assert context.run_id is None and context.original_user is None
    assert not context.recall_done and context.recall_status is None
    assert context.archive_id == "archive_1"
    assert context.write_index == 12 and context.completed_runs == 5


@pytest.mark.parametrize("status", ["success", "empty", "timeout", "error"])
def test_empty_recall_only_records_status(context, status):
    user = context.original_user
    context.inject_recall("", status)
    assert context.messages[0] is user
    assert context.recall_done and context.recall_status == status


@pytest.mark.parametrize("missing", ["run_id", "original_user"])
def test_injection_requires_active_run(context, missing):
    setattr(context, missing, None)
    previous, state = context.messages, deepcopy(context.to_state())
    with pytest.raises(RuntimeError, match=r"begin_run\(\) must precede recall injection"):
        context.inject_recall("参考内容", "success")
    assert_unchanged(context, previous, state)


def test_invalid_recall_status_does_not_mutate(context):
    previous, state = context.messages, deepcopy(context.to_state())
    with pytest.raises(ValueError, match="Invalid recall status"):
        context.inject_recall("参考内容", "invalid")
    assert_unchanged(context, previous, state)


@pytest.mark.parametrize("user_id", [None, ""])
def test_successful_injection_requires_native_id(user_id):
    user = HumanMessage(id=user_id, content="原始问题")
    cache = OVContext("s1", messages=[user, AIMessage(content="其他消息")])
    cache.begin_run("run1", user)
    previous, state = cache.messages, deepcopy(cache.to_state())
    with pytest.raises(ContextReadError, match="non-empty native ID"):
        cache.inject_recall("参考内容", "success")
    assert_unchanged(cache, previous, state)


@pytest.mark.parametrize("invalid_cache", ["missing", "duplicate", "assistant", "tool"])
def test_successful_injection_rejects_invalid_user_matches(context, invalid_cache):
    user = context.original_user
    context.messages = invalid_user_messages(invalid_cache, user)
    previous, state = context.messages, deepcopy(context.to_state())
    with pytest.raises(ContextReadError, match="exactly one cache match"):
        context.inject_recall("参考内容", "success")
    assert_unchanged(context, previous, state)


def invalid_user_messages(kind, user):
    if kind == "missing":
        return [HumanMessage(id="other", content=user.content)]
    if kind == "duplicate":
        return [user.model_copy(), user.model_copy()]
    if kind == "assistant":
        return [AIMessage(id=user.id, content=user.content)]
    if kind == "tool":
        return [ToolMessage(id=user.id, content=user.content, tool_call_id="call1")]
    return [HumanMessage(id=user.id, content="错误正文")]


def test_multimodal_injection_isolated_from_original_user():
    user = HumanMessage(
        id="u1",
        content=[
            {"type": "text", "text": "图片问题"},
            {"type": "image_url", "image_url": {"url": "original"}},
        ],
        additional_kwargs={"source": "app"},
    )
    cache = OVContext("s1", messages=[user])
    cache.begin_run("run1", user)
    cache.inject_recall("参考内容", "success")
    view = cache.messages[0]
    assert view.content[1:] == user.content
    view.content[1]["text"] = "changed"
    view.content[2]["image_url"]["url"] = "changed"
    assert user.content[0]["text"] == "图片问题"
    assert user.content[1]["image_url"]["url"] == "original"


@pytest.mark.parametrize("status", [None, "success", "empty", "timeout", "error"])
def test_replace_preserves_user_view_and_all_other_state(context, status):
    if status is not None:
        context.inject_recall("参考内容" if status == "success" else "", status)
    view = context.messages[0]
    state = context.to_state()
    snapshot_user = HumanMessage(id="u1", content=[{"type": "text", "text": "原始问题"}])
    summary = HumanMessage(id="ov_summary:archive_2", content="概要")
    messages = (summary, snapshot_user)
    context.replace(messages, "archive_2")
    assert context.messages == [summary, view]
    assert context.messages[-1] is view
    assert messages[-1] is snapshot_user
    assert context.archive_id == "archive_2"
    for field, value in state.items():
        if field not in {"messages", "archive_id"}:
            assert getattr(context, field) == value


@pytest.mark.parametrize("status", [None, "success", "empty", "timeout", "error"])
@pytest.mark.parametrize("invalid_snapshot", ["missing", "duplicate", "assistant", "tool", "body"])
def test_replace_rejects_invalid_snapshot_without_mutation(context, status, invalid_snapshot):
    if status is not None:
        context.inject_recall("参考内容" if status == "success" else "", status)
    previous, state = context.messages, deepcopy(context.to_state())
    messages = invalid_user_messages(invalid_snapshot, context.original_user)
    if invalid_snapshot == "missing":
        # Changed IDs with an identical unique body are valid after refresh.
        messages = [HumanMessage(id="other", content="另一个问题")]
    snapshot_before = deepcopy(messages)
    with pytest.raises(ContextReadError):
        context.replace(messages, "archive_2")
    assert_unchanged(context, previous, state)
    assert messages == snapshot_before


@pytest.mark.parametrize("invalid_cache", ["missing", "duplicate", "assistant", "tool"])
def test_replace_rejects_invalid_old_cache(context, invalid_cache):
    context.inject_recall("参考内容", "success")
    context.messages = invalid_user_messages(invalid_cache, context.original_user)
    previous, state = context.messages, deepcopy(context.to_state())
    with pytest.raises(ContextReadError, match="exactly one cache match"):
        context.replace([context.original_user.model_copy()], "archive_2")
    assert_unchanged(context, previous, state)


def test_initial_snapshot_can_load_without_active_run():
    cache = OVContext("s1", write_index=12, completed_runs=5)
    messages = [HumanMessage(id="history", content="历史消息")]
    cache.replace(messages, None)
    assert cache.messages == messages and cache.messages is not messages
    assert cache.write_index == 12 and cache.completed_runs == 5
    assert cache.archive_id is None


def test_replace_normalizes_text_blocks_and_checks_image_content():
    image = {"type": "image_url", "image_url": {"url": "original"}}
    user = HumanMessage(id="u1", content=[{"type": "text", "text": "图片问题"}, image])
    cache = OVContext("s1", messages=[user])
    cache.begin_run("run1", user)
    restored = HumanMessage(
        id="u1",
        content=[
            {"type": "text", "text": "图片"},
            {"type": "text", "text": "问题"},
            deepcopy(image),
        ],
    )
    cache.replace([restored], "archive_2")
    assert cache.messages[0] is user
    restored.content[-1]["image_url"]["url"] = "other"
    previous, state = cache.messages, deepcopy(cache.to_state())
    with pytest.raises(ContextReadError, match="does not match the original input"):
        cache.replace([restored], "archive_3")
    assert_unchanged(cache, previous, state)


async def test_real_reader_outputs_connect_to_production_context(context, client):
    client.find.return_value = FindResult(
        memories=[
            MatchedContext(
                "viking://user/memories/preferences/brief.md",
                ContextType.MEMORY,
                abstract="简短回答",
                score=0.9,
            )
        ],
        resources=[],
        skills=[],
    )
    reader = OVContextReader(client, MemoryReaderConfig(search_mode="find"))
    user = context.original_user
    for _ in range(2):
        if not context.recall_done:
            result = await reader.recall(user.content)
            context.inject_recall(result.content, result.status)
    assert context.recall_status == "success" and "简短回答" in context.messages[0].content
    client.find.assert_awaited_once()
    view = context.messages[0]
    client.get_session_context.return_value.update(
        latest_finished_archive_id="archive_2",
        latest_archive_overview="概要",
        messages=[
            {"id": user.id, "role": "user", "parts": [{"type": "text", "text": user.content}]}
        ],
    )
    snapshot = await reader.exchange_context(context.session_id, expected_archive_id="archive_2")
    context.replace(snapshot.messages, snapshot.archive_id)
    assert context.messages[0].id == "ov_summary:archive_2"
    assert context.messages[-1] is view and context.archive_id == "archive_2"
    assert context.write_index == 12 and context.completed_runs == 5

    previous, state = context.messages, deepcopy(context.to_state())
    client.get_session_context.return_value["messages"][0]["parts"][0]["text"] = "错误正文"
    snapshot = await reader.exchange_context(context.session_id, expected_archive_id="archive_2")
    with pytest.raises(ContextReadError, match="does not match the original input"):
        context.replace(snapshot.messages, snapshot.archive_id)
    assert_unchanged(context, previous, state)
