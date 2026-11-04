"""Cache-contract examples using a UT substitute, not the production OVContext implementation."""

from copy import copy

import pytest
from fake_ov_context import FakeOVContext
from langchain_core.messages import AIMessage, HumanMessage

from openviking.integrations.ov_memoryRetrieve import (
    ContextReadError,
    MemoryReaderConfig,
    OVContextReader,
)
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext


@pytest.mark.parametrize("status", ["success", "empty", "timeout", "error"])
def test_all_outcomes_are_attempted_once_and_reset_at_run_end(status):
    user = HumanMessage(id="u1", content="原始问题")
    cache = FakeOVContext("s1")
    cache.begin_run("run1", user)
    cache.append([user])
    cache.inject_recall("参考内容" if status == "success" else "", status)
    first_view = cache.messages[0]
    cache.begin_run("run1", user)
    cache.inject_recall("重复召回", "success")
    assert cache.recall_done and cache.recall_status == status
    assert cache.messages[0] is first_view
    assert user.content == "原始问题"
    cache.end_run()
    assert cache.messages[0] is user
    assert not cache.recall_done and cache.run_id is None


def test_multimodal_injection_preserves_original_blocks_and_metadata():
    user = HumanMessage(
        id="u1",
        content=[
            {"type": "text", "text": "原始问题"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ],
        additional_kwargs={"source": "app"},
    )
    cache = FakeOVContext("s1")
    cache.begin_run("run1", user)
    cache.append([user])
    cache.inject_recall("用户偏好", "success")
    view = cache.messages[0]
    assert view.id == user.id and view.additional_kwargs == user.additional_kwargs
    assert view.content[1:] == user.content
    view.content[2]["image_url"]["url"] = "changed"
    assert user.content[1]["image_url"]["url"] != "changed"


def test_refresh_preserves_injected_view_and_failed_refresh_preserves_entire_cache():
    user = HumanMessage(id="u1", content="原始问题")
    cache = FakeOVContext("s1")
    cache.begin_run("run1", user)
    cache.append([user, AIMessage(content="近期回复")])
    cache.inject_recall("参考记忆", "success")
    view = cache.messages[0]
    summary = HumanMessage(id="ov_summary:archive_2", content="概要")
    cache.replace([summary, user.model_copy()], "archive_2")
    assert cache.messages[1] is view
    assert cache.recall_done and cache.archive_id == "archive_2"
    previous = cache.messages
    with pytest.raises(ContextReadError):
        cache.replace([summary], "archive_3")
    assert cache.messages is previous and cache.archive_id == "archive_2"
    with pytest.raises(ContextReadError):
        cache.replace([HumanMessage(id="u1", content="错误用户内容")], "archive_3")
    assert cache.messages is previous
    cache.end_run()
    assert cache.messages[1] is user


def test_next_begin_run_cleans_up_interrupted_previous_run():
    cache = FakeOVContext("s1")
    first = HumanMessage(id="u1", content="第一轮")
    cache.begin_run("run1", first)
    cache.append([first])
    cache.inject_recall("旧召回", "success")
    second = HumanMessage(id="u2", content="第二轮")
    cache.begin_run("run2", second)
    cache.append([second])
    assert cache.messages[0] is first and not cache.recall_done
    cache.inject_recall("新召回", "success")
    assert "新召回" in cache.messages[1].content and first.content == "第一轮"


async def test_reader_and_cache_recall_once_per_run_across_model_loops(client):
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
    cache = FakeOVContext("s1")
    for run in range(2):
        original = HumanMessage(id=f"u{run}", content=f"原始问题{run}")
        cache.begin_run(f"run{run}", original)
        cache.append([original])
        for _ in range(3):
            if not cache.recall_done:
                recall = await reader.recall(original.content)
                cache.inject_recall(recall.content, recall.status)
        assert client.find.await_count == run + 1
        assert original.content == f"原始问题{run}"
        cache.end_run()


async def test_reader_snapshot_can_be_passed_to_the_documented_cache_replace_contract(client):
    original = HumanMessage(id="native_u", content="原始问题")
    cache = FakeOVContext("s1")
    cache.begin_run("run1", original)
    cache.append([original])
    client.get_session_context.return_value["messages"] = [
        {"id": original.id, "role": "user", "parts": [{"type": "text", "text": original.content}]}
    ]
    reader = OVContextReader(client)
    snapshot = await reader.exchange_context(session_id=cache.session_id)
    cache.replace(snapshot.messages, snapshot.archive_id)
    assert cache.archive_id is None
    assert cache.messages[0].id == original.id
    assert cache.messages[0].content == original.content


@pytest.mark.parametrize("recall_status", ["success", "empty", "timeout", "error"])
async def test_failed_boundary_exchange_preserves_write_run_and_recall_state(client, recall_status):
    original = HumanMessage(id="native_u", content="本 Run 原始问题")
    cache = FakeOVContext("s1")
    cache.begin_run("active_run", original)
    cache.append([original])
    cache.archive_id = "archive_001"
    # Test-only sentinels; implementing cursor/count logic belongs to other modules.
    cache.write_index = 12
    cache.completed_runs = 5
    cache.inject_recall("本轮参考记忆" if recall_status == "success" else "", recall_status)
    injected_user = cache.messages[0]
    reader = OVContextReader(client, MemoryReaderConfig(search_mode="find"))
    client.get_session_context.return_value.update(
        {
            "latest_finished_archive_id": "archive_002",
            "latest_archive_overview": "archive_001 的已有概要",
            "messages": [
                {
                    "id": "failed_raw",
                    "role": "user",
                    "parts": [{"type": "text", "text": "失败归档原文"}],
                },
                {
                    "id": original.id,
                    "role": "user",
                    "parts": [{"type": "text", "text": original.content}],
                },
            ],
        }
    )
    snapshot = await reader.exchange_context(session_id=cache.session_id)
    # Reader returns the snapshot without applying it to the caller's cache.
    assert cache.archive_id == "archive_001" and cache.messages[0] is injected_user
    cache.replace(snapshot.messages, snapshot.archive_id)
    assert cache.archive_id == "archive_002"
    assert cache.messages[-1] is injected_user
    assert cache.messages[1].content == "失败归档原文"
    assert cache.write_index == 12 and cache.completed_runs == 5
    assert cache.run_id == "active_run" and cache.original_user is original
    assert cache.recall_done and cache.recall_status == recall_status
    client.find.assert_not_awaited()


async def test_invalid_exchange_does_not_publish_a_replacement_for_the_old_cache(client):
    cache = FakeOVContext("s1")
    original = HumanMessage(id="native_u", content="本 Run 原始问题")
    cache.begin_run("active_run", original)
    cache.append([original])
    cache.archive_id = "archive_001"
    cache.inject_recall("本轮参考记忆", "success")
    previous = cache.messages
    client.get_session_context.return_value.pop("latest_finished_archive_id")
    with pytest.raises(ContextReadError):
        await OVContextReader(client).exchange_context(session_id="s1")
    assert cache.messages is previous and cache.archive_id == "archive_001"
    assert cache.run_id == "active_run" and cache.recall_done


@pytest.mark.parametrize(
    "user_records",
    [
        [],
        [("native_u", "错误用户内容")],
        [("native_u", "本 Run 原始问题"), ("native_u", "本 Run 原始问题")],
    ],
)
async def test_caller_rejects_invalid_current_user_without_replacing_old_cache(
    client, user_records
):
    original = HumanMessage(id="native_u", content="本 Run 原始问题")
    cache = FakeOVContext("s1")
    cache.begin_run("active_run", original)
    cache.append([original])
    cache.archive_id = "archive_001"
    cache.inject_recall("本轮参考记忆", "success")
    previous = cache.messages
    injected_user = cache.messages[0]
    client.get_session_context.return_value.update(
        {
            "latest_finished_archive_id": "archive_002",
            "messages": [
                {"id": mid, "role": "user", "parts": [{"type": "text", "text": content}]}
                for mid, content in user_records
            ],
        }
    )
    # The reader validates the response, without knowing this Run's original user.
    snapshot = await OVContextReader(client).exchange_context(session_id=cache.session_id)
    assert snapshot.archive_id == "archive_002"
    updated = copy(cache)
    updated.messages = list(cache.messages)
    with pytest.raises(ContextReadError):
        updated.replace(snapshot.messages, snapshot.archive_id)
    assert cache.messages is previous and cache.messages[0] is injected_user
    assert cache.archive_id == "archive_001"
    assert cache.run_id == "active_run" and cache.recall_status == "success"
