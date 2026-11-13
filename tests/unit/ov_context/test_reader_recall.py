import asyncio
from dataclasses import replace
from unittest.mock import call

import pytest

from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext


def memory(name, *, abstract=None, score=0.9, category="preferences", level=2):
    return MatchedContext(
        uri=f"viking://user/u1/memories/{category}/{name}.md",
        context_type=ContextType.MEMORY,
        level=level,
        score=score,
        category=category,
        abstract=name if abstract is None else abstract,
    )


def found(*items):
    return FindResult(memories=list(items), resources=[], skills=[])


async def test_find_mode_preserves_extra_event_content_when_reading_bodies(client):
    event = memory("done", abstract="Summary: 之前 PPT 修改已完成。", score=0.99, category="events")
    method = memory(
        "pptx",
        abstract="Skill: pptx\nBest for: 修改 PPT 文本\nGuidelines:\n- 按 run 定位",
        score=0.6,
        category="skills",
    )
    client.find.return_value = found(event, method)
    client.read.side_effect = lambda uri: (
        event.abstract + "\n2026-11-10 ChatLog:\n[ppt-agent]: 重复历史核验。"
        if uri == event.uri
        else method.abstract + "\n- 核对当前目标值。"
    )
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", prefer_abstract=False)
    ).recall("请修改 PPT 文本并交付文件")
    assert result.status == "success" and result.count == 2
    assert "Skill: pptx" in result.content
    assert "重复历史核验。" in result.content
    assert "ChatLog" not in result.content
    assert client.read.await_count == 2


@pytest.mark.parametrize(
    "config,query",
    [
        (MemoryReaderConfig(search_mode="find", enabled=False), "原始问题"),
        (MemoryReaderConfig(search_mode="find"), " \n "),
        (MemoryReaderConfig(search_mode="find", recall_limit=0), "原始问题"),
        (MemoryReaderConfig(search_mode="find", candidate_limit=0), "原始问题"),
        (MemoryReaderConfig(search_mode="find", max_injected_chars=0), "原始问题"),
        (MemoryReaderConfig(search_mode="find", target_uris=()), "原始问题"),
    ],
)
async def test_disabled_or_empty_recall_does_not_contact_ov(client, config, query):
    result = await OVContextReader(client, config).recall(query)
    assert (result.status, result.content, result.count) == ("empty", "", 0)
    client.find.assert_not_awaited()


async def test_original_query_scope_and_candidate_limit_are_forwarded(client):
    client.find.return_value = found(memory("brief", abstract="用户喜欢简短回答"))
    config = MemoryReaderConfig(
        search_mode="find", target_uris=("viking://user/u1/memories",), candidate_limit=11
    )
    result = await OVContextReader(client, config).recall("  用户喜欢什么回答风格？  ")
    client.find.assert_awaited_once_with(
        "  用户喜欢什么回答风格？  ",
        target_uri=config.target_uris[0],
        limit=11,
        score_threshold=config.score_threshold,
    )
    assert result.status == "success" and result.count == 1
    assert "用户喜欢简短回答" in result.content
    assert "来源：viking://user/u1/memories/preferences/brief.md" in result.content
    client.read.assert_not_awaited()


async def test_session_exchange_does_not_change_recall_scope_or_cache_results(client):
    client.find.return_value = found(memory("brief"))
    config = MemoryReaderConfig(search_mode="find", target_uris=("viking://user/u1/memories",))
    reader = OVContextReader(client, config)
    results = []
    for session_id in ("s1", "s2"):
        await reader.exchange_context(session_id=session_id)
        results.append(await reader.recall("原始问题"))
    assert results[0] == results[1] and results[0].status == "success"
    assert reader.client is client and reader.config is config
    assert (
        client.find.await_args_list
        == [
            call(
                "原始问题",
                target_uri=config.target_uris[0],
                limit=25,
                score_threshold=config.score_threshold,
            )
        ]
        * 2
    )


async def test_uri_dedup_and_semantic_threshold_across_scopes(client):
    low, high = (
        memory("same", abstract="旧摘要", score=0.6),
        memory("same", abstract="新摘要", score=0.9),
    )
    client.find.return_value = found(low, high, memory("excluded", score=0.1))
    cfg = MemoryReaderConfig(
        search_mode="find",
        target_uris=("viking://user/memories", "viking://agent/memories"),
        score_threshold=0.5,
    )
    result = await OVContextReader(client, cfg).recall("用户偏好")
    assert result.count == 1 and "新摘要" in result.content
    assert "旧摘要" not in result.content and "excluded" not in result.content


async def test_find_combines_roots_under_one_candidate_cap_and_injection_cap(client):
    roots = ("viking://user/memories", "viking://user/peers/p/memories")
    client.find.return_value = found(*(memory(str(i), category="events") for i in range(10)))
    config = MemoryReaderConfig(
        search_mode="find",
        target_uris=(*roots, roots[0]),
        candidate_limit=10,
        recall_limit=5,
        ranking="score",
        prefer_leaf=False,
    )
    result = await OVContextReader(client, config).recall("继续处理任务")
    client.find.assert_awaited_once_with(
        "继续处理任务", target_uri=list(roots), limit=10, score_threshold=config.score_threshold
    )
    assert result.status == "success" and result.count == 5


async def test_find_score_policy_has_no_query_boost_or_leaf_preference(client):
    client.find.return_value = found(
        memory("directory", score=0.99, level=0),
        memory("event", score=0.9, category="events"),
        memory("preference", score=0.85),
    )
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", ranking="score", prefer_leaf=False)
    ).recall("用户喜欢什么")
    assert result.count == 3
    assert result.content.index("directory.md") < result.content.index("event.md")
    assert result.content.index("event.md") < result.content.index("preference.md")


async def test_query_aware_preference_ranking_and_unboosted_threshold(client):
    client.find.return_value = found(
        memory("event", score=0.82, category="events"),
        memory("preference", score=0.8),
        memory("low-preference", score=0.49),
    )
    cfg = MemoryReaderConfig(search_mode="find", score_threshold=0.5, recall_limit=1)
    result = await OVContextReader(client, cfg).recall("用户喜欢什么")
    assert result.count == 1 and "preference.md" in result.content
    assert "low-preference" not in result.content


async def test_score_ranking_can_be_selected(client):
    client.find.return_value = found(
        memory("event", score=0.82, category="events"), memory("pref", score=0.8)
    )
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", ranking="score", recall_limit=1)
    ).recall("偏好")
    assert "event.md" in result.content


async def test_events_with_identical_summaries_keep_distinct_uris(client):
    client.find.return_value = found(
        memory("a", abstract="发布完成", category="events"),
        memory("b", abstract="发布完成", category="events"),
    )
    assert (
        await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("什么时候发布")
    ).count == 2


async def test_duplicate_abstracts_are_collapsed(client):
    client.find.return_value = found(
        memory("a", abstract="Short answers"), memory("b", abstract=" Short  answers ", score=0.8)
    )
    assert (
        await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("偏好")
    ).count == 1


async def test_directory_summary_used_only_when_no_leaf_candidates(client):
    directory = memory("directory", level=0)
    client.find.return_value = found(directory)
    result = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("偏好")
    assert result.count == 1
    client.read.assert_not_awaited()
    client.find.return_value = found(directory, memory("leaf"))
    result = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("偏好")
    assert result.count == 1 and "leaf.md" in result.content


@pytest.mark.parametrize("read_result", ["完整正文", "", RuntimeError("body unavailable")])
async def test_leaf_read_and_summary_fallback(client, read_result):
    item = memory("a", abstract="可用摘要")
    client.find.return_value = found(item)
    if isinstance(read_result, Exception):
        client.read.side_effect = read_result
    else:
        client.read.return_value = read_result
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", prefer_abstract=False)
    ).recall("问题")
    client.read.assert_awaited_once_with(item.uri)
    assert (read_result if read_result == "完整正文" else "可用摘要") in result.content


async def test_missing_abstract_reads_leaf_even_in_abstract_mode(client):
    client.find.return_value = found(memory("a", abstract=""))
    client.read.return_value = "正文"
    assert (
        "正文"
        in (
            await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
        ).content
    )


async def test_unreadable_memory_without_summary_is_skipped(client):
    client.find.return_value = found(memory("a", abstract=""))
    client.read.side_effect = RuntimeError("read failed")
    result = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    assert result.status == "error" and not result.content


async def test_character_limit_includes_header_uri_and_skips_whole_oversized_entries(client):
    small = memory("small", abstract="简短摘要", score=0.8)
    client.find.return_value = found(small)
    baseline = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    budget = len(baseline.content)
    client.find.return_value = found(memory("large", abstract="长" * 500), small)
    result = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", max_injected_chars=budget, recall_limit=1)
    ).recall("问题")
    assert result.content == baseline.content and len(result.content) == budget
    assert "large" not in result.content
    too_small = await OVContextReader(
        client, MemoryReaderConfig(search_mode="find", max_injected_chars=1)
    ).recall("问题")
    assert too_small.status == "empty"


async def test_partial_channel_failure_preserves_usable_results(client):
    client.find.side_effect = [
        RuntimeError("ordinary channel unavailable"),
        found(memory("usable", category="experiences")),
    ]
    client.read.return_value = "## Approach\n运行测试"
    cfg = MemoryReaderConfig(search_mode="find", experience_enabled=True)
    result = await OVContextReader(client, cfg).recall("帮我修复 service API 报错并完成发布")
    assert result.status == "success" and result.count == 1


async def test_failed_recall_without_results_reports_error(client):
    client.find.side_effect = RuntimeError("find failed")
    result = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    assert (result.status, result.content, result.count) == ("error", "", 0)


async def test_empty_and_non_memory_hits_do_not_inject(client):
    result = await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    assert result.status == "empty"
    resource = memory("not-a-memory")
    resource.context_type = ContextType.RESOURCE
    client.find.return_value = found(resource)
    assert (
        await OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    ).status == "empty"


async def test_experience_gate_avoids_casual_queries(client):
    cfg = MemoryReaderConfig(search_mode="find", enabled=False, experience_enabled=True)
    assert (
        await OVContextReader(client, cfg).recall("你好，请给我讲一个有趣的笑话")
    ).status == "empty"
    client.find.assert_not_awaited()


@pytest.mark.parametrize("minimum_chars,expected_calls", [(12, 1), (1000, 0)])
async def test_internal_experience_gate_respects_configured_query_length(
    client, minimum_chars, expected_calls
):
    cfg = MemoryReaderConfig(
        search_mode="find",
        enabled=False,
        experience_enabled=True,
        experience_min_query_chars=minimum_chars,
    )
    query = "帮我修复 service API 报错并完成发布"
    result = await OVContextReader(client, cfg).recall(query)
    assert result.status == "empty"
    assert client.find.await_count == expected_calls
    if expected_calls:
        client.find.assert_awaited_once_with(
            query,
            target_uri=cfg.experience_target_uris[0],
            limit=cfg.experience_candidate_limit,
            score_threshold=cfg.experience_score_threshold,
        )


async def test_experience_channel_is_structured_and_not_duplicated(client):
    experience = memory("deploy", category="experiences")
    ordinary = memory("brief", abstract="回答要简短")

    async def find(query, *, target_uri, **kwargs):
        return (
            found(experience)
            if target_uri.endswith("/experiences")
            else found(ordinary, experience)
        )

    client.find.side_effect = find
    client.read.return_value = (
        "## Situation\n服务发布\n## Approach\n先执行测试\n## Reflect\n避免跳过检查"
    )
    cfg = MemoryReaderConfig(search_mode="find", experience_enabled=True)
    result = await OVContextReader(client, cfg).recall("帮我修复 service API 报错并完成发布")
    assert result.count == 2 and result.content.count(experience.uri) == 1
    assert "任务步骤：\n先执行测试" in result.content
    assert "注意事项：\n避免跳过检查" in result.content
    assert result.content.index("[任务经验]") < result.content.index("[长期记忆]")
    assert len(result.content) <= cfg.max_injected_chars
    client.read.assert_awaited_once_with(experience.uri)
    assert (
        call(
            "帮我修复 service API 报错并完成发布",
            target_uri=cfg.experience_target_uris[0],
            limit=12,
            score_threshold=0.35,
        )
        in client.find.await_args_list
    )


async def test_experience_and_ordinary_recall_share_final_character_budget(client):
    client.find.return_value = found(memory("deploy", category="experiences"), memory("ordinary"))
    client.read.return_value = "## Approach\n运行测试"
    cfg = MemoryReaderConfig(search_mode="find", experience_enabled=True)
    reader = OVContextReader(client, cfg)
    complete = await reader.recall("帮我修复 service API 报错并完成发布")
    exp_section = complete.content.split("\n\n[长期记忆]")[0]
    limited = await OVContextReader(
        client, replace(cfg, max_injected_chars=len(exp_section))
    ).recall("帮我修复 service API 报错并完成发布")
    assert limited.count == 1 and limited.content == exp_section


async def test_experience_threshold_filters_server_results_too(client):
    client.find.return_value = found(memory("low", category="experiences", score=0.1))
    cfg = MemoryReaderConfig(search_mode="find", enabled=False, experience_enabled=True)
    assert (
        await OVContextReader(client, cfg).recall("帮我修复 service API 报错并完成发布")
    ).status == "empty"
    client.read.assert_not_awaited()


async def test_total_timeout_covers_all_body_reads_and_discards_partial_content(client):
    client.find.return_value = found(memory("a"), memory("b"))

    async def read(uri):
        await asyncio.sleep(0.06)
        return "完整正文"

    client.read.side_effect = read
    cfg = MemoryReaderConfig(search_mode="find", prefer_abstract=False, recall_timeout=0.1)
    result = await OVContextReader(client, cfg).recall("问题")
    assert (result.status, result.content, result.count) == ("timeout", "", 0)


async def test_late_result_from_cancellation_resistant_client_is_discarded(client):
    released = asyncio.Event()

    async def find(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await released.wait()
            return found(memory("too-late"))

    client.find.side_effect = find
    reader = OVContextReader(client, MemoryReaderConfig(search_mode="find", recall_timeout=0.02))
    result = await asyncio.wait_for(reader.recall("问题"), timeout=0.5)
    assert result.status == "timeout" and not result.content
    released.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert result.count == 0


async def test_caller_cancellation_is_propagated(client):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def find(*args, **kwargs):
        entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    client.find.side_effect = find
    task = asyncio.create_task(
        OVContextReader(client, MemoryReaderConfig(search_mode="find")).recall("问题")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), timeout=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recall_timeout": 0},
        {"context_timeout": float("inf")},
        {"score_threshold": float("nan")},
        {"experience_score_threshold": 1.1},
        {"max_injected_chars": -1},
        {"candidate_limit": True},
        {"ranking": "unknown"},
        {"target_uris": "viking://user/memories"},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MemoryReaderConfig(search_mode="find", **kwargs)
