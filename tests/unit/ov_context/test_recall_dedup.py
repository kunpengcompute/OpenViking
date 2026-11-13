"""Recall keeps information absent from the query and already accepted memories."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking_cli.client.identity import OVIdentity
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext

IDENTITY = OVIdentity("tenant", "alice", "ppt")
QUERY = "请修改 PPT 的标题，并交付文件。"
CODE = "```python\nimport re\n\ndef helper(text):\n    return re.sub(r'old', 'new', text)\n```"


def memory(name, body, score=0.9, category="events"):
    return MatchedContext(
        uri=f"viking://user/alice/memories/{category}/{name}.md",
        context_type=ContextType.MEMORY,
        category=category,
        level=2,
        score=score,
        abstract=body,
    )


def reader(items, bodies, **options):
    client = SimpleNamespace(
        find=AsyncMock(return_value=FindResult(memories=items, resources=[], skills=[])),
        read=AsyncMock(side_effect=lambda uri, **kwargs: bodies[uri]),
    )
    config = {
        "target_uris": ("viking://user/memories",),
        "recall_detail": "full",
        "max_injected_chars": 10000,
        "recall_max_tokens": 8000,
    }
    config.update(options)
    return OVContextReader(client, MemoryReaderConfig(**config))


async def test_event_removes_repeated_request_but_keeps_complete_extra_code():
    item = memory("helper", f"Summary: 用户要求{QUERY}")
    body = f"{item.abstract}\n\n{QUERY}\n\n可复用辅助函数：\n\n{CODE}"
    result = await reader([item], {item.uri: body}).recall(QUERY, identity=IDENTITY)

    assert result.status == "success" and result.count == 1
    assert CODE in result.content
    assert QUERY not in result.content
    assert 'detail="full"' in result.content


@pytest.mark.parametrize("mode", ["context", "find"])
async def test_query_only_candidate_uses_no_slot_and_later_candidate_fills_it(mode):
    repeated = memory("request", f"Summary: 用户要求{QUERY}", score=0.99)
    useful = memory("method", "先检查文本节点边界，再修改目标。", score=0.6)
    bodies = {repeated.uri: QUERY, useful.uri: useful.abstract}
    result = await reader(
        [repeated, useful], bodies, recall_limit=1, search_mode=mode, prefer_abstract=False
    ).recall(QUERY, identity=IDENTITY)

    assert result.count == 1 and useful.abstract in result.content
    assert repeated.uri not in result.content
    assert bodies[repeated.uri] == QUERY
    if mode == "context":
        reasons = {r["uri"]: r["output_status"] for r in result.stats["selection"]}
        assert reasons[repeated.uri] == "duplicate_content"
        assert reasons[useful.uri] == "selected"


@pytest.mark.parametrize("mode", ["context", "find"])
async def test_shared_code_appears_once_but_each_records_extra_information_survives(mode):
    first = memory("first", "函数使用方法", score=0.9, category="entities")
    second = memory("second", "函数校验方法", score=0.8, category="entities")
    bodies = {
        first.uri: "先确认输入文本。\n\n" + CODE,
        second.uri: CODE + "\n\n输出后检查修改数量。",
    }
    result = await reader([first, second], bodies, search_mode=mode, prefer_abstract=False).recall(
        QUERY, identity=IDENTITY
    )

    assert result.count == 2
    assert result.content.count(CODE) == 1
    assert "先确认输入文本。" in result.content
    assert "输出后检查修改数量。" in result.content


@pytest.mark.parametrize(
    "query,body",
    [
        ("将费用设为5.5元。", "将费用设为55元。"),
        ("请安排2026-11-11的会议。", "请安排2026-11-12的会议。"),
        ("请完成校验。", "请不要完成校验。"),
        ("请确认是否已付款，然后更新记录。", "已付款"),
    ],
)
async def test_changed_facts_negation_and_new_short_answers_are_preserved(query, body):
    item = memory("fact", body)
    result = await reader([item], {item.uri: body}).recall(query, identity=IDENTITY)
    assert result.count == 1 and body in result.content


async def test_changed_code_is_kept_as_a_complete_block():
    changed = CODE.replace("'new'", "'NEW'")
    item = memory("code", "代码实现")
    result = await reader([item], {item.uri: changed}).recall(
        QUERY + "\n\n" + CODE, identity=IDENTITY
    )
    assert result.count == 1 and changed in result.content


async def test_code_already_in_query_is_removed_as_one_block():
    item = memory("code", "代码实现")
    result = await reader([item], {item.uri: CODE + "\n\n输出需核对节点数量。"}).recall(
        QUERY + "\n\n" + CODE, identity=IDENTITY
    )
    assert result.count == 1 and "输出需核对节点数量。" in result.content
    assert "def helper" not in result.content


async def test_budget_rejection_does_not_mark_its_code_as_accepted():
    rejected = memory("x" * 600, "仅有标题", score=0.99)
    accepted = memory("short", "代码实现", score=0.8)
    result = await reader(
        [rejected, accepted],
        {rejected.uri: CODE, accepted.uri: CODE},
        recall_limit=1,
        max_injected_chars=400,
    ).recall(QUERY, identity=IDENTITY)
    assert result.count == 1 and CODE in result.content
    assert accepted.uri in result.content and rejected.uri not in result.content
    assert result.stats["used_chars"] == len(result.content) <= 400


async def test_original_history_request_keeps_historical_context():
    query = "查看之前修改 PPT 的原始任务要求"
    item = memory("history", query)
    body = query + "\n\n" + CODE
    result = await reader([item], {item.uri: body}).recall(query, identity=IDENTITY)
    assert body in result.content


async def test_equal_excerpts_do_not_hide_different_body_information():
    first = memory("first", "通用辅助函数", category="entities", score=0.9)
    second = memory("second", "通用辅助函数", category="entities", score=0.8)
    result = await reader(
        [first, second],
        {first.uri: CODE, second.uri: CODE + "\n\n发生空节点时保留原有段落。"},
    ).recall(QUERY, identity=IDENTITY)
    assert result.count == 2
    assert result.content.count(CODE) == 1
    assert "发生空节点时保留原有段落。" in result.content


async def test_query_dedup_precedes_budget_and_keeps_a_full_code_view():
    query = QUERY + "请按要求核对目标文字并交付修改后的文件。" * 100
    item = memory("helper", "代码实现")
    result = await reader([item], {item.uri: query + "\n\n" + CODE}, max_injected_chars=400).recall(
        query, identity=IDENTITY
    )
    assert result.count == 1 and CODE in result.content
    assert 'detail="full"' in result.content
    assert result.stats["used_chars"] == len(result.content) <= 400
    decision = result.stats["selection"][0]
    assert decision["query_duplicate_blocks"] == 1
    assert decision["content_chars_before"] > 400 > decision["content_chars_after"]


async def test_shared_reader_does_not_share_dedup_state_between_calls():
    item = memory("helper", "代码实现")
    instance = reader([item], {item.uri: CODE})
    for _ in range(2):
        result = await instance.recall(QUERY, identity=IDENTITY)
        assert result.count == 1 and CODE in result.content


@pytest.mark.parametrize("code", [CODE.replace("```", "~~~"), CODE.removesuffix("```")])
async def test_alternative_and_incomplete_fences_remain_atomic(code):
    item = memory("helper", "代码实现")
    result = await reader([item], {item.uri: QUERY + "\n\n" + code}).recall(
        QUERY, identity=IDENTITY
    )
    assert result.count == 1 and code in result.content
    assert QUERY not in result.content


@pytest.mark.parametrize("body", ["# Empty", "```python\n```", "```\n\n```"])
async def test_empty_content_is_not_reported_as_a_duplicate(body):
    item = memory("empty", "# Empty")
    result = await reader([item], {item.uri: body}).recall(QUERY, identity=IDENTITY)
    assert result.count == 0
    assert result.stats["selection"][0]["output_status"] == "no_information"


async def test_equal_code_text_in_a_different_language_is_not_a_duplicate():
    query_code = "```python\nvalue = 1\n```"
    memory_code = "```ruby\nvalue = 1\n```"
    item = memory("code", "代码实现")
    result = await reader([item], {item.uri: memory_code}).recall(
        QUERY + "\n\n" + query_code, identity=IDENTITY
    )
    assert result.count == 1 and memory_code in result.content


async def test_fully_repeated_body_cannot_consume_a_slot_via_a_different_abstract():
    first = memory("first", "函数实现A", score=0.99, category="entities")
    repeated = memory("second", "函数实现B", score=0.9, category="entities")
    novel = memory("third", "输出后需核对修改数量。", score=0.8, category="entities")
    result = await reader(
        [first, repeated, novel],
        {first.uri: CODE, repeated.uri: CODE, novel.uri: novel.abstract},
        recall_limit=2,
    ).recall(QUERY, identity=IDENTITY)
    assert result.count == 2
    assert first.uri in result.content and novel.uri in result.content
    assert repeated.uri not in result.content
    assert result.content.count(CODE) == 1
