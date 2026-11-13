"""Ranking relevance, noise resistance and lossless diversity contracts."""

import pytest

from openviking.integrations.langchain.ov_context.recall_ranking import rank_candidates
from openviking_cli.retrieve.context import RecallPolicy
from openviking_cli.retrieve.types import ContextType, MatchedContext


def memory(name, text, score=0.7, kind="tools", *, date="2026/11/03"):
    folder = f"events/{date}" if kind == "events" else kind
    return MatchedContext(
        uri=f"viking://user/alice/memories/{folder}/{name}.md",
        context_type=ContextType.MEMORY,
        category=kind,
        abstract=text,
        score=score,
    )


def ranked(items, query, **kwargs):
    trace = []
    result = rank_candidates(
        items, query, RecallPolicy(**kwargs), experience=False, diagnostics=trace
    )
    return result, {row["uri"]: row for row in trace}


def test_unstructured_excerpts_do_not_guess_method_value():
    first = memory("requested", "release audit deployment rollback validation", 0.91, "events")
    second = memory("completed", "release audit deployment rollback validation done", 0.9, "events")
    method = memory("procedure", "inspect config and refresh refs", 0.72, "skills")
    rows, trace = ranked([first, second, method], "办理发布核验")
    assert rows == [first, second, method]
    assert trace[method.uri]["reason"] == "relevance_fallback"
    assert trace[first.uri]["reason"] == trace[second.uri]["reason"] == "relevance_fallback"


def test_different_dated_events_are_not_treated_as_repetitions():
    first = memory("first", "release validation succeeded", 0.9, "events")
    second = memory("second", "release validation succeeded", 0.85, "events", date="2026/11/04")
    rows, trace = ranked([first, second], "汇总发布结果")
    assert rows == [first, second]
    assert trace[second.uri]["status"] == "ranked"


def test_same_summary_events_still_keep_separate_identity():
    items = [memory(str(n), "发布验证成功", 0.9 - n * 0.01, "events") for n in range(3)]
    rows, _ = ranked(items, "发布记录")
    assert rows == items


@pytest.mark.parametrize("query", ["2030 2030 11 11", "请输出 update.csv", "请保存 like.csv"])
def test_numbers_and_output_names_do_not_create_query_bonuses(query):
    event = memory("old", "Historical output update csv like", 0.7, "events")
    method = memory("method", "Read a current form and verify it", 0.71)
    rows, trace = ranked([event, method], query)
    assert rows[0] == method
    assert trace[event.uri]["lexical_bonus"] == 0
    assert trace[event.uri]["base_score"] == pytest.approx(0.82)


def test_namespace_is_not_lexical_evidence():
    scope = memory("unrelated", "unrelated content", 0.7)
    scope.uri = "viking://user/retry/peers/retry/memories/tools/unrelated.md"
    actual = memory("actual", "retry transient requests", 0.7)
    rows, trace = ranked([scope, actual], "retry")
    assert rows[0] == actual
    assert trace[scope.uri]["lexical_bonus"] == 0


def test_repeating_query_terms_does_not_multiply_a_bonus():
    item = memory("procedure", "retry transient requests")
    _, single = ranked([item], "retry")
    _, repeated = ranked([item], "retry retry retry retry")
    assert single[item.uri]["lexical_bonus"] == repeated[item.uri]["lexical_bonus"]


def test_exact_words_do_not_match_substrings():
    substring = memory("finance", "capital allocation", 0.7)
    exact = memory("procedure", "api request validation", 0.7)
    rows, trace = ranked([substring, exact], "api")
    assert rows[0] == exact
    assert trace[substring.uri]["lexical_bonus"] == 0


def test_chinese_business_terms_are_matched():
    unrelated = memory("other", "差旅酒店预订与付款", 0.7)
    relevant = memory("procedure", "核对报工时长，保存后检查累计增量", 0.7)
    rows, trace = ranked([unrelated, relevant], "核对报工时长")
    assert rows[0] == relevant
    assert trace[relevant.uri]["lexical_bonus"] > trace[unrelated.uri]["lexical_bonus"]


@pytest.mark.parametrize("query", ["读取 /tmp/report.csv", "find report.csv", "report.csv 在哪里"])
def test_explicit_file_lookups_preserve_filename_matching(query):
    unrelated = memory("other", "invoice processing", 0.7)
    relevant = memory("document", "report csv contains the report data", 0.7)
    rows, trace = ranked([unrelated, relevant], query)
    assert rows[0] == relevant
    assert trace[relevant.uri]["lexical_bonus"] > 0


def test_later_output_verb_does_not_inherit_an_earlier_lookup():
    output = memory("image", "unique_export png", 0.7)
    _, trace = ranked([output], "读取页面后，将截图保存为 unique_export.png")
    assert trace[output.uri]["lexical_bonus"] == 0


def test_score_mode_and_experience_channel_remain_semantic_only():
    items = [
        memory("first", "overlapping release validation", 0.9, "events"),
        memory("second", "overlapping release validation done", 0.85, "events"),
        memory("method", "browser retry", 0.7, "skills"),
    ]
    for policy, experience in [(RecallPolicy(ranking="score"), False), (RecallPolicy(), True)]:
        assert rank_candidates(items, "browser browser", policy, experience=experience) == items


def test_noise_repair_does_not_change_the_raw_semantic_threshold():
    low = memory("browser", "browser", 0.49)
    high = memory("other", "unrelated", 0.51)
    rows, _ = ranked([low, high], "browser", score_threshold=0.5)
    assert rows == [high]


def test_empty_excerpts_do_not_produce_a_redundancy_penalty():
    items = [memory("a", "", 0.8, "events"), memory("b", "", 0.7, "events")]
    rows, trace = ranked(items, "事件")
    assert rows == items
    assert trace[items[1].uri]["reason"] == "relevance_fallback"


def test_novel_but_low_relevance_document_does_not_jump_over_strong_evidence():
    first = memory("first", "release deployment validation completed", 0.95, "events")
    second = memory("second", "release deployment validation completed successfully", 0.9, "events")
    unrelated = memory("unrelated", "hotel room reservation payment", 0.12, "skills")
    rows, _ = ranked([first, second, unrelated], "发布核验")
    assert rows.index(second) < rows.index(unrelated)


def test_nonoverlapping_documents_keep_their_relevance_order():
    items = [
        memory("a", "release deployment", 0.9),
        memory("b", "hotel reservation", 0.8),
        memory("c", "invoice payment", 0.7),
    ]
    rows, trace = ranked(items, "处理")
    assert rows == items
    assert all(row["reason"] == "relevance_fallback" for row in trace.values())


def test_event_category_alone_does_not_reduce_a_candidates_relevance():
    event = memory("completed", "Summary: 用户要求修改 PPT；此前已交付旧版本。", 0.99, "events")
    method = memory(
        "pptx", "Skill: pptx\nBest for: 修改 PPT 文本\nGuidelines:\n- 按 run 定位", 0.6, "skills"
    )
    rows, trace = ranked([event, method], "请修改 PPT 文本并交付文件")
    assert rows == [event, method]
    assert trace[event.uri]["reason"] == "relevance_fallback"


def test_historical_event_lookup_keeps_event_priority():
    event = memory("completed", "Summary: 之前修改 PPT 已完成。", 0.99, "events")
    method = memory("pptx", "修改 PPT 的方法", 0.6, "skills")
    rows, trace = ranked([event, method], "查看之前修改 PPT 的结果")
    assert rows[0] == event
    assert trace[event.uri]["reason"] != "event_summary_deferred"
