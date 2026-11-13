# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Offline contracts through the real reader, SDK, router and search service."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.retrieve.intent_analyzer import IntentAnalyzer
from openviking.utils.token_estimation import estimate_text_tokens
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import InvalidArgumentError, PermissionDeniedError
from openviking_cli.retrieve.context import ContextSearchOptions, RecallPolicy
from openviking_cli.retrieve.types import ContextType
from openviking_cli.session.user_id import UserIdentifier
from tests.ov_context_support.recall_api import context, hit


async def reader_context(
    sdk, query, *, mode="context", detail=None, max_tokens=1600, recall_policy=None, quotas=None
):
    values = dict(recall_policy or {})
    config = MemoryReaderConfig(
        **values, recall_detail=detail, recall_max_tokens=max_tokens, recall_quotas=quotas
    )
    result = await OVContextReader(sdk, config).recall(
        query, identity=OVIdentity("tenant", "alice", "assistant")
    )
    # Parse the caller-visible XML view while retaining Reader's diagnostics.
    import re

    entries = []
    for uri, category, score, level, text in re.findall(
        r'<memory uri="([^"]+)" type="([^"]+)" score="([^"]+)" detail="([^"]+)">\n(.*?)\n</memory>',
        result.content,
        re.S,
    ):
        entries.append(
            {"uri": uri, "category": category, "score": float(score), "detail": level, "text": text}
        )
    return {"entries": entries, "rendered": result.content, "stats": result.stats}


@pytest.mark.parametrize("config", [None, MemoryReaderConfig()])
async def test_reader_default_context_finds_user_peer_once_without_session(
    api, monkeypatch, config
):
    fs, service, sdk = api
    search = AsyncMock(wraps=sdk.search)
    monkeypatch.setattr(sdk, "search", search)
    find = AsyncMock(wraps=sdk.find)
    monkeypatch.setattr(sdk, "find", find)
    reader = OVContextReader(sdk) if config is None else OVContextReader(sdk, config)
    assert reader.config.search_mode == "context"
    result = await reader.recall(
        "  original query  ",
        session_id="ignored-session",
        identity=OVIdentity("tenant", "alice", "assistant"),
    )
    assert result.status == "success" and result.count == 5
    assert len(fs.calls) == 1 and fs.embedder.calls == 1
    assert fs.calls[0][0] == "  original query  "
    assert fs.calls[0][2] == [
        "viking://user/memories",
        "viking://user/peers/assistant/memories",
    ]
    assert fs.calls[0][3] == 25
    assert not fs.reads and len(result.content) <= 4000
    search.assert_not_awaited()
    assert result.stats["candidate_sdk_calls"] == 1
    find.assert_awaited_once()
    assert find.await_args.kwargs["identity"] == OVIdentity("tenant", "alice", "assistant")
    service.sessions.session.assert_not_called()
    fs.search.assert_not_called()
    IntentAnalyzer.analyze.assert_not_called()


async def test_custom_scopes_and_other_memory_types_are_preserved(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context", target_uris=("viking://user/alice/memories/cases",)
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 1 and "/cases/0.md" in result.content
    assert fs.calls[0][2] == cfg.target_uris[0]


async def test_multiple_reader_scopes_share_one_embedding(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context",
        target_uris=(
            "viking://user/alice/memories/cases",
            "viking://user/alice/memories/preferences",
        ),
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 3
    assert len(fs.calls) == 1 and fs.embedder.calls == 1


async def test_context_combined_scope_failure_is_diagnosed_without_retries(api, monkeypatch):
    fs, _, sdk = api
    targets = []

    async def find(query, *, target_uri, **kwargs):
        targets.append(target_uri)
        raise RuntimeError("offline scope")

    monkeypatch.setattr(fs, "find", find)
    cfg = MemoryReaderConfig(
        search_mode="context",
        target_uris=(
            "viking://user/alice/memories/unavailable",
            "viking://user/alice/memories/cases",
        ),
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.status == "error" and result.count == 0
    assert not result.content
    assert targets == [list(cfg.target_uris)]


async def test_auth_errors_are_raised_but_optional_failures_are_diagnosed(api):
    fs, _, sdk = api
    reader = OVContextReader(sdk, MemoryReaderConfig(search_mode="context"))
    fs.error = PermissionDeniedError("forbidden")
    with pytest.raises(PermissionDeniedError):
        await reader.recall("q", identity=OVIdentity("tenant", "alice", "assistant"))
    fs.error = RuntimeError("unavailable")
    assert (
        await reader.recall("q", identity=OVIdentity("tenant", "alice", "assistant"))
    ).status == "error"


async def test_reader_timeout_cancels_each_category_find(api):
    fs, _, sdk = api
    fs.embedder.release.clear()
    cfg = MemoryReaderConfig(
        search_mode="context", recall_timeout=0.03, recall_quotas={"events": 1, "preferences": 1}
    )
    assert (
        await OVContextReader(sdk, cfg).recall(
            "q", identity=OVIdentity("tenant", "alice", "assistant")
        )
    ).status == "timeout"
    for _ in range(20):
        await asyncio.sleep(0)
    assert fs.embedder.calls == fs.embedder.cancelled == 2


async def test_memories_quota_rejected_at_reader_sdk_service_and_http(api):
    fs, service, sdk = api

    with pytest.raises(ValueError):
        MemoryReaderConfig(search_mode="context", recall_quotas={"memories": 1})
    with pytest.raises(ValueError):
        await sdk.search("q", mode="context", quotas={"memories": 1})
    with pytest.raises(InvalidArgumentError):
        await service.search.search("q", ctx=context(), mode="context", quotas={"memories": 1})
    response = await sdk._http.post(
        "http://fixture/api/v1/search/search",
        json={"query": "q", "mode": "context", "quotas": {"memories": 1}},
    )
    assert response.status_code == 422
    assert not fs.calls and fs.embedder.calls == 0
    assert ContextSearchOptions(mode="context", detail={"memories": "abstract"}).detail


async def test_reader_recalls_all_six_buckets_with_local_skill_roots(api):
    fs, _, sdk = api
    fs.extra_items = [
        hit("viking://user/alice/resources/guide.md", ContextType.RESOURCE),
        hit("viking://user/alice/skills/deploy/SKILL.md", ContextType.SKILL),
        hit("viking://user/alice/memories/skills/usage.md"),
        hit("viking://user/bob/skills/private/SKILL.md", ContextType.SKILL, 0.99),
    ]
    quotas = dict.fromkeys(
        ("events", "entities", "preferences", "experiences", "resources", "skills"), 1
    )
    cfg = MemoryReaderConfig(search_mode="context", recall_quotas=quotas, recall_limit=1)
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.status == "success" and result.count == 6
    for category in quotas:
        assert f'type="{category}"' in result.content
    assert "usage.md" not in result.content and "/bob/" not in result.content
    targets = {
        root for call in fs.calls for root in (call[2] if isinstance(call[2], list) else [call[2]])
    }
    assert "viking://user/alice/skills" in targets
    assert "viking://agent/skills" not in targets
    assert fs.embedder.calls == len(fs.calls) == 6


async def test_quota_scope_prevents_named_and_nested_category_starvation(api):
    fs, _, sdk = api
    fs.categories = {"events": 50, "preferences": 1}
    fs.extra_items = [
        hit(f"viking://user/alice/memories/custom/preferences/{i}.md", score=0.99)
        for i in range(50)
    ]
    cfg = MemoryReaderConfig(
        search_mode="context", candidate_limit=20, recall_quotas={"preferences": 1}
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 1 and "/memories/preferences/0.md" in result.content
    assert len(fs.calls) == 1
    assert all(root.endswith("/memories/preferences") for root in fs.calls[0][2])


@pytest.mark.parametrize(
    "target", ["viking://user/alice/memories", "viking://user/alice/memories/cases"]
)
async def test_reader_explicit_memory_scope_never_expands_to_resources_or_skills(api, target):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context",
        target_uris=(target,),
        recall_quotas={"resources": 1, "skills": 1},
    )
    assert (
        await OVContextReader(sdk, cfg).recall(
            "q", identity=OVIdentity("tenant", "alice", "assistant")
        )
    ).status == "empty"
    assert not fs.calls


async def test_flat_recall_keeps_other_memory_category_but_skips_profile_and_preset_root(api):
    fs, _, sdk = api
    fs.categories = {"trajectories": 1, "tools": 1, "skills": 1, "custom": 1}
    root = "viking://user/alice/memories"
    fs.extra_items = [hit(f"{root}/{name}.md") for name in ("profile", "identity", "soul")]
    fs.extra_items += [
        hit(f"{root}/{name}.md", level=level)
        for name, level in ((".abstract", 0), (".overview", 1))
    ]
    fs.extra_items += [hit(f"{root}/custom/experiences/nested.md")]
    cfg = MemoryReaderConfig(
        search_mode="context",
        recall_limit=20,
        recall_detail={"memories": "abstract"},
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 7
    assert "profile.md" not in result.content
    assert "/memories/.abstract.md" not in result.content
    assert "/memories/.overview.md" not in result.content
    assert result.content.count('type="memories"') == 7


async def test_explicit_empty_reader_scope_disables_quota_search(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(search_mode="context", target_uris=(), recall_quotas={"events": 1})
    assert (
        await OVContextReader(sdk, cfg).recall(
            "q", identity=OVIdentity("tenant", "alice", "assistant")
        )
    ).status == "empty"
    assert not fs.calls


@pytest.mark.parametrize("detail", ["full", "overview", "abstract"])
@pytest.mark.parametrize("quotas", [None, {"events": 2}])
@pytest.mark.parametrize("higher_scope", ["user", "peer"])
async def test_event_mirrors_share_one_output_slot_and_backfill(api, detail, quotas, higher_scope):
    fs, _, sdk = api
    fs.categories = {}
    user = "viking://user/alice/memories/events/2026/11/03/meeting.md"
    peer = "viking://user/alice/peers/assistant/memories/events/2026/11/03/meeting.md"
    other = "viking://user/alice/memories/events/2026/11/04/meeting.md"
    winner, replica = (user, peer) if higher_scope == "user" else (peer, user)
    fs.extra_items = [hit(winner, score=0.99), hit(replica, score=0.98), hit(other, score=0.8)]

    async def read(uri, *, ctx):
        return b"same complete event and source metadata" if uri != other else b"another event"

    fs.read = AsyncMock(side_effect=read)
    # Keep the mirrored replica adjacent so this contract exercises body dedup,
    # independently of query-aware diversity ordering.
    policy = RecallPolicy(
        ranking="score",
        target_uris=("viking://user/memories", "viking://user/peers/assistant/memories"),
        recall_limit=2,
        max_injected_chars=10000,
    )
    result = await reader_context(
        sdk, "q", mode="context", detail=detail, quotas=quotas, recall_policy=policy.model_dump()
    )
    assert [e["uri"] for e in result["entries"]] == [winner, other]
    assert result["stats"]["event_replicas_suppressed"] == 1
    reads = [call.args[0] for call in fs.read.call_args_list]
    assert reads.count(user) == reads.count(peer) == 1
    assert estimate_text_tokens(result["rendered"]) <= 1600


@pytest.mark.parametrize(
    "peer_path,body",
    [
        ("2026/11/03/meeting.md", b"different full event"),
        ("2026/11/04/meeting.md", b"same full event"),
        ("2026/11/03/other.md", b"same full event"),
        ("meeting.md", b"same full event"),
        (
            "2026/11/03/meeting.md",
            b'same full event\n\n<!-- MEMORY_FIELDS\n{"source_session_id":"another"}\n-->',
        ),
    ],
)
async def test_distinct_events_survive_identical_summaries(api, peer_path, body):
    fs, _, sdk = api
    fs.categories = {}
    user = "viking://user/alice/memories/events/2026/11/03/meeting.md"
    peer = f"viking://user/alice/peers/assistant/memories/events/{peer_path}"
    fs.extra_items = [hit(user, score=0.99), hit(peer, score=0.98)]
    for item in fs.extra_items:
        item.abstract = "identical event summary"

    async def read(uri, *, ctx):
        return b"same full event" if uri == user else body

    fs.read = AsyncMock(side_effect=read)
    result = await reader_context(
        sdk,
        "q",
        mode="context",
        detail="abstract",
        recall_policy=RecallPolicy(
            target_uris=("viking://user/memories", "viking://user/peers/assistant/memories")
        ).model_dump(),
    )
    assert [e["uri"] for e in result["entries"]] == [user, peer]
    assert result["stats"]["event_replicas_suppressed"] == 0


async def test_event_replica_comparison_failure_preserves_both_candidates(api):
    fs, _, sdk = api
    fs.categories = {}
    user = "viking://user/alice/memories/events/2026/11/03/meeting.md"
    peer = "viking://user/alice/peers/assistant/memories/events/2026/11/03/meeting.md"
    fs.extra_items = [hit(user, score=0.99), hit(peer, score=0.98)]
    fs.read = AsyncMock(side_effect=RuntimeError("storage temporarily unavailable"))
    result = await reader_context(
        sdk,
        "q",
        mode="context",
        detail="abstract",
        recall_policy=RecallPolicy(
            target_uris=("viking://user/memories", "viking://user/peers/assistant/memories")
        ).model_dump(),
    )
    assert [e["uri"] for e in result["entries"]] == [user, peer]
    assert result["stats"]["event_replicas_suppressed"] == 0
    assert len(result["stats"]["read_errors"]) == 2


async def test_event_replica_rejected_by_budget_does_not_hide_smaller_copy(api):
    from openviking.integrations.langchain.ov_context.recall_assembly import _render_entry

    fs, _, sdk = api
    fs.categories = {}
    user = "viking://user/alice/memories/events/2026/11/03/meeting.md"
    peer = "viking://user/alice/peers/assistant/memories/events/2026/11/03/meeting.md"
    fs.extra_items = [hit(peer, score=0.99), hit(user, score=0.98)]
    fs.read = AsyncMock(return_value=b"same event")
    for item in fs.extra_items:
        item.abstract = ""
    budget = estimate_text_tokens(
        _render_entry(
            {
                "uri": user,
                "category": "events",
                "score": 0.98,
                "detail": "full",
                "text": "same event",
            }
        )
    )
    result = await reader_context(
        sdk,
        "q",
        mode="context",
        detail="full",
        max_tokens=budget,
        recall_policy=RecallPolicy(
            target_uris=("viking://user/memories", "viking://user/peers/assistant/memories")
        ).model_dump(),
    )
    assert [e["uri"] for e in result["entries"]] == [user]
    assert result["stats"]["event_replicas_suppressed"] == 0
    assert estimate_text_tokens(result["rendered"]) <= budget


@pytest.mark.parametrize(
    "uri,category,level,actor",
    [
        ("viking://user/bob/memories/events/2026/11/03/e.md", "events", 2, "assistant"),
        (
            "viking://user/alice/peers/other/memories/events/2026/11/03/e.md",
            "events",
            2,
            "assistant",
        ),
        ("viking://user/alice/memories/events/2026/11/03/e.md", "events", 0, "assistant"),
        ("viking://user/alice/memories/events/2026/11/03/e.md", "memories", 2, "assistant"),
        ("viking://user/alice/memories/events/2026/11/03/e.md", "events", 2, None),
    ],
)
def test_event_replica_scope_is_limited_to_current_user_peer(uri, category, level, actor):
    from openviking.integrations.langchain.ov_context.recall_assembly import (
        Candidate,
        _event_replica_scope,
        _RecallScope,
    )

    ctx = _RecallScope(UserIdentifier("tenant", "alice"), actor_peer_id=actor)
    candidate = Candidate(hit(uri, level=level), category, False)
    assert _event_replica_scope(candidate, ctx) is None


@pytest.mark.parametrize("detail", ["full", "overview", "abstract"])
async def test_memory_recall_cleans_internal_fields_without_changing_stored_content(api, detail):
    fs, _, sdk = api
    fs.categories = {}
    uri = "viking://user/alice/memories/tools/browser.md"
    body = (
        "Tool: browser\n\n- Optimal params: Use batch --json with fresh refs.\n"
        "- Common failures: Stale refs cause incorrect fields."
    )
    raw = body + '\n\n<!-- MEMORY_FIELDS\n{"source_id":"private-source"}\n-->'
    item = hit(uri)
    item.abstract = raw
    fs.extra_items = [item]
    fs.read = AsyncMock(return_value=raw.encode())
    result = await reader_context(sdk, "q", mode="context", detail=detail, max_tokens=3200)
    assert len(result["entries"]) == 1
    assert "MEMORY_FIELDS" not in result["rendered"] and "private-source" not in result["rendered"]
    assert "batch --json" in result["rendered"] and "Stale refs" in result["rendered"]
    assert fs.read.return_value == raw.encode()
    if detail == "full":
        assert result["entries"][0]["text"] == body
    assert estimate_text_tokens(result["rendered"]) <= 3200


async def test_resource_full_view_preserves_documented_memory_field_examples(api):
    fs, _, sdk = api
    fs.categories = {}
    fs.extra_items = [hit("viking://user/alice/resources/format.md", ContextType.RESOURCE)]
    raw = 'Example format:\n\n<!-- MEMORY_FIELDS\n{"example":"documented syntax"}\n-->'
    fs.read = AsyncMock(return_value=raw.encode())
    result = await reader_context(sdk, "q", mode="context", detail="full", quotas={"resources": 1})
    assert result["entries"][0]["text"] == raw


async def test_heading_only_memory_preview_does_not_consume_output_slot(api):
    fs, _, sdk = api
    fs.categories = {}
    item = hit("viking://user/alice/memories/tools/browser.md")
    item.abstract = "Tool: browser\n\n## Guidelines\n### Good Cases\n### Bad Cases"
    fs.extra_items = [item]
    fs.read = AsyncMock(return_value=item.abstract.encode())
    result = await reader_context(sdk, "q", mode="context", detail="overview")
    assert result["entries"] == []


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("missing_abstract", [False, True])
async def test_applicable_method_precedes_event_and_budget_controls_the_event_view(
    api, limit, missing_abstract
):
    fs, _, sdk = api
    fs.categories = {}
    event_uri = "viking://user/alice/memories/events/2026/11/10/ppt_done.md"
    method_uri = "viking://user/alice/memories/skills/pptx.md"
    event = hit(event_uri, score=0.99)
    event.abstract = "Summary: 之前 PPT 修改已完成。"
    event_body = event.abstract + "\n2026-11-10 ChatLog:\n[ppt-agent]: " + "重复核验旧目标。" * 1000
    event_summary = event.abstract
    if missing_abstract:
        event.abstract = ""
    method = hit(method_uri, score=0.6)
    method.abstract = (
        "# PPT修改\n- Domain: PPT\n- Trigger: 修改 PPT 文本\n"
        "- Operation Intent:\n  - Family: update_existing_object\n"
        "  - Primary object: PPT\n- Procedure:\n"
        "  1. 按 run 定位\n  2. 修改并核对目标文字"
    )
    fs.extra_items = [event, method]
    fs.read = AsyncMock(
        side_effect=lambda uri, **kwargs: (
            event_body if uri == event_uri else method.abstract
        ).encode()
    )
    result = await reader_context(
        sdk,
        "请修改 PPT 文本并交付文件",
        mode="context",
        detail="full",
        max_tokens=3200,
        recall_policy=RecallPolicy(recall_limit=limit, max_injected_chars=16000).model_dump(),
    )
    assert result["entries"][0]["uri"] == method_uri
    assert "ChatLog" not in result["rendered"]
    events = [e for e in result["entries"] if e["category"] == "events"]
    if limit == 1:
        assert events == []
    else:
        assert len(events) == 1
        assert events[0]["detail"] in {"abstract", "overview"}
        assert " ".join(events[0]["text"].split()) == event_summary


async def test_history_lookup_still_allows_full_event_chatlog(api):
    fs, _, sdk = api
    fs.categories = {}
    uri = "viking://user/alice/memories/events/2026/11/10/ppt_done.md"
    event = hit(uri, score=0.99)
    event.abstract = "Summary: 之前 PPT 修改已完成。"
    fs.extra_items = [event]
    body = event.abstract + "\n2026-11-10 ChatLog:\n[ppt-agent]: 历史交付详情。"
    fs.read = AsyncMock(return_value=body.encode())
    result = await reader_context(
        sdk,
        "查看之前修改 PPT 的结果",
        mode="context",
        detail="full",
        max_tokens=3200,
        recall_policy=RecallPolicy().model_dump(),
    )
    assert result["entries"][0]["text"] == body
    assert result["entries"][0]["detail"] == "full"


async def test_value_selection_reports_actual_provided_methods_after_budgeting(api):
    fs, _, sdk = api
    fs.categories = {}
    method_uri = "viking://user/alice/memories/trajectories/create_meeting.md"
    event_uri = "viking://user/alice/memories/events/2026/11/03/old.md"
    body = """# 新建日程
- Domain: 日程管理
- Trigger: 新建会议日程
- Operation Intent:
  - Family: create_new_object
  - Primary object: 会议日程
- Preconditions:
  1. 已登录应用
- Procedure:
  1. 打开表单后 snapshot -i 获取 refs
  2. 使用 batch --json 批量填写表单字段
  3. 保存后核对日程详情
"""
    method = hit(method_uri, score=0.6)
    method.abstract = body
    event = hit(event_uri, score=0.95)
    event.abstract = "Summary: 用户要求新建会议日程。"
    fs.extra_items = [event, method]
    fs.read = AsyncMock(
        side_effect=lambda uri, **kwargs: (
            body.encode() if uri == method_uri else event.abstract.encode()
        )
    )
    result = await reader_context(
        sdk,
        "请新建会议日程",
        mode="context",
        detail="full",
        max_tokens=3200,
        recall_policy=RecallPolicy().model_dump(),
    )
    assert result["entries"][0]["uri"] == method_uri
    assert result["stats"]["structured_method_capabilities"] == ["batch_form", "refresh_refs"]
    decision = next(x for x in result["stats"]["selection"] if x["uri"] == method_uri)
    assert decision["reason"] == "applicable_primary_procedure"
    assert decision["output_status"] == "selected" and decision["actual_detail"] == "full"
    assert estimate_text_tokens(result["rendered"]) <= 3200

    # A planned method is not counted as provided when every view exceeds budget.
    empty = await reader_context(
        sdk,
        "请新建会议日程",
        mode="context",
        detail="full",
        max_tokens=1,
        recall_policy=RecallPolicy().model_dump(),
    )
    assert empty["entries"] == []
    assert empty["stats"]["structured_method_capabilities"] == []
    decision = next(x for x in empty["stats"]["selection"] if x["uri"] == method_uri)
    assert decision["output_status"] == "budget"


async def test_explicit_incompatible_operation_is_not_read_or_injected(api):
    fs, _, sdk = api
    fs.categories = {}
    uri = "viking://user/alice/memories/trajectories/create.md"
    row = hit(uri, score=0.95)
    row.abstract = """# 新建日程
- Domain: 日程管理
- Trigger: 新建会议日程
- Operation Intent:
  - Family: create_new_object
  - Primary object: 会议日程
- Procedure:
  1. 打开新建日程表单
  2. 保存新建的会议日程
"""
    fs.extra_items = [row]
    fs.read = AsyncMock(side_effect=AssertionError("An incompatible contract must not be read"))
    result = await reader_context(
        sdk,
        "修改已有会议日程，不要新建",
        mode="context",
        detail="full",
        recall_policy=RecallPolicy().model_dump(),
    )
    assert result["entries"] == []
    fs.read.assert_not_awaited()
    decision = result["stats"]["selection"][0]
    assert decision["reason"] == "explicit_operation_conflict"
    assert decision["output_status"] == "not_applicable"


async def test_reader_four_buckets_find_independently_and_read_complete_experience(api):
    fs, service, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context",
        experience_enabled=True,
        recall_quotas={"events": 2, "entities": 1, "preferences": 1, "experiences": 1},
    )
    result = await OVContextReader(sdk, cfg).recall(
        "hello", session_id="ignored", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.status == "success" and result.count == 5
    assert len(fs.calls) == fs.embedder.calls == 4
    assert "recorded request ID" in result.content
    assert 'type="experiences"' in result.content
    assert all("/experiences/" in uri for uri, _ in fs.reads)
    assert all(call[0] == "hello" for call in fs.calls)
    service.sessions.session.assert_not_called()


async def test_reader_custom_scopes_and_other_memory_types_are_preserved(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context", target_uris=("viking://user/alice/memories/cases",)
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 1 and "/cases/0.md" in result.content
    assert fs.calls[0][2] == cfg.target_uris[0]


async def test_reader_multiple_reader_scopes_share_one_embedding(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context",
        target_uris=(
            "viking://user/alice/memories/cases",
            "viking://user/alice/memories/preferences",
        ),
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 3
    assert len(fs.calls) == 1 and fs.embedder.calls == 1


async def test_reader_combined_scope_failure_is_diagnosed_without_retries(api, monkeypatch):
    fs, _, sdk = api
    targets = []

    async def find(query, *, target_uri, **kwargs):
        targets.append(target_uri)
        raise RuntimeError("offline scope")

    monkeypatch.setattr(fs, "find", find)
    cfg = MemoryReaderConfig(
        search_mode="context",
        target_uris=(
            "viking://user/alice/memories/unavailable",
            "viking://user/alice/memories/cases",
        ),
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.status == "error" and result.count == 0
    assert not result.content
    assert targets == [list(cfg.target_uris)]


async def test_reader_auth_errors_are_raised_but_optional_failures_are_diagnosed(api):
    fs, _, sdk = api
    reader = OVContextReader(sdk, MemoryReaderConfig(search_mode="context"))
    fs.error = PermissionDeniedError("forbidden")
    with pytest.raises(PermissionDeniedError):
        await reader.recall("q", identity=OVIdentity("tenant", "alice", "assistant"))
    fs.error = RuntimeError("unavailable")
    assert (
        await reader.recall("q", identity=OVIdentity("tenant", "alice", "assistant"))
    ).status == "error"


async def test_reader_memories_quota_rejected_at_reader_sdk_service_and_http(api):
    fs, service, sdk = api

    with pytest.raises(ValueError):
        MemoryReaderConfig(search_mode="context", recall_quotas={"memories": 1})
    with pytest.raises(ValueError):
        await sdk.search("q", mode="context", quotas={"memories": 1})
    with pytest.raises(InvalidArgumentError):
        await service.search.search("q", ctx=context(), mode="context", quotas={"memories": 1})
    response = await sdk._http.post(
        "http://fixture/api/v1/search/search",
        json={"query": "q", "mode": "context", "quotas": {"memories": 1}},
    )
    assert response.status_code == 422
    assert not fs.calls and fs.embedder.calls == 0
    assert ContextSearchOptions(mode="context", detail={"memories": "abstract"}).detail


async def test_reader_quota_scope_prevents_named_and_nested_category_starvation(api):
    fs, _, sdk = api
    fs.categories = {"events": 50, "preferences": 1}
    fs.extra_items = [
        hit(f"viking://user/alice/memories/custom/preferences/{i}.md", score=0.99)
        for i in range(50)
    ]
    cfg = MemoryReaderConfig(
        search_mode="context", candidate_limit=20, recall_quotas={"preferences": 1}
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 1 and "/memories/preferences/0.md" in result.content
    assert len(fs.calls) == 1
    assert all(root.endswith("/memories/preferences") for root in fs.calls[0][2])


async def test_reader_quotas_override_legacy_switches_and_counts_but_preserve_budgets(api):
    _, _, sdk = api
    cfg = MemoryReaderConfig(
        search_mode="context",
        enabled=False,
        experience_enabled=False,
        recall_limit=0,
        experience_limit=0,
        candidate_limit=1,
        experience_candidate_limit=1,
        recall_quotas={"events": 6, "experiences": 2},
        recall_max_tokens=8000,
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    # Quotas activate both channels, but the fixture's two experience files
    # contain the same complete body and must consume only one output slot.
    assert result.count == 7
    experience_decisions = [
        row for row in result.stats["selection"] if "/experiences/" in row["uri"]
    ]
    assert sorted(row["output_status"] for row in experience_decisions) == [
        "duplicate_content",
        "selected",
    ]
    limited = MemoryReaderConfig(
        search_mode="context",
        recall_quotas={"events": 6},
        max_injected_chars=1,
    )
    assert (
        await OVContextReader(sdk, limited).recall(
            "q", identity=OVIdentity("tenant", "alice", "assistant")
        )
    ).count == 0


async def test_reader_flat_recall_keeps_other_memory_category_but_skips_profile_and_preset_root(
    api,
):
    fs, _, sdk = api
    fs.categories = {"trajectories": 1, "tools": 1, "skills": 1, "custom": 1}
    root = "viking://user/alice/memories"
    fs.extra_items = [hit(f"{root}/{name}.md") for name in ("profile", "identity", "soul")]
    fs.extra_items += [
        hit(f"{root}/{name}.md", level=level)
        for name, level in ((".abstract", 0), (".overview", 1))
    ]
    fs.extra_items += [hit(f"{root}/custom/experiences/nested.md")]
    cfg = MemoryReaderConfig(
        search_mode="context",
        recall_limit=20,
        recall_detail={"memories": "abstract"},
    )
    result = await OVContextReader(sdk, cfg).recall(
        "q", identity=OVIdentity("tenant", "alice", "assistant")
    )
    assert result.count == 7
    assert "profile.md" not in result.content
    assert "/memories/.abstract.md" not in result.content
    assert "/memories/.overview.md" not in result.content
    assert result.content.count('type="memories"') == 7


async def test_reader_explicit_empty_reader_scope_disables_quota_search(api):
    fs, _, sdk = api
    cfg = MemoryReaderConfig(search_mode="context", target_uris=(), recall_quotas={"events": 1})
    assert (
        await OVContextReader(sdk, cfg).recall(
            "q", identity=OVIdentity("tenant", "alice", "assistant")
        )
    ).status == "empty"
    assert not fs.calls


@pytest.mark.parametrize("count_limit", [1, 5])
async def test_ordinary_file_method_reaches_actual_input_under_count_and_token_limits(
    api, count_limit
):
    fs, _, sdk = api
    fs.categories = {}
    method_uri = "viking://user/alice/memories/entities/sheet_processing.md"
    body = (
        "# 工作簿统计方法\n"
        "- 直接读取原工作簿 inputs/orders.xlsx，独立计算指标。\n"
        "- 使用 read_rows(sheet) 返回 raw 全部原字段和 source_row 原行号；"
        "先按期末截断，再按 event_id 去重、逻辑尝试取最大 seq。\n"
        "- 区分首次与重试，保留公式、分子、分母，校验结果的集合覆盖。"
    )
    method = hit(method_uri, score=0.62)
    method.abstract = body
    events = []
    for i in range(6):
        row = hit(f"viking://user/alice/memories/events/old-{i}.md", score=0.95 - i / 100)
        row.abstract = f"Summary: 订单统计报告第{i + 1}章已完成，历史成功率80%。"
        events.append(row)
    fs.extra_items = [*events, method]
    by_uri = {row.uri: row.abstract for row in fs.extra_items}
    fs.read = AsyncMock(side_effect=lambda uri, **kwargs: by_uri[uri].encode())
    result = await reader_context(
        sdk,
        "编写订单统计报告，读取原工作簿，保留原字段与原行号，区分首次重试，计算指标并核验结果。",
        detail="full",
        max_tokens=4000,
        recall_policy=RecallPolicy(recall_limit=count_limit, max_injected_chars=4000).model_dump(),
    )
    assert result["entries"][0]["uri"] == method_uri
    assert "read_rows(sheet)" in result["rendered"]
    assert "source_row" in result["rendered"]
    assert len(result["entries"]) <= count_limit
    assert estimate_text_tokens(result["rendered"]) <= 4000
    assert len(result["rendered"]) <= 4000
    assert {"preserve_source_records", "calculate_metric", "validate_result"} <= set(
        result["stats"]["structured_method_capabilities"]
    )
    assert by_uri[method_uri] == body
