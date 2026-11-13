# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Offline contracts through the real reader, SDK, router and search service."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.client.local import LocalClient
from openviking.models.embedder.base import embed_compat
from openviking.models.embedder.query_cache import query_embed_cache_scope
from openviking.retrieve.intent_analyzer import IntentAnalyzer
from openviking.utils.token_estimation import estimate_text_tokens
from openviking_cli.client.identity import OVIdentity
from openviking_cli.client.sync_http import SyncHTTPClient
from openviking_cli.exceptions import InvalidArgumentError, PermissionDeniedError
from openviking_cli.retrieve.context import ContextSearchOptions, RecallPolicy
from openviking_cli.retrieve.types import ContextType, FindResult, MatchedContext
from tests.ov_context_support.recall_api import Embedder, context, hit


async def test_context_with_session_id_bypasses_history_on_server_and_local(api):
    fs, service, sdk = api
    result = await sdk.search("q", mode="context", session_id="no-load")
    assert result["entries"] and len(fs.calls) == 1
    local = LocalClient.__new__(LocalClient)
    local._service, local._ctx = service, context()
    result = await local.search("q", mode="context", session_id="no-load")
    assert result["entries"] and fs.embedder.calls == 2
    service.sessions.session.assert_not_called()


@pytest.mark.parametrize("quotas", [{}, {"events": 0, "experiences": 0}])
async def test_empty_quotas_do_not_call_embedding_or_storage(api, quotas):
    fs, _, sdk = api
    result = await sdk.search("q", mode="context", quotas=quotas)
    assert result["entries"] == [] and result["rendered"] == ""
    assert not fs.calls and fs.embedder.calls == 0


@pytest.mark.parametrize("limit", [-1, 0, 201])
@pytest.mark.parametrize("quotas", [{"events": 1}, {}, {"events": 0}])
async def test_quota_mode_ignores_out_of_range_limit(api, limit, quotas):
    fs, _, sdk = api
    result = await sdk.search("q", mode="context", quotas=quotas, limit=limit)
    assert len(result["entries"]) == sum(quotas.values())
    assert result["stats"]["ignored"] == ["limit"]
    if quotas.get("events", 0):
        assert result["entries"][0]["category"] == "events"
        assert all(call[3] == 1 for call in fs.calls)
    else:
        assert not fs.calls and not fs.embedder.calls


@pytest.mark.parametrize("limit", [-1, 0, 201])
async def test_context_without_quotas_still_rejects_out_of_range_limit(api, limit):
    fs, service, sdk = api
    with pytest.raises(InvalidArgumentError, match="Context limit must be between 1 and 200"):
        await sdk.search("q", mode="context", limit=limit)
    with pytest.raises(InvalidArgumentError, match="Context limit must be between 1 and 200"):
        await service.search.search("q", ctx=context(), mode="context", limit=limit)
    assert not fs.calls and not fs.embedder.calls


async def test_body_prefetch_is_bounded_by_final_output_count(api):
    fs, _, sdk = api
    result = await sdk.search(
        "q",
        mode="context",
        target_uri="viking://user/alice/memories/events",
        limit=2,
        detail="full",
    )
    assert len(result["entries"]) == 2 and len(fs.reads) == 2


async def test_server_rejects_expansion_without_loading_session(api):
    fs, service, sdk = api
    response = await sdk._http.post(
        "http://fixture/api/v1/search/search",
        json={"query": "q", "mode": "context", "session_id": "ignored", "query_expansion": "auto"},
    )
    assert response.status_code == 422
    assert not fs.calls
    service.sessions.session.assert_not_called()


async def test_quotas_intersect_explicit_target_and_preserve_filter(api):
    fs, _, sdk = api
    result = await sdk.search(
        "q",
        mode="context",
        target_uri="viking://user/alice/memories/events",
        quotas={"events": 1, "preferences": 1},
        filter={"op": "contains", "field": "uri", "substring": "0.md"},
    )
    assert len(fs.calls) == 1 and fs.embedder.calls == 1
    assert len(result["entries"]) == 1 and "/events/0.md" in result["entries"][0]["uri"]


async def test_independent_group_thresholds_and_hard_limits(api):
    fs, _, sdk = api
    policy = RecallPolicy(
        experience_enabled=True,
        recall_limit=1,
        experience_limit=1,
        score_threshold=0.9,
        experience_score_threshold=0.7,
    )
    result = await sdk.search("q", mode="context", recall_policy=policy.model_dump())
    assert len(fs.calls) == 2 and fs.embedder.calls == 1
    assert len(result["entries"]) == 1 and result["entries"][0]["category"] == "events"
    policy.experience_score_threshold = 0.4
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"events": 5, "experiences": 5},
        recall_policy=policy.model_dump(),
    )
    # Explicit quotas supersede the two legacy count caps, but keep thresholds.
    assert len([e for e in result["entries"] if e["category"] == "events"]) == 5
    assert len([e for e in result["entries"] if e["category"] == "experiences"]) == 2


@pytest.mark.parametrize("budget", [1, 40, 120, 400, 1600])
async def test_tokens_and_character_limits_include_rendered_envelope(api, budget):
    fs, _, sdk = api
    fs.body = "long body " * 2000
    policy = RecallPolicy(experience_enabled=True, max_injected_chars=600, experience_max_chars=240)
    result = await sdk.search(
        "q", mode="context", recall_policy=policy.model_dump(), detail="full", max_tokens=budget
    )
    assert estimate_text_tokens(result["rendered"]) <= budget
    assert len(result["rendered"]) <= 600
    experiences = [e for e in result["entries"] if e["category"] == "experiences"]
    assert len(experiences) <= 1
    assert result["stats"]["used_chars"] == len(result["rendered"])


async def test_zero_quota_and_unused_slots(api):
    fs, _, sdk = api
    fs.categories["entities"] = 0
    result = await sdk.search(
        "q", mode="context", quotas={"events": 1, "entities": 3, "experiences": 0}
    )
    assert len(fs.calls) == 4 and len(result["entries"]) == 4
    assert all(e["category"] == "events" for e in result["entries"])


async def test_excluding_directory_excludes_entire_subtree_and_render_can_be_off(api):
    _, _, sdk = api
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"events": 2, "preferences": 1},
        exclude_uris=["viking://user/alice/memories/events"],
        render=False,
    )
    assert result["rendered"] == "" and result["entries"]
    assert all(e["category"] != "events" for e in result["entries"])


async def test_concurrent_users_and_requests_do_not_share_cache(api):
    fs, _, sdk = api

    async def request(user):
        return await sdk.search(
            "same",
            mode="context",
            quotas={"events": 1, "preferences": 1},
            identity=OVIdentity("tenant", user, "assistant"),
        )

    alice, bob = await asyncio.gather(request("alice"), request("bob"))
    assert fs.embedder.calls == 2
    assert all("/alice/" in e["uri"] for e in alice["entries"])
    assert all("/bob/" in e["uri"] for e in bob["entries"])


@pytest.mark.parametrize(
    "options",
    [
        {"quotas": {"events": -1}},
        {"quotas": {"events": True}},
        {"quotas": {"made_up": 1}},
        {"quotas": {"memories": 1}},
        {"query_expansion": "auto"},
        {"rewrite": True},
        {"detail": {"made_up": "full"}},
        {"max_tokens": 0},
        {"recall_policy": {"score_threshold": float("inf")}},
        {"recall_policy": {"extra": 1}},
    ],
)
async def test_invalid_options_are_rejected_before_retrieval(api, options):
    fs, _, sdk = api
    with pytest.raises(ValueError):
        await sdk.search("q", mode="context", **options)
    assert not fs.calls


async def test_legacy_list_still_dispatches_to_old_search(api):
    fs, service, sdk = api
    fs.search.side_effect = None
    fs.search.return_value = FindResult(memories=[], resources=[], skills=[])
    result = await sdk.search("q")
    assert isinstance(result, FindResult)
    fs.search.assert_awaited_once()
    assert not fs.calls
    service.sessions.session.assert_not_called()


@pytest.mark.parametrize("detail", ["full", "overview"])
@pytest.mark.parametrize("summary_detail", ["abstract", "overview"])
@pytest.mark.parametrize("read_failure", ["error", "empty"])
async def test_read_failure_falls_back_to_actual_summary_detail(
    api, detail, summary_detail, read_failure
):
    fs, _, sdk = api
    fs.categories = {}
    fs.extra_items = [
        MatchedContext(
            uri="viking://user/alice/memories/experiences/fallback.md",
            context_type=ContextType.MEMORY,
            category="experiences",
            score=0.9,
            **{summary_detail: "Fallback summary."},
        )
    ]
    fs.body = ""
    if read_failure == "error":
        fs.read_error = RuntimeError("unavailable")
    result = await sdk.search("q", mode="context", quotas={"experiences": 1}, detail=detail)
    assert len(result["entries"]) == len(fs.reads) == 1
    assert result["entries"][0]["text"] == "Fallback summary."
    assert result["entries"][0]["detail"] == summary_detail
    assert f'detail="{summary_detail}"' in result["rendered"]
    assert bool(result["stats"]["read_errors"]) == (read_failure == "error")
    exact_budget = estimate_text_tokens(result["rendered"])
    assert result["stats"]["used_tokens"] == exact_budget
    assert result["stats"]["used_chars"] == len(result["rendered"])

    fits = await sdk.search(
        "q", mode="context", quotas={"experiences": 1}, detail=detail, max_tokens=exact_budget
    )
    assert fits["entries"] == result["entries"]
    assert fits["rendered"] == result["rendered"]
    too_small = await sdk.search(
        "q", mode="context", quotas={"experiences": 1}, detail=detail, max_tokens=exact_budget - 1
    )
    assert too_small["entries"] == [] and too_small["rendered"] == ""


@pytest.mark.parametrize(
    "level,detail,abstract,overview,body,expected_detail,expected_text",
    [
        (2, "full", "summary", "", "full body", "full", "full body"),
        (0, "full", "summary", "", "directory overview", "overview", "directory overview"),
        (1, "full", "summary", "", "directory overview", "overview", "directory overview"),
        (2, "overview", "summary", "", "# Summary\n\npreview", "overview", "preview"),
        (2, "overview", "summary", "", "# Summary\n\n## Details", "abstract", "summary"),
        (2, "abstract", "", "stored overview", "", "overview", "stored overview"),
        (2, "abstract", "", "", "full body", "full", "full body"),
        (0, "abstract", "", "", "directory overview", "overview", "directory overview"),
    ],
)
async def test_detail_reflects_selected_content_source(
    api, level, detail, abstract, overview, body, expected_detail, expected_text
):
    fs, _, sdk = api
    fs.categories = {}
    uri = "viking://user/alice/memories/events/source"
    if level == 2:
        uri += ".md"
    fs.extra_items = [
        MatchedContext(
            uri=uri,
            context_type=ContextType.MEMORY,
            category="events",
            score=0.9,
            level=level,
            abstract=abstract,
            overview=overview,
        )
    ]
    fs.body = body
    result = await sdk.search("q", mode="context", quotas={"events": 1}, detail=detail)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["detail"] == expected_detail
    assert result["entries"][0]["text"] == expected_text
    assert f'detail="{expected_detail}"' in result["rendered"]
    assert result["stats"]["used_tokens"] == estimate_text_tokens(result["rendered"])


async def test_cache_separates_input_models_and_resource_writes():
    one, two = Embedder(), Embedder()
    async with query_embed_cache_scope():
        await asyncio.gather(*(embed_compat(one, "same", is_query=True) for _ in range(12)))
        await embed_compat(one, "other", is_query=True)
        await embed_compat(two, "same", is_query=True)
        await embed_compat(one, "same", is_query=False)
        await embed_compat(one, "same", is_query=False)
    assert one.calls == 4 and two.calls == 1
    async with query_embed_cache_scope():
        await embed_compat(one, "same", is_query=True)
    assert one.calls == 5


async def test_cache_failure_can_retry_and_cancelled_waiter_does_not_cancel_others():
    model = Embedder()
    async with query_embed_cache_scope():
        model.fail = True
        with pytest.raises(RuntimeError):
            await embed_compat(model, "q", is_query=True)
        model.fail = False
        model.release.clear()
        one = asyncio.create_task(embed_compat(model, "q", is_query=True))
        two = asyncio.create_task(embed_compat(model, "q", is_query=True))
        for _ in range(5):
            await asyncio.sleep(0)
        one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await one
        model.release.set()
        await two
    assert model.calls == 2 and model.cancelled == 0


def test_sync_sdk_forwards_context_options(monkeypatch):
    # Test forwarding in the caller thread, without depending on the process-wide
    # event loop owned by unrelated integration fixtures.
    import openviking_cli.client.sync_http as module

    monkeypatch.setattr(module, "run_async", asyncio.run)
    expected = {"entries": [], "rendered": "", "stats": {}}
    asynchronous = SimpleNamespace(search=AsyncMock(return_value=expected))
    sdk = SyncHTTPClient.__new__(SyncHTTPClient)
    sdk._async_client = asynchronous
    assert sdk.search("q", mode="context", quotas={"events": 1}, max_tokens=500) is expected
    assert asynchronous.search.call_args.kwargs["mode"] == "context"
    assert asynchronous.search.call_args.kwargs["max_tokens"] == 500


async def test_resource_scopes_and_type_filter_are_applied_before_top_k(api):
    fs, _, sdk = api
    fs.categories = {"events": 50}
    expected = {
        "viking://resources/shared.md",
        "viking://user/alice/resources/experiences/guide.md",
        "viking://user/alice/peers/assistant/resources/guide.md",
    }
    fs.extra_items = [hit(uri, ContextType.RESOURCE) for uri in expected]
    fs.extra_items += [
        hit("viking://user/bob/resources/private.md", ContextType.RESOURCE, 0.99),
        hit("viking://user/alice/peers/other/resources/private.md", ContextType.RESOURCE, 0.99),
    ]
    result = await sdk.search("q", mode="context", quotas={"resources": 3})
    assert {e["uri"] for e in result["entries"]} == expected
    assert {e["category"] for e in result["entries"]} == {"resources"}


async def test_reader_resource_subscope_intersects_filter_and_preserves_type(api):
    fs, _, sdk = api
    good = "viking://user/alice/resources/project/memories/events/keep.md"
    fs.extra_items = [
        hit(good, ContextType.RESOURCE),
        hit("viking://user/alice/resources/other/keep.md", ContextType.RESOURCE, 0.99),
    ]
    policy = RecallPolicy(target_uris=("viking://user/alice/resources/project",))
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"resources": 1},
        filter={"op": "contains", "field": "uri", "substring": "keep.md"},
        recall_policy=policy.model_dump(),
    )
    assert [e["uri"] for e in result["entries"]] == [good]
    assert result["entries"][0]["category"] == "resources"


async def test_quota_width_supports_backfill_and_ignores_limit(api):
    fs, _, sdk = api
    fs.categories = {"events": 10, "entities": 0, "preferences": 10}
    policy = RecallPolicy(candidate_limit=1, recall_limit=1)
    result = await sdk.search(
        "q",
        mode="context",
        limit=1,
        quotas={"events": 2, "entities": 2, "preferences": 2},
        recall_policy=policy.model_dump(),
    )
    assert len(result["entries"]) == 6
    assert {e["category"] for e in result["entries"]} == {"events", "preferences"}
    assert all(c[3] >= 6 for c in fs.calls)
    assert "limit" in result["stats"]["ignored"]
    fs.calls.clear()
    result = await sdk.search("q", mode="context", limit=6, quotas={"entities": 2})
    assert not result["entries"]
    assert all(c[2].endswith("/memories/entities") for c in fs.calls)


async def test_category_directory_summary_is_not_a_preset_memory_root(api):
    fs, _, sdk = api
    fs.categories = {}
    uri = "viking://user/alice/memories/preferences/topic/.abstract.md"
    fs.extra_items = [hit(uri, level=0)]
    result = await sdk.search("q", mode="context", quotas={"preferences": 1}, detail="abstract")
    assert [e["uri"] for e in result["entries"]] == [uri]


async def test_quota_buckets_rank_and_deduplicate_independently(api):
    fs, _, sdk = api
    fs.categories = {"events": 1}
    fs.extra_items = [
        hit("viking://user/alice/memories/preferences/topic/.abstract.md", level=0),
        hit("viking://user/alice/resources/guide.md", ContextType.RESOURCE),
        hit("viking://user/alice/skills/deploy/SKILL.md", ContextType.SKILL),
    ]
    # Real stored hits commonly have no `category` metadata. The category must
    # come from the owning bucket, even if summaries happen to be identical.
    for item in fs.extra_items:
        item.abstract = "same summary"
    policy = RecallPolicy(prefer_leaf=True)
    result = await sdk.search(
        "q",
        mode="context",
        quotas=dict.fromkeys(("events", "preferences", "resources", "skills"), 1),
        detail="abstract",
        recall_policy=policy.model_dump(),
    )
    assert len(result["entries"]) == 4
    assert {e["category"] for e in result["entries"]} == {
        "events",
        "preferences",
        "resources",
        "skills",
    }


async def test_default_policy_uses_original_query_without_session_or_body_reads(api):
    fs, service, sdk = api
    result = await sdk.search(
        "  original query  ",
        mode="context",
        session_id="ignored-session",
        recall_policy=RecallPolicy().model_dump(),
    )
    assert len(result["entries"]) == 5
    assert len(fs.calls) == fs.embedder.calls == 1
    assert fs.calls[0][0] == "  original query  "
    assert fs.calls[0][2] == "viking://user/memories" and fs.calls[0][3] == 20
    assert not fs.reads and len(result["rendered"]) <= 4000
    service.sessions.session.assert_not_called()
    fs.search.assert_not_called()
    IntentAnalyzer.analyze.assert_not_called()


async def test_four_buckets_share_embedding_and_read_complete_experience(api):
    fs, service, sdk = api
    result = await sdk.search(
        "hello",
        mode="context",
        session_id="ignored",
        quotas={"events": 2, "entities": 1, "preferences": 1, "experiences": 1},
        recall_policy=RecallPolicy(experience_enabled=True).model_dump(),
    )
    assert len(result["entries"]) == 5
    assert len(fs.calls) == 8 and fs.embedder.calls == 1
    assert result["stats"]["embedding_computations"] == 1
    assert result["stats"]["embedding_cache_hits"] == 7
    assert "recorded request ID" in result["rendered"]
    assert all("/experiences/" in uri for uri, _ in fs.reads)
    assert all(call[0] == "hello" for call in fs.calls)
    service.sessions.session.assert_not_called()


async def test_multiple_policy_scopes_preserve_custom_memories_and_share_embedding(api):
    fs, _, sdk = api
    policy = RecallPolicy(target_uris=("viking://user/alice/memories/cases",))
    result = await sdk.search("q", mode="context", recall_policy=policy.model_dump())
    assert len(result["entries"]) == 1
    assert result["entries"][0]["uri"].endswith("/cases/0.md")
    assert result["entries"][0]["category"] == "memories"
    policy.target_uris += ("viking://user/alice/memories/preferences",)
    fs.calls.clear()
    fs.embedder.calls = 0
    result = await sdk.search("q", mode="context", recall_policy=policy.model_dump())
    assert len(result["entries"]) == 3
    assert len(fs.calls) == 2 and fs.embedder.calls == 1


async def test_context_scope_failure_preserves_usable_results_and_diagnostics(api, monkeypatch):
    fs, _, sdk = api
    original = fs.find

    async def find(query, *, target_uri, **kwargs):
        if target_uri.endswith("/unavailable"):
            raise RuntimeError("offline scope")
        return await original(query, target_uri=target_uri, **kwargs)

    monkeypatch.setattr(fs, "find", find)
    policy = RecallPolicy(
        target_uris=(
            "viking://user/alice/memories/unavailable",
            "viking://user/alice/memories/cases",
        )
    )
    result = await sdk.search("q", mode="context", recall_policy=policy.model_dump())
    assert len(result["entries"]) == 1
    assert result["stats"]["retrieval_errors"] == ["RuntimeError"]


@pytest.mark.parametrize("operation", ["find", "read"])
async def test_auth_errors_propagate_and_optional_failures_are_diagnosed(api, operation):
    fs, _, sdk = api
    attribute = "error" if operation == "find" else "read_error"
    setattr(fs, attribute, PermissionDeniedError("forbidden"))
    with pytest.raises(PermissionDeniedError):
        await sdk.search("q", mode="context", quotas={"experiences": 1}, detail="full")
    setattr(fs, attribute, RuntimeError("unavailable"))
    result = await sdk.search("q", mode="context", quotas={"experiences": 1}, detail="full")
    diagnostic = "retrieval_errors" if operation == "find" else "read_errors"
    assert result["stats"][diagnostic]
    assert bool(result["entries"]) == (operation == "read")


async def test_cancelled_search_cleans_up_shared_embedding(api):
    fs, _, sdk = api
    fs.embedder.release.clear()
    pending = asyncio.create_task(
        sdk.search("q", mode="context", quotas={"events": 1, "preferences": 1})
    )
    try:
        await asyncio.wait_for(fs.embedder.started.wait(), timeout=1)
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert fs.embedder.calls == fs.embedder.cancelled == 1
    fs.embedder.release.set()
    result = await sdk.search("q", mode="context", quotas={"events": 1})
    assert result["entries"] and fs.embedder.calls == 2


async def test_memories_quota_rejected_at_sdk_service_and_http(api):
    fs, service, sdk = api
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


async def test_all_six_buckets_use_local_roots_and_preserve_type_isolation(api):
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
    result = await sdk.search(
        "q", mode="context", quotas=quotas, recall_policy=RecallPolicy(recall_limit=1).model_dump()
    )
    assert len(result["entries"]) == 6
    assert {entry["category"] for entry in result["entries"]} == set(quotas)
    assert "usage.md" not in result["rendered"] and "/bob/" not in result["rendered"]
    targets = {c[2] for c in fs.calls}
    assert "viking://user/alice/skills" in targets
    assert "viking://agent/skills" not in targets
    assert fs.embedder.calls == 1


async def test_quota_filter_precedes_candidate_cut_and_ignores_nested_names(api):
    fs, _, sdk = api
    fs.categories = {"events": 50, "preferences": 1}
    fs.extra_items = [
        hit(f"viking://user/alice/memories/custom/preferences/{i}.md", score=0.99)
        for i in range(50)
    ]
    result = await sdk.search("q", mode="context", quotas={"preferences": 1})
    assert [e["uri"] for e in result["entries"]] == [
        "viking://user/alice/memories/preferences/0.md"
    ]
    assert all(c[2].endswith("/memories/preferences") for c in fs.calls)


@pytest.mark.parametrize(
    "target", ["viking://user/alice/memories", "viking://user/alice/memories/cases"]
)
async def test_policy_memory_scope_never_expands_to_resources_or_skills(api, target):
    fs, _, sdk = api
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"resources": 1, "skills": 1},
        recall_policy=RecallPolicy(target_uris=(target,)).model_dump(),
    )
    assert not result["entries"] and not fs.calls


async def test_quotas_override_legacy_switches_and_counts_but_preserve_budgets(api):
    _, _, sdk = api
    policy = RecallPolicy(
        enabled=False,
        experience_enabled=False,
        recall_limit=0,
        experience_limit=0,
        candidate_limit=1,
        experience_candidate_limit=1,
    )
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"events": 6, "experiences": 2},
        max_tokens=8000,
        recall_policy=policy.model_dump(),
    )
    assert len(result["entries"]) == 8
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"events": 6},
        recall_policy=RecallPolicy(max_injected_chars=1).model_dump(),
    )
    assert not result["entries"]


async def test_flat_policy_keeps_other_memories_but_excludes_profile_and_preset_root(api):
    fs, _, sdk = api
    fs.categories = {"trajectories": 1, "tools": 1, "skills": 1, "custom": 1}
    root = "viking://user/alice/memories"
    fs.extra_items = [hit(f"{root}/{name}.md") for name in ("profile", "identity", "soul")]
    fs.extra_items += [
        hit(f"{root}/{name}.md", level=level)
        for name, level in ((".abstract", 0), (".overview", 1))
    ]
    fs.extra_items += [hit(f"{root}/custom/experiences/nested.md")]
    result = await sdk.search(
        "q",
        mode="context",
        detail={"memories": "abstract"},
        recall_policy=RecallPolicy(recall_limit=20).model_dump(),
    )
    assert len(result["entries"]) == 7
    assert "profile.md" not in result["rendered"]
    assert "/memories/.abstract.md" not in result["rendered"]
    assert "/memories/.overview.md" not in result["rendered"]
    assert {e["category"] for e in result["entries"]} == {"memories"}


async def test_explicit_empty_policy_scope_disables_quota_search(api):
    fs, _, sdk = api
    result = await sdk.search(
        "q",
        mode="context",
        quotas={"events": 1},
        recall_policy=RecallPolicy(target_uris=()).model_dump(),
    )
    assert not result["entries"] and not fs.calls


async def test_async_facade_preserves_context_options_and_structured_result(api, monkeypatch):
    from openviking.async_client import AsyncOpenViking

    fs, service, _ = api
    local = LocalClient.__new__(LocalClient)
    local._service, local._ctx = service, context()
    client = object.__new__(AsyncOpenViking)
    client._client = local
    monkeypatch.setattr(client, "_ensure_initialized", AsyncMock())
    result = await client.search("q", mode="context", quotas={"preferences": 1}, max_tokens=500)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["category"] == "preferences"
    assert result["stats"]["used_tokens"] <= 500 and fs.embedder.calls == 1


def test_sync_facade_forwards_context_options(monkeypatch):
    import openviking.sync_client as module

    expected = {"entries": [], "rendered": "", "stats": {}}
    asynchronous = SimpleNamespace(search=AsyncMock(return_value=expected))
    client = object.__new__(module.SyncOpenViking)
    client._async_client = asynchronous
    monkeypatch.setattr(module, "run_async", asyncio.run)
    assert client.search("q", mode="context", quotas={"skills": 1}, max_tokens=500) is expected
    assert asynchronous.search.call_args.kwargs["quotas"] == {"skills": 1}
    assert asynchronous.search.call_args.kwargs["max_tokens"] == 500


async def test_mcp_context_search_uses_service_and_never_loads_session(api, monkeypatch):
    from openviking.server import mcp_endpoint as endpoint

    fs, service, _ = api
    monkeypatch.setattr(endpoint, "get_service", lambda: service)
    token = endpoint._mcp_ctx.set(context())
    try:
        # Exercise the registered tool as well as its JSON-schema argument validation.
        content = await endpoint.mcp.call_tool(
            "search",
            {
                "query": "q",
                "mode": "context",
                "session_id": "ignored",
                "quotas": {"experiences": 1},
                "detail": {"experiences": "full"},
                "max_tokens": 500,
            },
        )
    finally:
        endpoint._mcp_ctx.reset(token)
    # MCP releases supporting structured output return (content, structured).
    if isinstance(content, tuple):
        content, structured = content
        assert structured["result"] == content[0].text
    assert len(content) == 1 and "recorded request ID" in content[0].text
    assert estimate_text_tokens(content[0].text) <= 500
    assert fs.embedder.calls == 1
    service.sessions.session.assert_not_called()
    IntentAnalyzer.analyze.assert_not_called()


async def test_mcp_legacy_search_still_loads_session(api, monkeypatch):
    from openviking.server import mcp_endpoint as endpoint

    fs, service, _ = api
    session = SimpleNamespace(load=AsyncMock(), get_context_for_search=AsyncMock(return_value=None))
    service.sessions.session = lambda ctx, session_id: session
    fs.search.side_effect = None
    fs.search.return_value = FindResult(memories=[], resources=[], skills=[])
    monkeypatch.setattr(endpoint, "get_service", lambda: service)
    token = endpoint._mcp_ctx.set(context())
    try:
        assert await endpoint.search("q", session_id="session-1") == "No matching context found."
    finally:
        endpoint._mcp_ctx.reset(token)
    session.load.assert_awaited_once()
    session.get_context_for_search.assert_awaited_once_with("q")
    fs.search.assert_awaited_once()
    assert not fs.calls


@pytest.mark.parametrize("options", [{"quotas": {"events": 1}}, {"max_tokens": 500}])
async def test_list_mode_rejects_context_only_options_without_retrieving(api, options):
    fs, _, sdk = api
    with pytest.raises(ValueError, match="Context options require"):
        await sdk.search("q", **options)
    assert not fs.calls
    fs.search.assert_not_called()


async def test_find_multiple_paths_shares_one_embedding_and_combined_limit(api):
    fs, _, sdk = api
    roots = [f"viking://user/alice/memories/{category}" for category in ("events", "preferences")]
    result = await sdk.find(
        "原始问题",
        target_uri=roots,
        limit=2,
        identity=OVIdentity("tenant", "alice", "assistant"),
    )
    assert len(fs.calls) == 1 and fs.embedder.calls == 1
    assert fs.calls[0][2] == roots
    assert len(result.memories) == 2
    assert all(item.category == "events" for item in result.memories)


async def test_find_multiple_paths_propagates_authentication_failure(api):
    fs, _, sdk = api
    fs.error = PermissionDeniedError("blocked")
    with pytest.raises(PermissionDeniedError):
        await sdk.find(
            "q",
            target_uri=["viking://user/memories", "viking://user/peers/assistant/memories"],
            identity=OVIdentity("tenant", "alice", "assistant"),
        )
