from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from openviking.retrieve.evidence_fusion import (
    AuxHit,
    EvidenceIntent,
    classify_evidence_intent,
    fuse_auxiliary,
    time_candidates,
)
from openviking.retrieve.evidence_retriever import EvidenceIndex
from openviking.retrieve.evidence_strategy import (
    build_evidence_groups,
    fuse_candidates,
    retrieve_coverage,
    select_evidence_groups,
)


@pytest.fixture
def index(monkeypatch):
    def features(text):
        return [], set(text.lower().split())

    for module in ("evidence_retriever", "evidence_ranking", "evidence_strategy"):
        monkeypatch.setattr(f"openviking.retrieve.{module}.extract_keys", features)
    result = EvidenceIndex()
    result.append_chunk(
        "chat",
        [
            {"id": "q", "text": "Where are my keys?", "speaker": "alice", "date": "2026-11-09"},
            {
                "id": "a",
                "text": "The keys are under the bed.",
                "speaker": "bob",
                "date": "2026-11-09",
            },
            {"id": "other", "text": "Unrelated content.", "speaker": "alice", "date": "2026-11-09"},
        ],
        user_space="viking://user/alice",
        archive_uri="viking://user/alice/sessions/chat/history/archive_001",
    )
    return result


def test_question_answer_group_is_never_partially_selected(index):
    hit = dict(index.get_turns(["chat:1"])[0], turn_id="chat:1", score=1.0)
    groups = build_evidence_groups(index, [hit])
    assert groups[0]["turn_ids"] == ["chat:0", "chat:1"]
    assert select_evidence_groups(groups, top_k=1) == []
    assert len(select_evidence_groups(groups, top_k=2)) == 2


@pytest.mark.asyncio
async def test_coverage_budget_and_lexical_fallback(index):
    embed = AsyncMock(side_effect=OSError("provider unavailable"))
    evidence, diagnostics = await retrieve_coverage(
        index,
        "Where are my keys?",
        embed=embed,
        top_k=2,
        max_supplementary_queries=0,
    )
    assert len(evidence) == 2
    assert diagnostics["retrieval_calls"] == 1
    assert diagnostics["errors"][0]["stage"] == "embedding"
    assert all("emb" not in turn for turn in evidence)


@pytest.mark.asyncio
async def test_supplementary_queries_have_hard_budget(index):
    _, diagnostics = await retrieve_coverage(
        index,
        "What do Alice and Bob own?",
        embed=AsyncMock(return_value=None),
        top_k=2,
        max_supplementary_queries=1,
    )
    assert diagnostics["retrieval_calls"] == 2
    assert "target_coverage" in diagnostics


def test_auxiliary_fusion_routes_by_intent():
    content = [{"turn_id": "a", "score": 0.5}]
    graph = [AuxHit("b", "graph", 0.9, ("w",))]
    time = [AuxHit("c", "time", 1.0, ("w",))]
    hits, diag = fuse_auxiliary(
        content, graph, time, EvidenceIntent("ENTITY_RELATIONS", 0.3, "graph"), candidate_limit=2
    )
    assert [h["turn_id"] for h in hits] == ["b", "a"]
    assert diag["selected_channels"] == ["content", "graph"]
    assert classify_evidence_intent("Decisions in November 2026").channel == "time"
    unchanged, diag = fuse_auxiliary(
        content, [], [], EvidenceIntent("OTHER", 1.0, None), candidate_limit=2
    )
    assert unchanged == content
    assert diag["rho"] == 1.0


def test_time_candidates_use_recorded_window_time(index):
    index.graph["windows"]["w"] = {
        "turn_ids": ["chat:0"],
        "when_start": "2026-11-09T00:00:00+00:00",
        "when_end": "2026-11-09T00:00:00+00:00",
    }
    hits = time_candidates(
        index,
        "Decisions in November 2026",
        now=datetime(2026, 11, 10, tzinfo=timezone.utc),
        content_turn_ids=set(),
    )

    assert hits[0].turn_id == "chat:0"
    index.graph_unavailable = True
    assert (
        time_candidates(
            index,
            "Decisions in November 2026",
            now=datetime.now(timezone.utc),
            content_turn_ids=set(),
        )
        == []
    )


def test_repeated_queries_and_memory_hints_do_not_multiply_source_votes():
    direct = {"turn_id": "direct", "score": 0.032}
    repeated = {"turn_id": "other", "score": 0.02}
    hits = fuse_candidates([[direct, repeated], [repeated]], [repeated, direct])
    assert [h["turn_id"] for h in hits] == ["direct", "other"]
    assert hits[0]["score"] == 0.032
    assert hits[1]["score"] == 0.02


def test_memory_hints_can_recall_missing_sources_without_borrowing_raw_scores():
    hits = fuse_candidates(
        [[{"turn_id": "direct", "score": 0.03}]],
        [{"turn_id": "new", "score": 1000}],
    )
    assert [h["turn_id"] for h in hits] == ["direct", "new"]
    assert hits[1]["query_branches"] == []


def test_relevant_sources_from_same_episode_are_not_penalized(index):
    rows = index.get_turns(["chat:0", "chat:1", "chat:2"])
    groups = [
        {
            "anchor_id": str(i),
            "turn_ids": [str(i)],
            "turns": [dict(t, episode=ep)],
            "score": score,
            "query_branches": [0],
        }
        for i, (t, ep, score) in enumerate(
            zip(rows, ["same", "same", "other"], [2, 1.1, 1], strict=True)
        )
    ]
    assert [t["episode"] for t in select_evidence_groups(groups, top_k=2)] == ["same", "same"]


def append_story(index, messages, session="story"):
    index.append_chunk(
        session,
        [
            {"id": f"{session}-{i}", "text": text, "speaker": speaker}
            for i, (speaker, text) in enumerate(messages)
        ],
    )
    return dict(index.get_turns([f"{session}:0"])[0], turn_id=f"{session}:0", score=1.0)


def test_counter_question_keeps_both_answers_and_stops_at_topic_change(index):
    hit = append_story(
        index,
        [
            ("Milo", "How long have you worked with clay?"),
            ("Aria", "Since college. How long have you worked with clay?"),
            ("Milo", "Nine years."),
            ("Aria", "Do you own pets?"),
            ("Milo", "I own cats."),
        ],
    )
    groups = build_evidence_groups(index, [hit])
    assert groups[0]["turn_ids"] == ["story:0", "story:1", "story:2"]
    assert len(select_evidence_groups(groups, top_k=3)) == 3
    assert select_evidence_groups(groups, top_k=2) == []


@pytest.mark.parametrize("question", ["What did you learn from it?", "For walking or running?"])
def test_followup_question_and_answer_are_kept_together(index, question):
    hit = append_story(
        index,
        [
            ("Milo", "Here is the new item I bought."),
            ("Aria", question),
            ("Milo", "It is for exercise."),
            ("Aria", "Any travel plans?"),
        ],
    )
    assert build_evidence_groups(index, [hit])[0]["turn_ids"] == ["story:0", "story:1", "story:2"]


def test_anaphoric_answer_includes_the_antecedent(index):
    append_story(
        index,
        [
            ("Milo", "Here is my new book."),
            ("Aria", "What did you learn from it?"),
            ("Milo", "It taught me patience."),
        ],
    )
    hit = dict(index.get_turns(["story:2"])[0], turn_id="story:2", score=1.0)
    assert build_evidence_groups(index, [hit])[0]["turn_ids"] == ["story:0", "story:1", "story:2"]


def test_followup_expansion_does_not_cross_session_boundaries(index):
    hit = append_story(index, [("Milo", "Here is my new book.")])
    append_story(index, [("Aria", "What did you learn from it?")], session="unrelated")
    assert build_evidence_groups(index, [hit])[0]["turn_ids"] == ["story:0"]


def test_unanswered_followup_does_not_displace_its_antecedent(index):
    hit = append_story(
        index,
        [
            ("Milo", "Here is my new book."),
            ("Aria", "What did you learn from it?"),
        ],
    )
    assert build_evidence_groups(index, [hit])[0]["turn_ids"] == ["story:0"]


@pytest.mark.asyncio
async def test_returned_facts_are_supported_by_selected_sources(index):
    from types import SimpleNamespace

    keep = {"source_turn_ids": ["a"]}
    drop = {"source_turn_ids": ["missing"]}
    hints = SimpleNamespace(
        candidates=lambda *args: [], diagnostics={}, events=[], selected_facts=[keep, drop]
    )
    evidence, _ = await retrieve_coverage(
        index,
        "Where are my keys?",
        embed=AsyncMock(return_value=None),
        top_k=2,
        max_supplementary_queries=0,
        memory_sources=hints,
    )
    assert {t["source_id"] for t in evidence} == {"q", "a"}
    assert hints.selected_facts == [keep]
