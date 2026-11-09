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
