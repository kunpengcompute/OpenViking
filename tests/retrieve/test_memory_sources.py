from unittest.mock import patch

import pytest

from openviking.retrieve.evidence_retriever import EvidenceIndex
from openviking.retrieve.memory_sources import MemorySources


@pytest.fixture
def people(monkeypatch):
    def features(text):
        return [], set(text.lower().replace("?", "").split())

    for module in ("evidence_retriever", "memory_sources"):
        monkeypatch.setattr(f"openviking.retrieve.{module}.extract_keys", features)
    index = EvidenceIndex()
    facts = []
    for i, (speaker, role, subject, text) in enumerate(
        [
            ("Dana", "user", "Dana", "Robin, I bought pottery."),
            ("Robin", "assistant", "Robin", "I bought figurines yesterday."),
            ("assistant", "assistant", "Robin", "Robin bought invented figurines."),
            ("Dana", "user", "Robin", "Robin bought shoes."),
        ]
    ):
        source_id = f"m{i}"
        index.append_chunk(
            "chat",
            [{"id": source_id, "speaker": speaker, "role": role, "text": text}],
        )
        facts.append(
            {
                "id": f"f{i}",
                "subject": subject,
                "predicate": "buy",
                "object": text,
                "speaker": role,
                "assertion_type": "fact",
                "source_turn_ids": [source_id],
            }
        )
    return index, MemorySources(facts=facts)


def test_named_person_is_independent_of_transport_role(people):
    index, hints = people
    with patch("openviking.retrieve.memory_sources.verify_fact", return_value=True):
        hits = hints.candidates("What has Robin bought?", index, 10)
    assert {h["source_id"] for h in hits} == {"m1", "m3"}
    own = next(f for f in hints.selected_facts if f["id"] == "f1")
    assert own["source_speaker"] == "Robin"
    assert own["source_role"] == "assistant"


def test_first_person_question_keeps_user_source_boundary(people):
    index, hints = people
    with patch("openviking.retrieve.memory_sources.verify_fact", return_value=True):
        hits = hints.candidates("What have I bought?", index, 10)
    assert {h["source_id"] for h in hits} == {"m0", "m3"}


def test_named_sources_still_require_verification(people):
    index, hints = people
    with patch("openviking.retrieve.memory_sources.verify_fact", return_value=False):
        assert hints.candidates("What has Robin bought?", index, 10) == []
    assert hints.selected_facts == []


def test_swapping_named_participants_roles_does_not_change_results(people):
    index, hints = people
    with patch("openviking.retrieve.memory_sources.verify_fact", return_value=True):
        before = hints.candidates("What has Robin bought?", index, 10)
        for turn in index._turn_by_id.values():
            if turn["speaker"] in {"Dana", "Robin"}:
                turn["role"] = "assistant" if turn["role"] == "user" else "user"
        after = hints.candidates("What has Robin bought?", index, 10)
    assert {h["source_id"] for h in before} == {h["source_id"] for h in after}


def test_assistant_suggestions_still_require_curated_metadata(people):
    index, hints = people
    with patch("openviking.retrieve.memory_sources.verify_fact", return_value=True):
        assert hints.candidates("What did you say Robin bought?", index, 10) == []
        hints.metadata_by_id["f2"] = {}
        hits = hints.candidates("What did you say Robin bought?", index, 10)
    assert [h["source_id"] for h in hits] == ["m2"]


def test_speaker_name_boundaries(people):
    index, _ = people
    from openviking.retrieve.evidence_features import query_speakers

    assert query_speakers("What did Robinson buy?", index._turn_by_id.values()) == set()
    assert query_speakers("Robin's purchases", index._turn_by_id.values()) == {"robin"}
