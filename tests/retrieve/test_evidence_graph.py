from openviking.retrieve.evidence_retriever import EvidenceIndex

ROOT = "viking://user/alice"
SESSION = ROOT + "/sessions/chat"


def add(index, number, session="chat"):
    uri = ROOT + f"/sessions/{session}/history/archive_{number:03d}"
    index.append_chunk(
        session,
        [
            {
                "id": f"{session}-{number}",
                "role": "user",
                "speaker": "user",
                "text": "We updated parser.py.",
                "date": f"2026-11-{number:02d}T10:00:00Z",
            }
        ],
        user_space=ROOT,
        archive_uri=uri,
        graph_enabled=True,
    )
    return uri


def test_graph_replay_is_idempotent_and_source_deletion_removes_windows():
    index = EvidenceIndex()
    add(index, 1)
    add(index, 2)
    before = index.graph_diagnostics()
    add(index, 2)
    assert index.graph_diagnostics() == before
    assert before["window_count"] == 2
    assert before["edge_count"] > 0
    restored = EvidenceIndex.from_dict(index.to_dict())
    assert not restored.graph_unavailable
    assert restored.graph_diagnostics() == before
    index.remove_session(SESSION)
    assert not index.graph["windows"] and not index.graph["appears_in"]
    assert not index._graph_inv


def test_old_snapshot_keeps_raw_evidence_and_requests_graph_rebuild():
    index = EvidenceIndex()
    uri = add(index, 1)
    old = {"version": 3, "episodes": index.to_dict()["episodes"]}
    restored = EvidenceIndex.from_dict(old)
    assert restored.graph_unavailable
    assert restored.resolve_sources(["chat-1"])
    result = restored.rebuild_graph(
        [{"archive_uri": uri, "session_uri": SESSION, "source_ids": ["chat-1"]}]
    )
    assert result["built"] == 1
    assert not restored.graph_unavailable


def test_corrupt_graph_falls_back_without_losing_raw_sources():
    index = EvidenceIndex()
    add(index, 1)
    data = index.to_dict()
    data["graph"] = {"windows": "corrupt"}
    restored = EvidenceIndex.from_dict(data)
    assert restored.graph_unavailable
    assert restored.resolve_sources(["chat-1"])
