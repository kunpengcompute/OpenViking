import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.retrieve.evidence_retriever import EvidenceIndex, load_index


@pytest.fixture(autouse=True)
def lexical(monkeypatch):
    def features(text):
        return [], set(text.lower().split())

    for module in ("evidence_retriever", "evidence_ranking"):
        monkeypatch.setattr(f"openviking.retrieve.{module}.extract_keys", features)


def append(index, session="ordinary", ids=("m1", "m2")):
    index.append_chunk(
        session,
        [
            {
                "id": sid,
                "speaker": "user",
                "text": "alpha beta" if i == 0 else "other topic",
                "date": "2026-11-09T10:00:00Z",
            }
            for i, sid in enumerate(ids)
        ],
        user_space="viking://user/alice",
    )


def test_stable_ids_retry_and_roundtrip():
    index = EvidenceIndex()
    append(index)
    append(index)
    append(index, ids=("m3",))
    assert index._n_turns == 3
    restored = EvidenceIndex.from_dict(json.loads(json.dumps(index.to_dict())))
    assert restored.resolve_sources(["m3"])[0]["turn"] == 2
    assert "emb" not in restored.get_turns(["ordinary:0"])[0]


def test_session_removal_revokes_all_lookup_channels():
    index = EvidenceIndex()
    append(index)
    append(index, session="survives", ids=("m3",))
    assert index.remove_session("viking://user/alice/sessions/ordinary") == 2
    assert index.resolve_sources(["m1"]) == []
    assert index.get_turns(["ordinary:0"]) == []
    assert index._n_turns == 1
    assert all(t.startswith("survives:") for t in index._lex_inv["alpha"])


def test_source_replacement_rebuilds_inversions_and_drops_vector():
    index = EvidenceIndex()
    append(index)
    index._turn_by_id["ordinary:0"]["emb"] = [1.0, 0.0]
    index.replace_source("m1", "replacement", "user", "2026-11-10")
    assert index.resolve_sources(["m1"])[0]["text"] == "replacement"
    assert index._lex_inv.get("alpha", set()) == set()
    assert "emb" not in index._turn_by_id["ordinary:0"]


def test_semantic_candidates_survive_lexical_miss_and_have_no_vectors():
    index = EvidenceIndex()
    append(index)
    index._turn_by_id["ordinary:0"]["emb"] = [1.0, 0.0]
    hits = index.retrieve_candidates("absent", limit=1, q_vec=[1.0, 0.0])
    assert hits[0]["source_id"] == "m1"
    assert "semantic_rank" in hits[0]
    assert "emb" not in hits[0]


@pytest.mark.asyncio
async def test_only_missing_storage_is_empty():
    fs = SimpleNamespace(read_file=AsyncMock(side_effect=FileNotFoundError()))
    assert (await EvidenceIndex.load(fs, "index", None))._n_turns == 0
    fs.read_file.side_effect = PermissionError("denied")
    with pytest.raises(PermissionError):
        await EvidenceIndex.load(fs, "index", None)
    fs.read_file.side_effect = None
    fs.read_file.return_value = "{broken"
    with pytest.raises(ValueError):
        await EvidenceIndex.load(fs, "index", None)


@pytest.mark.asyncio
async def test_pending_correction_blocks_every_normal_loader():
    index = EvidenceIndex()
    index.pending_source_updates = ["m1"]
    fs = SimpleNamespace(read_file=AsyncMock(return_value=json.dumps(index.to_dict())))
    from openviking_cli.exceptions import ConflictError

    with pytest.raises(ConflictError):
        await load_index(fs, "index", None)
    assert (await load_index(fs, "index", None, allow_pending=True)).pending_source_updates == [
        "m1"
    ]


@pytest.mark.asyncio
async def test_failed_save_preserves_dirty_snapshot_for_retry():
    index = EvidenceIndex()
    append(index)
    fs = SimpleNamespace(write_file=AsyncMock(side_effect=OSError("disk unavailable")))
    with pytest.raises(OSError):
        await index.save(fs, "index", None, lease_ref="lease")
    assert index._dirty
    fs.write_file.side_effect = None
    await index.save(fs, "index", None, lease_ref="lease")
    assert not index._dirty
    assert fs.write_file.call_args.kwargs["lease_ref"] == "lease"
