import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.retrieve.evidence_retriever import EvidenceIndex, load_index
from openviking.server.identity import RequestContext, Role
from openviking.session.session import Session
from openviking_cli.session.user_id import UserIdentifier


class EvidenceFS:
    def __init__(self):
        self.files = {}
        self.locks = {}
        self._async_agfs = SimpleNamespace(
            pathlock_acquire_exact=self.acquire, pathlock_release=self.release
        )
        self._get_embedder = lambda ctx: None

    def _uri_to_path(self, uri, ctx=None):
        return f"{ctx.account_id}/{uri}"

    async def acquire(self, path, timeout_secs=30):
        lock = self.locks.setdefault(path, asyncio.Lock())
        await lock.acquire()
        return lock

    async def release(self, lease):
        lease.release()

    async def read_file(self, uri, ctx=None):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        await asyncio.sleep(0)
        self.files[uri] = content.decode() if isinstance(content, bytes) else content


@pytest.mark.asyncio
async def test_failed_embedding_survives_reload_and_repairs_without_duplicate(monkeypatch):
    from openviking.models.embedder import base
    from openviking.retrieve.evidence_embedding import repair_embeddings

    fs = EvidenceFS()
    fs._get_embedder = lambda ctx: object()
    chat = session(fs, "retry-chat")
    monkeypatch.setattr(base, "embed_compat", AsyncMock(side_effect=TimeoutError()))
    await chat._append_to_evidence_index(
        [Message(id="m1", role="user", parts=[TextPart("I planted a tree.")])],
        chat.uri + "/history/archive_001",
    )
    uri = "viking://user/alice/.index/evidence.json"
    before = await load_index(fs, uri, chat.ctx)
    turn = next(iter(before._turn_by_id.values()))
    assert not turn.get("emb")
    assert turn["embedding_state"]["status"] == "pending"
    assert before.embedding_diagnostics()["missing_embedding"] == 1
    turn["embedding_state"]["next_retry_at"] = 0
    before._dirty = True
    await before.save(fs, uri, chat.ctx)
    call = AsyncMock(return_value=SimpleNamespace(dense_vector=[1.0, 0.0]))
    monkeypatch.setattr(base, "embed_compat", call)
    # Concurrent/repeated repairs claim the persisted attempt only once.
    await asyncio.gather(*(repair_embeddings(fs, uri, chat.ctx) for _ in range(2)))
    await repair_embeddings(fs, uri, chat.ctx)
    after = await load_index(fs, uri, chat.ctx)
    assert after._n_turns == 1
    assert next(iter(after._turn_by_id.values()))["emb"] == [1.0, 0.0]
    assert next(iter(after._turn_by_id.values()))["embedding_state"] == {
        "status": "ready",
        "attempts": 2,
    }
    assert after.embedding_diagnostics()["pending_embedding"] == 0
    assert "embedding_state" not in after.resolve_sources(["m1"])[0]
    call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("bad input"), Exception("401 unauthorized"), Exception("AccountQuotaExceeded")],
)
async def test_permanent_embedding_error_stays_failed_without_retry(monkeypatch, error):
    from openviking.models.embedder import base
    from openviking.retrieve.evidence_embedding import repair_embeddings

    fs = EvidenceFS()
    fs._get_embedder = lambda ctx: object()
    chat = session(fs, "chat")
    call = AsyncMock(side_effect=error)
    monkeypatch.setattr(base, "embed_compat", call)
    await chat._append_to_evidence_index(
        [Message(id="m1", role="user", parts=[TextPart("text")])], chat.uri + "/history/a"
    )
    uri = "viking://user/alice/.index/evidence.json"
    await repair_embeddings(fs, uri, chat.ctx)
    turn = next(iter((await load_index(fs, uri, chat.ctx))._turn_by_id.values()))
    assert turn["embedding_state"]["status"] == "failed"
    assert turn["embedding_state"]["attempts"] == 1
    call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "correct"])
async def test_repair_does_not_restore_deleted_or_corrected_vectors(monkeypatch, change):
    from openviking.models.embedder import base
    from openviking.retrieve.evidence_embedding import repair_embeddings

    fs = EvidenceFS()
    fs._get_embedder = lambda ctx: object()
    chat = session(fs, "chat")
    uri = "viking://user/alice/.index/evidence.json"
    index = EvidenceIndex()
    index.append_chunk(
        "chat",
        [{"id": "m1", "text": "old"}],
        "viking://user/alice",
        embedding_states=[{"status": "pending", "attempts": 1, "next_retry_at": 0}],
    )
    await index.save(fs, uri, chat.ctx)

    async def embed(*args, **kwargs):
        assert all(not lock.locked() for lock in fs.locks.values())
        current = await load_index(fs, uri, chat.ctx)
        if change == "delete":
            current.remove_session(chat.uri)
        else:
            current.replace_source("m1", "new", "user", "")
        await current.save(fs, uri, chat.ctx)
        return SimpleNamespace(dense_vector=[1.0, 0.0])

    monkeypatch.setattr(base, "embed_compat", embed)
    await repair_embeddings(fs, uri, chat.ctx)
    final = await load_index(fs, uri, chat.ctx)
    assert all(not t.get("emb") for t in final._turn_by_id.values())
    if change == "correct":
        assert final.resolve_sources(["m1"])[0]["text"] == "new"


@pytest.mark.asyncio
async def test_repair_stops_at_persisted_attempt_limit(monkeypatch):
    from openviking.models.embedder import base
    from openviking.retrieve.evidence_embedding import MAX_ATTEMPTS, repair_embeddings

    fs = EvidenceFS()
    fs._get_embedder = lambda ctx: object()
    chat = session(fs, "chat")
    uri = "viking://user/alice/.index/evidence.json"
    index = EvidenceIndex()
    index.append_chunk(
        "chat",
        [{"id": "m1", "text": "old"}],
        "viking://user/alice",
        embedding_states=[{"status": "pending", "attempts": MAX_ATTEMPTS, "next_retry_at": 0}],
    )
    await index.save(fs, uri, chat.ctx)
    call = AsyncMock()
    monkeypatch.setattr(base, "embed_compat", call)
    await repair_embeddings(fs, uri, chat.ctx)
    final = await load_index(fs, uri, chat.ctx)
    assert next(iter(final._turn_by_id.values()))["embedding_state"]["status"] == "failed"
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_repair_keeps_durable_claim_and_resumes(monkeypatch):
    from openviking.models.embedder import base
    from openviking.retrieve.evidence_embedding import repair_embeddings

    fs = EvidenceFS()
    fs._get_embedder = lambda ctx: object()
    chat = session(fs, "chat")
    uri = "viking://user/alice/.index/evidence.json"
    index = EvidenceIndex()
    index.append_chunk(
        "chat",
        [{"id": "m1", "text": "original"}],
        "viking://user/alice",
        embedding_states=[{"status": "pending", "attempts": 1, "next_retry_at": 0}],
    )
    await index.save(fs, uri, chat.ctx)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(base, "embed_compat", blocked)
    task = asyncio.create_task(repair_embeddings(fs, uri, chat.ctx))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    persisted = await load_index(fs, uri, chat.ctx)
    state = next(iter(persisted._turn_by_id.values()))["embedding_state"]
    assert state["status"] == "pending" and state["attempts"] == 2
    state["next_retry_at"] = 0  # Simulate the persisted claim expiring after restart.
    persisted._dirty = True
    await persisted.save(fs, uri, chat.ctx)
    monkeypatch.setattr(
        base, "embed_compat", AsyncMock(return_value=SimpleNamespace(dense_vector=[1.0, 0.0]))
    )
    await repair_embeddings(fs, uri, chat.ctx)
    final = await load_index(fs, uri, chat.ctx)
    assert final._n_turns == 1 and final.embedding_diagnostics()["ready"] == 1
    assert next(iter(final._turn_by_id.values()))["embedding_state"]["attempts"] == 3
    assert all(not lock.locked() for lock in fs.locks.values())


@pytest.mark.asyncio
async def test_repair_scanner_preserves_account_and_user_context(monkeypatch):
    import openviking.retrieve.evidence_embedding as repair

    fs = EvidenceFS()

    async def listing(path):
        if path == "/local":
            return [{"name": name, "isDir": True} for name in ["tenant-a", "tenant-b", "_system"]]
        return [{"name": "alice", "isDir": True}]

    fs._async_agfs.ls = listing
    call = AsyncMock()
    monkeypatch.setattr(repair, "repair_embeddings", call)
    await repair.EvidenceEmbeddingRepair(fs).scan_once()
    assert [(c.args[2].account_id, c.args[2].user.user_id) for c in call.await_args_list] == [
        ("tenant-a", "alice"),
        ("tenant-b", "alice"),
    ]
    assert all(
        c.args[1] == "viking://user/alice/.index/evidence.json" for c in call.await_args_list
    )


def session(fs, name):
    ctx = RequestContext(UserIdentifier("tenant", "alice"), Role.USER)
    return Session(viking_fs=fs, session_id=name, ctx=ctx)


@pytest.mark.asyncio
async def test_concurrent_sessions_append_without_lost_turns():
    fs = EvidenceFS()
    a, b = session(fs, "first"), session(fs, "second")
    await asyncio.gather(
        *(
            item._append_to_evidence_index(
                [Message(id=name, role="user", parts=[TextPart(name)])],
                item.uri + "/history/archive_001",
            )
            for item, name in [(a, "m1"), (b, "m2")]
        )
    )
    index = await load_index(fs, "viking://user/alice/.index/evidence.json", a.ctx)
    assert index._n_turns == 2
    assert index.resolve_sources(["m1", "m2"])
    assert all(not lock.locked() for lock in fs.locks.values())


@pytest.mark.asyncio
async def test_archive_indexes_all_text_parts_and_message_role():
    fs = EvidenceFS()
    a = session(fs, "chat")
    await a._append_to_evidence_index(
        [
            Message(
                id="m1",
                role="user",
                parts=[TextPart("[Caroline]: I went to Paris."), TextPart("I returned yesterday.")],
            )
        ],
        a.uri + "/history/archive_001",
    )
    index = await load_index(fs, "viking://user/alice/.index/evidence.json", a.ctx)
    turn = index.resolve_sources(["m1"])[0]
    assert turn["role"] == "user" and turn["speaker"] == "Caroline"
    assert "returned yesterday" in turn["text"]


@pytest.mark.asyncio
async def test_index_lock_releases_after_failed_write():
    fs = EvidenceFS()
    a = session(fs, "chat")
    fs.write_file = AsyncMock(side_effect=OSError("unavailable"))
    with pytest.raises(OSError):
        await a._append_to_evidence_index(
            [Message(id="m1", role="user", parts=[TextPart("hello")])],
            a.uri + "/history/archive_001",
        )
    assert all(not lock.locked() for lock in fs.locks.values())


@pytest.mark.asyncio
async def test_same_session_archives_keep_contiguous_question_answer_context():
    from openviking.retrieve.evidence_strategy import build_evidence_groups

    fs = EvidenceFS()
    chat = session(fs, "conversation")
    await chat._append_to_evidence_index(
        [
            Message(id="m1", role="user", parts=[TextPart("[Dana]: Here is my new book.")]),
            Message(
                id="m2", role="assistant", parts=[TextPart("[Robin]: What did you learn from it?")]
            ),
        ],
        chat.uri + "/history/archive_001",
    )
    await chat._append_to_evidence_index(
        [Message(id="m3", role="user", parts=[TextPart("[Dana]: It taught me patience.")])],
        chat.uri + "/history/archive_002",
    )
    index = await load_index(fs, "viking://user/alice/.index/evidence.json", chat.ctx)
    hit = dict(index.resolve_sources(["m3"])[0], turn_id="conversation:2", score=1.0)
    group = build_evidence_groups(index, [hit])[0]
    assert group["turn_ids"] == ["conversation:0", "conversation:1", "conversation:2"]
    # Deleting this session still revokes every archive through the same URI.
    assert len({t["uri"] for t in group["turns"]}) == 1


@pytest.mark.asyncio
async def test_model_fact_verification_distinguishes_named_subject_and_role():
    from datetime import datetime, timezone

    from openviking.retrieve.memory_sources import MemorySources, verify_fact
    from openviking.session.memory.atomic_facts import select_user_facts
    from openviking.utils.spacy_utils import load_spacy_model

    if load_spacy_model("en_core_web_sm") is None:
        pytest.skip(
            "install the deterministic extra and en_core_web_sm for model integration tests"
        )
    text = "[Caroline]: I completed 7 articles since last year."
    date = datetime(2023, 5, 8, tzinfo=timezone.utc)
    rows = select_user_facts(text, "m1", date, "Caroline")
    assert rows
    fact = rows[0]["fact"]
    assert fact["speaker"] == "user"
    assert fact["subject"] == "Caroline"
    index = EvidenceIndex()
    index.append_chunk(
        "chat",
        [
            {
                "id": "m1",
                "role": "user",
                "speaker": "Caroline",
                "text": text,
                "date": date.isoformat(),
            }
        ],
        user_space="viking://user/alice",
    )
    assert verify_fact(fact, index)
    hints = MemorySources(facts=[fact])
    assert hints.candidates("What articles did Caroline complete?", index, 10)
    tampered = dict(fact, object="invented content")
    assert not verify_fact(tampered, index)
    index.replace_source("m1", "[Caroline]: I completed 4 articles.", "Caroline", date.isoformat())
    assert not verify_fact(fact, index)
