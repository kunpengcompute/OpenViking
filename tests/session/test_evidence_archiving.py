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
