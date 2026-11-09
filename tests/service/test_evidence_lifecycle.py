import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.retrieve.evidence_retriever import EvidenceIndex, load_index
from openviking.server.identity import RequestContext, Role
from openviking.service.session_service import SessionService
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.source_cleanup import cleanup_source_memories
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.session.user_id import UserIdentifier

ROOT = "viking://user/alice"
SESSION = ROOT + "/sessions/chat"
INDEX = ROOT + "/.index/evidence.json"


class MemoryFS:
    def __init__(self):
        self.files = {}
        self._async_agfs = AsyncMock()
        self._ensure_access = AsyncMock()
        self._delete_from_vector_store = AsyncMock()
        self._get_embedder = lambda ctx: None

    def _uri_to_path(self, uri, ctx=None):
        return uri

    async def exists(self, uri, ctx=None):
        return uri in self.files or any(path.startswith(uri + "/") for path in self.files)

    async def stat(self, uri, ctx=None, **kwargs):
        if not await self.exists(uri):
            raise FileNotFoundError(uri)
        return {}

    async def read_file(self, uri, ctx=None):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        self.files[uri] = content.decode() if isinstance(content, bytes) else content

    async def rm(self, uri, recursive=False, ctx=None, lease_ref=None):
        for path in list(self.files):
            if path == uri or recursive and path.startswith(uri + "/"):
                del self.files[path]

    async def ls(self, uri, ctx=None, **kwargs):
        return [
            {"name": name}
            for name in sorted(
                {p[len(uri) + 1 :].split("/")[0] for p in self.files if p.startswith(uri + "/")}
            )
        ]

    async def glob(self, pattern, uri, ctx=None, **kwargs):
        suffix = "/messages.jsonl" if pattern.endswith("messages.jsonl") else ".md"
        return {
            "matches": [p for p in self.files if p.startswith(uri + "/") and p.endswith(suffix)],
            "truncated": False,
        }


@pytest.fixture
def setup(monkeypatch):
    ctx = RequestContext(UserIdentifier("tenant", "alice"), Role.USER)
    fs = MemoryFS()
    message = Message(
        id="m1",
        role="user",
        parts=[TextPart("I completed 4 articles.")],
        created_at="2023-05-08T10:00:00Z",
    )
    archive = SESSION + "/history/archive_001/messages.jsonl"
    fs.files[archive] = message.to_jsonl() + "\n"
    fs.files[SESSION + "/messages.jsonl"] = ""
    index = EvidenceIndex()
    index.append_chunk(
        "chat",
        [
            {
                "id": "m1",
                "role": "user",
                "speaker": "user",
                "text": message.content,
                "date": message.created_at,
            }
        ],
        user_space=ROOT,
    )
    fs.files[INDEX] = json.dumps(index.to_dict())
    monkeypatch.setattr(
        "openviking.service.session_service.get_task_tracker",
        lambda: SimpleNamespace(has_running=AsyncMock(return_value=False)),
    )
    monkeypatch.setattr(MemoryUpdater, "generate_overview", AsyncMock(return_value=True))
    compressor = SimpleNamespace(
        _extract_user_memories=AsyncMock(return_value=SimpleNamespace(memory_diff=None))
    )
    service = SessionService(viking_fs=fs)
    service._session_compressor = compressor
    return fs, ctx, service, archive


@pytest.mark.asyncio
async def test_delete_purges_index_and_authoritative_session(setup):
    fs, ctx, service, _ = setup
    assert await service.delete("chat", ctx)
    assert not await fs.exists(SESSION)
    index = await load_index(fs, INDEX, ctx)
    assert index.resolve_sources(["m1"]) == []
    assert index.source_invalidations["m1"] == SESSION


@pytest.mark.asyncio
async def test_correction_replaces_original_and_index_then_reextracts(setup):
    fs, ctx, service, archive = setup
    result = await service.correct_message("chat", "m1", "I completed 7 articles.", ctx)
    assert result["reextracted"]
    assert "7 articles" in fs.files[archive] and "4 articles" not in fs.files[archive]
    index = await load_index(fs, INDEX, ctx)
    assert index.resolve_sources(["m1"])[0]["text"] == "I completed 7 articles."
    assert index.source_invalidations["m1"] == SESSION
    assert index.pending_source_updates == []
    service._session_compressor._extract_user_memories.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_source_write_remains_blocked_and_same_correction_repairs_it(setup):
    from openviking_cli.exceptions import ConflictError, InternalError

    fs, ctx, service, archive = setup
    write = fs.write_file
    fail = True

    async def failing(uri, content, **kwargs):
        nonlocal fail
        if uri == archive and fail:
            fail = False
            raise OSError("temporarily unavailable")
        await write(uri, content, **kwargs)

    fs.write_file = failing
    with pytest.raises(InternalError) as error:
        await service.correct_message("chat", "m1", "I completed 7 articles.", ctx)
    assert error.value.details["retry_same_correction"]
    with pytest.raises(ConflictError):
        await load_index(fs, INDEX, ctx)
    await service.correct_message("chat", "m1", "I completed 7 articles.", ctx)
    assert not (await load_index(fs, INDEX, ctx)).pending_source_updates


@pytest.mark.asyncio
async def test_cleanup_removes_only_affected_fact_supports_in_peer_cards(setup):
    fs, ctx, _, _ = setup
    uri = ROOT + "/peers/bob/memories/entities/person/alice.md"
    fields = {
        "memory_type": "entities",
        "name": "Alice",
        "category": "person",
        "content": "both",
        "fact_records": [
            {"id": "old", "source_turn_ids": ["m1"], "source_quotes": {"m1": "old assertion"}},
            {
                "id": "keep",
                "source_turn_ids": ["m2"],
                "source_quotes": {"m2": "surviving assertion"},
            },
        ],
    }
    fs.files[uri] = MemoryFileUtils.write(MemoryFile.from_parsed(uri, fields))
    index = await load_index(fs, INDEX, ctx)
    result = await cleanup_source_memories(fs, index, INDEX, "lease", ["m1"], ctx)
    assert result == {"updated": 1, "deleted": 0}
    card = MemoryFileUtils.read(fs.files[uri], uri=uri)
    assert card.content == "surviving assertion"
    assert [f["id"] for f in card.extra_fields["fact_records"]] == ["keep"]
    assert not (await load_index(fs, INDEX, ctx)).pending_memory_cleanup


@pytest.mark.asyncio
async def test_delete_also_revokes_live_sources_that_were_never_indexed(setup):
    fs, ctx, service, _ = setup
    fs.files[SESSION + "/messages.jsonl"] = Message(
        id="live", role="user", parts=[TextPart("unarchived source")]
    ).to_jsonl()
    await service.delete("chat", ctx)
    assert (await load_index(fs, INDEX, ctx)).source_invalidations["live"] == SESSION


@pytest.mark.asyncio
async def test_failed_cleanup_preserves_journal_and_retry_refreshes_already_pruned_card(
    setup, monkeypatch
):
    fs, ctx, _, _ = setup
    uri = ROOT + "/memories/entities/person/alice.md"
    fields = {
        "memory_type": "entities",
        "name": "Alice",
        "category": "person",
        "content": "both",
        "fact_records": [
            {"id": "old", "source_turn_ids": ["m1"], "source_quotes": {"m1": "old"}},
            {"id": "keep", "source_turn_ids": ["m2"], "source_quotes": {"m2": "surviving"}},
        ],
    }
    fs.files[uri] = MemoryFileUtils.write(MemoryFile.from_parsed(uri, fields))
    refresh = AsyncMock(side_effect=RuntimeError("queue unavailable"))
    monkeypatch.setattr(MemoryUpdater, "refresh_file_embedding", refresh)
    index = await load_index(fs, INDEX, ctx)
    with pytest.raises(RuntimeError, match="queue unavailable"):
        await cleanup_source_memories(fs, index, INDEX, "lease", ["m1"], ctx, vikingdb=object())
    index = await load_index(fs, INDEX, ctx)
    assert index.pending_memory_cleanup == [uri]
    assert MemoryFileUtils.read(fs.files[uri], uri=uri).content == "surviving"
    refresh.side_effect = None
    refresh.return_value = True
    await cleanup_source_memories(fs, index, INDEX, "lease", ["m1"], ctx, vikingdb=object())
    assert refresh.await_count == 2
    assert (await load_index(fs, INDEX, ctx)).pending_memory_cleanup == []
