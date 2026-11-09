from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.retrieve.evidence_retriever import EvidenceIndex
from openviking.session.memory.source_cleanup import guard_source_update, prune_fields


def fields():
    return {
        "memory_type": "entities",
        "name": "Alice",
        "aliases": ["Alice"],
        "content": "Old merged description containing both supports.",
        "fact_records": [
            {
                "id": "f1",
                "source_turn_ids": ["old"],
                "source_quotes": {"old": "Removed assertion."},
            },
            {
                "id": "f2",
                "source_turn_ids": ["keep"],
                "source_quotes": {"keep": "Surviving assertion."},
            },
        ],
    }


def test_cleanup_rebuilds_prose_from_surviving_supports():
    original = fields()
    pruned = prune_fields(original, lambda sid, fact: sid == "old")
    assert pruned["content"] == "Surviving assertion."
    assert len(pruned["fact_records"]) == 1
    assert len(original["fact_records"]) == 2
    assert prune_fields(original, lambda sid, fact: True) is None
    assert prune_fields(original, lambda sid, fact: False) is original


@pytest.mark.asyncio
async def test_final_guard_rejects_deleted_sources():
    index = EvidenceIndex()
    index.invalidate_sources(["old", "keep"], "session")
    assert await guard_source_update(None, fields(), "card", None, index=index) is None


@pytest.mark.asyncio
async def test_peer_guard_reads_the_owning_users_index(monkeypatch):
    index = EvidenceIndex()
    loader = AsyncMock(return_value=index)
    monkeypatch.setattr("openviking.retrieve.evidence_retriever.load_index", loader)
    fs, ctx = object(), object()
    await guard_source_update(
        fs, fields(), "viking://user/alice/peers/bob/memories/entities/a.md", ctx
    )
    loader.assert_awaited_once_with(fs, "viking://user/alice/.index/evidence.json", ctx)


@pytest.mark.asyncio
async def test_cleanup_enumerates_self_and_peer_memory_roots(monkeypatch):
    from openviking.session.memory import source_cleanup

    cleanup = AsyncMock(return_value={"updated": 1, "deleted": 0})
    monkeypatch.setattr(source_cleanup, "_cleanup_memory_root", cleanup)
    fs = SimpleNamespace(
        exists=AsyncMock(return_value=True),
        ls=AsyncMock(return_value=[{"name": "bob"}, {"name": ".."}]),
    )
    index = EvidenceIndex()
    result = await source_cleanup.cleanup_source_memories(
        fs, index, "viking://user/alice/.index/evidence.json", "lease", ["old"], "ctx"
    )
    assert result == {"updated": 2, "deleted": 0}
    assert {call.kwargs["root"] for call in cleanup.await_args_list} == {
        "viking://user/alice/memories",
        "viking://user/alice/peers/bob/memories",
    }


@pytest.mark.asyncio
async def test_fresh_read_failure_does_not_overwrite_prefetched_memory(monkeypatch):
    from pathlib import Path

    from openviking.server.identity import RequestContext, Role
    from openviking.session.memory.dataclass import ResolvedOperation
    from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
    from openviking.session.memory.memory_updater import MemoryReadError, MemoryUpdater
    from openviking_cli.session.user_id import UserIdentifier

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_directory(
        str(Path(__file__).parents[3] / "openviking/prompts/templates/memory")
    )
    fs = SimpleNamespace(
        read_file=AsyncMock(side_effect=PermissionError("denied")), write_file=AsyncMock()
    )
    updater = MemoryUpdater(registry=registry)
    updater._viking_fs = fs
    operation = ResolvedOperation(
        memory_type="entities",
        uris=["viking://user/alice/memories/entities/a.md"],
        memory_fields={"content": "new"},
    )
    with pytest.raises(MemoryReadError):
        await updater._apply_upsert(
            operation, RequestContext(UserIdentifier("tenant", "alice"), Role.USER)
        )
    fs.write_file.assert_not_awaited()
