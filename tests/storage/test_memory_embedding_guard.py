import hashlib
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.storage.collection_schemas import TextEmbeddingHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,revision,current",
    [
        ("new", hashlib.sha256(b"old").hexdigest(), False),
        ("same", hashlib.sha256(b"same").hexdigest(), True),
        (None, hashlib.sha256(b"deleted").hexdigest(), False),
    ],
)
async def test_memory_embedding_guard_checks_revision_and_holds_lease(
    monkeypatch, content, revision, current
):
    fs = SimpleNamespace(
        _async_agfs=SimpleNamespace(
            pathlock_acquire_exact=AsyncMock(return_value="lease"), pathlock_release=AsyncMock()
        ),
        _uri_to_path=lambda uri, ctx=None: uri,
        read_file=AsyncMock(return_value=content),
    )
    if content is None:
        fs.read_file.side_effect = FileNotFoundError()
    module = importlib.import_module("openviking.storage.viking_fs")
    monkeypatch.setattr(module._wrapped, "get_viking_fs", lambda: fs, raising=False)
    async with TextEmbeddingHandler._memory_write_guard("card", revision, "ctx", 2) as valid:
        assert valid is current
        fs._async_agfs.pathlock_release.assert_not_awaited()
    fs._async_agfs.pathlock_release.assert_awaited_once_with("lease")


@pytest.mark.asyncio
async def test_overview_revision_reads_its_sidecar_and_propagates_storage_failure(monkeypatch):
    fs = SimpleNamespace(
        _async_agfs=AsyncMock(),
        _uri_to_path=lambda uri, ctx=None: uri,
        read_file=AsyncMock(side_effect=PermissionError("denied")),
    )
    module = importlib.import_module("openviking.storage.viking_fs")
    monkeypatch.setattr(module._wrapped, "get_viking_fs", lambda: fs, raising=False)
    with pytest.raises(PermissionError):
        async with TextEmbeddingHandler._memory_write_guard("directory", "revision", "ctx", 1):
            pytest.fail("storage errors must not publish stale vectors")
    fs.read_file.assert_awaited_once_with("directory/.overview.md", ctx="ctx")
    fs._async_agfs.pathlock_release.assert_awaited_once()
