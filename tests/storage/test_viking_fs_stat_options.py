# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Stat enrichment is optional; existence and access semantics are unchanged."""

import json
from types import SimpleNamespace

import pytest

import openviking.storage.transaction as transaction_module
from openviking.pyagfs.exceptions import (
    AGFSConnectionError,
    AGFSNotFoundError,
    AGFSPermissionDeniedError,
    AGFSTimeoutError,
)
from openviking.server.identity import RequestContext, Role
from openviking.service.fs_service import FSService
from openviking.service.session_service import SessionService
from openviking.session.session import Session
from openviking.storage.expr import PathScope
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier


class _CountingAGFS:
    def __init__(self):
        self.entries = {}
        self.files = {}
        self.stats = []
        self.reads = []
        self.error = None

    def stat(self, path):
        self.stats.append(path)
        if self.error is not None:
            raise self.error
        if path in self.files:
            return {"name": path.rsplit("/", 1)[-1], "isDir": False}
        if path not in self.entries:
            raise FileNotFoundError(path)
        return dict(self.entries[path])

    def read(self, path, *args):
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class _CountingLockManager:
    def __init__(self):
        self.paths = []

    async def is_path_locked_async(self, path, ignore_stale=True):
        self.paths.append(path)
        return True


class _CountingVectorStore:
    def __init__(self):
        self.calls = []

    async def count(self, *, filter, ctx):
        self.calls.append((filter, ctx))
        return 7


@pytest.fixture
def stat_case(monkeypatch):
    agfs = _CountingAGFS()
    locks = _CountingLockManager()
    vectors = _CountingVectorStore()
    monkeypatch.setattr(transaction_module, "get_lock_manager", lambda: locks)
    ctx = RequestContext(user=UserIdentifier("acct", "user-1"), role=Role.USER)
    fs = VikingFS(agfs=agfs, vector_store=vectors)
    return SimpleNamespace(agfs=agfs, locks=locks, vectors=vectors, ctx=ctx, fs=fs)


@pytest.mark.parametrize("is_dir", [False, True], ids=["file", "directory"])
@pytest.mark.parametrize(
    "options,count_enabled,lock_enabled",
    [
        ({}, True, True),
        ({"skip_count": False, "include_lock_status": True}, True, True),
        ({"skip_count": True, "include_lock_status": True}, False, True),
        ({"skip_count": False, "include_lock_status": False}, True, False),
        ({"skip_count": True, "include_lock_status": False}, False, False),
    ],
    ids=["defaults", "both", "lock-only", "count-only", "neither"],
)
async def test_stat_enrichment_is_independently_optional(
    stat_case, is_dir, options, count_enabled, lock_enabled
):
    case = stat_case
    path = "/local/acct/resources/item"
    metadata = {"name": "item", "isDir": is_dir, "size": 123, "modTime": "2026-01-01"}
    case.agfs.entries[path] = metadata

    result = await case.fs.stat("viking://resources/item", case.ctx, **options)

    assert {key: result[key] for key in metadata} == metadata
    assert case.agfs.stats == [path]
    assert case.agfs.reads == []
    assert ("count" in result) is (is_dir and count_enabled)
    assert ("isLocked" in result) is lock_enabled
    if lock_enabled:
        assert result["isLocked"] is True
    if is_dir and count_enabled:
        assert result["count"] == 7
    assert case.locks.paths == ([path] if lock_enabled else [])
    assert case.vectors.calls == (
        [(PathScope("uri", "viking://resources/item", depth=-1), case.ctx)]
        if is_dir and count_enabled
        else []
    )


@pytest.mark.parametrize("entrypoint", ["session", "service"])
async def test_session_existence_does_not_query_count_or_locks(stat_case, entrypoint):
    case = stat_case
    path = "/local/acct/user/user-1/sessions/present"
    case.agfs.entries[path] = {"name": "present", "isDir": True}
    # Existence must not parse or preload messages or metadata.
    case.agfs.files[f"{path}/messages.jsonl"] = b"invalid JSONL"
    case.agfs.files[f"{path}/.meta.json"] = b"invalid JSON"

    if entrypoint == "session":
        session = Session(viking_fs=case.fs, session_id="present", ctx=case.ctx)
        assert await session.exists() is True
    else:
        session = await SessionService(viking_fs=case.fs).open("present", case.ctx)
        assert session.session_id == "present"

    assert case.agfs.stats == [path]
    assert case.agfs.reads == []
    assert case.vectors.calls == []
    assert case.locks.paths == []


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        TimeoutError("timed out"),
        ConnectionError("unavailable"),
        AGFSPermissionDeniedError("denied"),
        AGFSTimeoutError("timed out"),
        AGFSConnectionError("unavailable"),
    ],
)
async def test_session_exists_preserves_backend_failures(stat_case, error):
    case = stat_case
    case.agfs.error = error
    session = Session(viking_fs=case.fs, session_id="present", ctx=case.ctx)

    with pytest.raises(type(error)) as caught:
        await session.exists()

    # Python 3.11 may reconstruct TimeoutError while adapting a thread future.
    assert type(caught.value) is type(error)
    assert caught.value.args == error.args
    assert len(case.agfs.stats) == 1
    assert case.vectors.calls == []
    assert case.locks.paths == []


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing"), AGFSNotFoundError("missing"), NotFoundError("missing")]
)
async def test_session_exists_returns_false_for_missing_storage(stat_case, error):
    case = stat_case
    case.agfs.error = error
    session = Session(viking_fs=case.fs, session_id="missing", ctx=case.ctx)

    assert await session.exists() is False
    assert len(case.agfs.stats) == 1
    assert case.vectors.calls == []
    assert case.locks.paths == []


async def test_lightweight_stat_preserves_user_and_actor_access_checks(stat_case):
    case = stat_case
    case.ctx.actor_peer_id = "visible-peer"
    for uri in (
        "viking://user/other-user/sessions/private",
        "viking://user/user-1/peers/hidden-peer/resources/private",
    ):
        with pytest.raises(PermissionDeniedError):
            await case.fs.stat(uri, case.ctx, skip_count=True, include_lock_status=False)
    session = Session(
        viking_fs=case.fs,
        session_id="private",
        ctx=case.ctx,
        session_uri="viking://user/other-user/sessions/private",
    )
    with pytest.raises(PermissionDeniedError):
        await session.exists()

    assert case.agfs.stats == []
    assert case.vectors.calls == []
    assert case.locks.paths == []


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "user-scoped"])
@pytest.mark.parametrize("owner", ["user-1", "other-user"])
async def test_session_exists_keeps_legacy_fallback_and_owner_visibility(stat_case, nested, owner):
    case = stat_case
    prefix = "/local/acct/session/user-1" if nested else "/local/acct/session"
    path = f"{prefix}/legacy"
    case.agfs.entries[path] = {"name": "legacy", "isDir": True}
    case.agfs.files[f"{path}/.meta.json"] = json.dumps({"created_by_user_id": owner}).encode()
    session = Session(
        viking_fs=case.fs,
        session_id="legacy",
        ctx=case.ctx,
        session_uri="viking://session/legacy",
    )

    assert await session.exists() is (owner == "user-1")
    assert f"{path}/.meta.json" in case.agfs.reads
    assert case.vectors.calls == []
    assert case.locks.paths == []


@pytest.mark.parametrize("include_lock_status", [False, True])
async def test_synthetic_legacy_session_root_honors_lock_option(stat_case, include_lock_status):
    case = stat_case

    result = await case.fs.stat(
        "viking://session", case.ctx, skip_count=True, include_lock_status=include_lock_status
    )

    assert result["isDir"] is True
    assert result["name"] == "session"
    assert ("isLocked" in result) is include_lock_status
    if include_lock_status:
        assert result["isLocked"] is False
    assert "count" not in result
    assert case.vectors.calls == []
    assert case.locks.paths == []


async def test_fs_service_default_stat_retains_count_and_lock_status(stat_case):
    case = stat_case
    path = "/local/acct/resources"
    case.agfs.entries[path] = {"name": "resources", "isDir": True}

    result = await FSService(viking_fs=case.fs).stat("viking://resources", case.ctx)

    assert result["count"] == 7
    assert result["isLocked"] is True
    assert case.locks.paths == [path]
    assert case.vectors.calls == [(PathScope("uri", "viking://resources", depth=-1), case.ctx)]
