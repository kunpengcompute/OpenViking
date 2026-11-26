# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""VikingFS 读路径优化（P0/P1/P2）的单元测试。

覆盖三类改动：
  P0  探测合并：先 read，只有读到空内容才补一次 stat；
  P1  响应预编码：/content/read 在 payload 是纯字符串时直接输出预编码 JSON；
  P2  读内容缓存：命中时跳过路径映射与 AGFS 读盘，鉴权照常执行。

同时覆盖正确性（空文件 / 目录 / 不存在 / 切片 / 跨用户隔离 / 写后失效）
与前后时延对比（固定注入 AGFS 延迟）。
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response as FastAPIResponse

from openviking.pyagfs.exceptions import AGFSNotFoundError
from openviking.server.identity import RequestContext, Role
from openviking.server.models import Response as ServerResponse
from openviking.server.routers import content as content_router
from openviking.storage.viking_fs import VikingFS, _read_cache
from openviking.storage.viking_fs._read_cache import CACHE as READ_CACHE
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


class FakeAsyncAGFS:
    """Minimal in-memory stand-in for AsyncAGFSClient."""

    def __init__(self, delay: float = 0.0) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.delay = delay
        self.read_calls = 0
        self.stat_calls = 0
        self.write_calls = 0

    async def read(self, path: str, *args, **kwargs) -> bytes:
        self.read_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.files.get(path, b"")

    async def stat(self, path: str, *args, **kwargs) -> dict:
        self.stat_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if path in self.dirs:
            return {"isDir": True, "size": 0}
        if path in self.files:
            return {"isDir": False, "size": len(self.files[path])}
        raise AGFSNotFoundError(path)

    async def write(self, path: str, data, *args, **kwargs) -> str:
        self.write_calls += 1
        self.files[path] = data if isinstance(data, bytes) else data.encode("utf-8")
        return path

    async def ensure_parent_dirs(self, path: str, *args, **kwargs) -> None:
        return None

    async def mkdir(self, path: str, *args, **kwargs) -> None:
        self.dirs.add(path)

    async def pathlock_acquire_exact(self, path: str, *args, **kwargs) -> dict:
        return {"lease_ref": f"lease:{path}"}

    async def pathlock_acquire_tree(self, path: str, *args, **kwargs) -> dict:
        return {"lease_ref": f"lease:{path}"}

    async def pathlock_release(self, lease, *args, **kwargs) -> None:
        return None

    async def rm(self, path: str, *args, **kwargs):
        self.files.pop(path, None)
        self.dirs.discard(path)
        return {}


def _ctx(user_id: str = "default") -> RequestContext:
    return RequestContext(user=UserIdentifier("default", user_id), role=Role.ROOT)


def _make_fs(delay: float = 0.0) -> tuple[VikingFS, FakeAsyncAGFS]:
    fs = VikingFS(agfs=object())
    fake = FakeAsyncAGFS(delay=delay)
    fs._async_agfs = fake
    return fs, fake


def _path(fs: VikingFS, uri: str) -> str:
    return fs._uri_to_path(uri, ctx=_ctx())


@pytest.fixture(autouse=True)
def _reset_read_cache():
    READ_CACHE.clear()
    READ_CACHE.reset_stats()
    yield
    READ_CACHE.clear()
    READ_CACHE.reset_stats()


# ---------------------------------------------------------------------------
# P2 correctness
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _enable_kunpeng_opt(monkeypatch):
    """单测在任意架构上开启鲲鹏优化总开关（CI 多为 x86）。"""
    monkeypatch.setattr("openviking._kunpeng_opt.enabled", lambda: True)


async def test_second_read_hits_cache(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"hello\nworld\n"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    assert await fs.read_file(uri, ctx=_ctx()) == "hello\nworld\n"
    assert await fs.read_file(uri, ctx=_ctx()) == "hello\nworld\n"

    assert fake.read_calls == 1
    assert READ_CACHE.hits == 1


async def test_cache_hit_still_runs_authorization(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"cached"
    ensure_access = AsyncMock()
    monkeypatch.setattr(fs, "_ensure_access", ensure_access)

    await fs.read_file(uri, ctx=_ctx())
    await fs.read_file(uri, ctx=_ctx())

    assert ensure_access.await_count == 2
    assert fake.read_calls == 1


async def test_write_file_invalidates_cache(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"before"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    assert await fs.read_file(uri, ctx=_ctx()) == "before"
    await fs.write_file(uri, "after", ctx=_ctx())

    assert await fs.read_file(uri, ctx=_ctx()) == "after"
    assert fake.read_calls == 2


async def test_append_file_invalidates_cache(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"hello"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    assert await fs.read_file(uri, ctx=_ctx()) == "hello"
    await fs.append_file(uri, " world", ctx=_ctx())
    calls_after_append = fake.read_calls

    assert await fs.read_file(uri, ctx=_ctx()) == "hello world"
    assert fake.read_calls == calls_after_append + 1


async def test_rm_invalidates_cache(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"gone"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    assert await fs.read_file(uri, ctx=_ctx()) == "gone"
    await fs.rm(uri, ctx=_ctx())

    with pytest.raises(NotFoundError):
        await fs.read_file(uri, ctx=_ctx())


async def test_offset_limit_matches_between_cache_paths(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"a\nb\nc\nd\n"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    await fs.read_file(uri, ctx=_ctx())
    hit = await fs.read_file(uri, ctx=_ctx(), offset=1, limit=2)
    READ_CACHE.clear()
    miss = await fs.read_file(uri, ctx=_ctx(), offset=1, limit=2)

    assert hit == miss == "b\nc\n"
    assert fake.read_calls == 2


async def test_cache_is_isolated_per_user(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"shared"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    assert await fs.read_file(uri, ctx=_ctx("alice")) == "shared"
    assert await fs.read_file(uri, ctx=_ctx("bob")) == "shared"

    assert fake.read_calls == 2


async def test_cache_can_be_disabled_by_env(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"uncached"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "0")

    assert await fs.read_file(uri, ctx=_ctx()) == "uncached"
    assert await fs.read_file(uri, ctx=_ctx()) == "uncached"

    assert fake.read_calls == 2
    assert len(READ_CACHE) == 0


# ---------------------------------------------------------------------------
# P1 correctness
# ---------------------------------------------------------------------------


class _FakeFs:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def read(self, uri, ctx, offset, limit):
        return self.value

    async def read_visible(self, uri, ctx, offset, limit):
        return self.value


class _FakeService:
    def __init__(self, value: Any) -> None:
        self.fs = _FakeFs(value)


async def test_read_fast_json_matches_native_response(monkeypatch):
    value = "line one\nline two"
    monkeypatch.setattr(content_router, "get_service", lambda: _FakeService(value))
    # A 32-hex vector record id skips request URI validation.
    uri = "0123456789abcdef0123456789abcdef"
    ctx = _ctx()

    monkeypatch.delenv("OPENVIKING_READ_FAST_JSON", raising=False)
    fast = await content_router.read(uri=uri, offset=0, limit=-1, raw=False, _ctx=ctx)
    assert isinstance(fast, FastAPIResponse)
    payload = json.loads(fast.body)

    monkeypatch.setenv("OPENVIKING_READ_FAST_JSON", "0")
    native = await content_router.read(uri=uri, offset=0, limit=-1, raw=False, _ctx=ctx)
    assert isinstance(native, ServerResponse)

    assert native.status == payload["status"]
    assert native.result == payload["result"]
    # 两条路径的 JSON 必须字段同构（含 error/telemetry/profile 的 null）
    assert set(payload) == set(native.model_dump())
    assert native.model_dump() == payload


async def test_p1_non_str_result_falls_back_to_native(monkeypatch):
    """非 JSON 原生类型（datetime 等）必须退回原生路径，不能直接 json.dumps。"""
    result = {"file": "profile.md", "updated_at": datetime(2023, 5, 8)}
    monkeypatch.setattr(content_router, "get_service", lambda: _FakeService(result))
    monkeypatch.delenv("OPENVIKING_READ_FAST_JSON", raising=False)
    uri = "0123456789abcdef0123456789abcdef"

    response = await content_router.read(uri=uri, offset=0, limit=-1, raw=False, _ctx=_ctx())

    assert isinstance(response, ServerResponse)
    assert response.result == result
    encoded = json.loads(json.dumps(jsonable_encoder(response)))
    assert encoded["result"]["updated_at"] == "2023-05-08T00:00:00"


# ---------------------------------------------------------------------------
# P0 correctness: 空文件 / 目录 / 不存在 / 正常文件不 stat
# ---------------------------------------------------------------------------


async def test_p0_normal_file_skips_stat(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"hello"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "0")

    assert await fs.read_file(uri, ctx=_ctx()) == "hello"
    assert fake.read_calls == 1
    assert fake.stat_calls == 0


async def test_p0_empty_file_returns_empty(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/empty.txt"
    fake.files[_path(fs, uri)] = b""
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "0")

    assert await fs.read_file(uri, ctx=_ctx()) == ""
    assert fake.read_calls == 1
    assert fake.stat_calls == 1


async def test_p0_directory_raises_invalid_argument(monkeypatch):
    fs, fake = _make_fs()
    uri = "viking://resources/folder"
    fake.dirs.add(_path(fs, uri))
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    with pytest.raises(InvalidArgumentError):
        await fs.read_file(uri, ctx=_ctx())


class _DirectoryReadError(Exception):
    """模拟 LocalFS 读目录时抛出的插件错误。"""


async def test_p0_directory_backend_error_keeps_error_contract(monkeypatch):
    """后端读目录抛错时仍返回带 expected/actual 的 InvalidArgumentError。"""
    fs, fake = _make_fs()
    uri = "viking://resources/folder"
    fake.dirs.add(_path(fs, uri))

    async def _raise_on_read(*args, **kwargs):
        raise _DirectoryReadError("is a directory")

    fake.read = _raise_on_read
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    with pytest.raises(InvalidArgumentError) as excinfo:
        await fs.read_file(uri, ctx=_ctx())
    assert excinfo.value.details == {
        "resource": uri,
        "expected": "file",
        "actual": "directory",
    }
    assert "List it first, then read a file URI" in str(excinfo.value)


async def test_p0_non_directory_read_error_is_not_masked(monkeypatch):
    """非目录的后端读错误原样上抛，不被 stat 兜底逻辑吞掉。"""
    fs, fake = _make_fs()
    uri = "viking://resources/broken.txt"
    fake.files[_path(fs, uri)] = b"ok"

    async def _boom(*args, **kwargs):
        raise OSError("backend exploded")

    fake.read = _boom
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    with pytest.raises(OSError, match="backend exploded"):
        await fs.read_file(uri, ctx=_ctx())


async def test_p0_missing_file_raises_not_found(monkeypatch):
    fs, _ = _make_fs()
    uri = "viking://resources/missing.txt"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())

    with pytest.raises(NotFoundError):
        await fs.read_file(uri, ctx=_ctx())


# ---------------------------------------------------------------------------
# 时延：缓存 vs 不走缓存 / 合并探测 vs stat+read / 预编码 vs 原生
# ---------------------------------------------------------------------------


async def test_p2_cached_read_is_faster_than_uncached(monkeypatch):
    fs, fake = _make_fs(delay=0.002)
    uri = "viking://resources/notes.txt"
    fake.files[_path(fs, uri)] = b"payload\n" * 100
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    rounds = 20

    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "0")
    started = time.perf_counter()
    for _ in range(rounds):
        await fs.read_file(uri, ctx=_ctx())
    uncached_ms = (time.perf_counter() - started) * 1000

    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE")
    await fs.read_file(uri, ctx=_ctx())  # warm the cache
    started = time.perf_counter()
    for _ in range(rounds):
        await fs.read_file(uri, ctx=_ctx())
    cached_ms = (time.perf_counter() - started) * 1000

    assert fake.read_calls == rounds + 1
    assert cached_ms < uncached_ms * 0.5
    print(f"\nP2: uncached={uncached_ms:.1f}ms cached={cached_ms:.1f}ms")


async def test_p0_fused_read_is_faster_than_stat_plus_read(monkeypatch):
    fs, fake = _make_fs(delay=0.002)
    uri = "viking://resources/notes.txt"
    path = _path(fs, uri)
    fake.files[path] = b"hello\n"
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "0")
    rounds = 20

    started = time.perf_counter()
    for _ in range(rounds):
        await fake.stat(path)
        await fake.read(path)
    legacy_ms = (time.perf_counter() - started) * 1000

    fake.stat_calls = 0
    started = time.perf_counter()
    for _ in range(rounds):
        assert await fs.read_file(uri, ctx=_ctx()) == "hello\n"
    fused_ms = (time.perf_counter() - started) * 1000

    assert fake.stat_calls == 0
    assert fused_ms < legacy_ms
    print(f"\nP0: stat+read={legacy_ms:.1f}ms fused={fused_ms:.1f}ms")


async def test_p1_fast_json_is_faster_than_native(monkeypatch):
    monkeypatch.setattr(content_router, "get_service", lambda: _FakeService("line\n" * 100))
    uri = "0123456789abcdef0123456789abcdef"
    ctx = _ctx()
    rounds = 2000

    def _render(response) -> str:
        """复现框架把响应转成字节前做的事，保证两条路径比较口径一致。"""
        if isinstance(response, ServerResponse):
            return json.dumps(jsonable_encoder(response))
        return response.body.decode()

    monkeypatch.setenv("OPENVIKING_READ_FAST_JSON", "0")
    started = time.perf_counter()
    for _ in range(rounds):
        _render(await content_router.read(uri=uri, offset=0, limit=-1, raw=False, _ctx=ctx))
    native_ms = (time.perf_counter() - started) * 1000

    monkeypatch.delenv("OPENVIKING_READ_FAST_JSON")
    started = time.perf_counter()
    for _ in range(rounds):
        _render(await content_router.read(uri=uri, offset=0, limit=-1, raw=False, _ctx=ctx))
    fast_ms = (time.perf_counter() - started) * 1000

    assert fast_ms < native_ms
    print(
        f"\nP1: native={native_ms:.1f}ms fast={fast_ms:.1f}ms "
        f"({native_ms / rounds * 1000:.1f}us -> {fast_ms / rounds * 1000:.1f}us)"
    )


# ---------------------------------------------------------------------------
# P2 consistency: 版本号失效 / 并发不残留旧内容 / 多 worker 自动关闭
# ---------------------------------------------------------------------------


def test_stale_refill_after_concurrent_write_is_missed(monkeypatch):
    """读 miss -> 写完成 -> 读线程迟到回填旧正文：下一次读必须 miss。"""
    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE", raising=False)
    cache = _read_cache.ReadContentCache()
    uri = "viking://user/alice/memories/m1.md"
    key = _read_cache.make_key(_ctx("alice"), uri)

    version_before = cache.version(uri)  # 读线程：读之前取版本
    cache.invalidate(uri)  # 写线程：写成功后失效（版本 +1）
    cache.put(key, "stale", version_before)  # 读线程：迟到的旧内容回填

    assert cache.get(key, cache.version(uri)) is None


def test_concurrent_read_write_never_returns_stale(monkeypatch):
    """多线程并发读写，读者看到的版本不低于读开始时已提交的版本。"""
    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE", raising=False)
    cache = _read_cache.ReadContentCache()
    uri = "viking://user/alice/memories/m1.md"
    key = _read_cache.make_key(_ctx("alice"), uri)
    state = {"gen": 0, "text": "0", "committed": 0}
    lock = threading.Lock()
    stop = threading.Event()
    stale: list[tuple[str, int]] = []

    def reader():
        while not stop.is_set():
            with lock:
                committed = state["committed"]
            version = cache.version(uri)
            text = cache.get(key, version)
            if text is None:
                with lock:
                    text = state["text"]
                cache.put(key, text, version)
            if int(text) < committed:
                stale.append((text, committed))

    def writer():
        while not stop.is_set():
            with lock:
                state["gen"] += 1
                state["text"] = str(state["gen"])
                gen = state["gen"]
            cache.invalidate(uri)
            with lock:
                state["committed"] = max(state["committed"], gen)
            time.sleep(0.0005)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join()

    assert not stale, f"stale reads: {stale[:5]}"


def test_invalidate_affects_all_users(monkeypatch):
    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE", raising=False)
    cache = _read_cache.ReadContentCache()
    uri = "viking://user/shared/notes.md"
    alice = _read_cache.make_key(_ctx("alice"), uri)
    bob = _read_cache.make_key(_ctx("bob"), uri)
    version = cache.version(uri)
    cache.put(alice, "a", version)
    cache.put(bob, "b", version)

    cache.invalidate(uri)
    newer = cache.version(uri)

    assert newer != version
    assert cache.get(alice, newer) is None
    assert cache.get(bob, newer) is None


def test_single_worker_keeps_cache(monkeypatch):
    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE", raising=False)

    assert _read_cache.disable_for_multi_worker(1) is False
    assert "OPENVIKING_CONTENT_CACHE" not in os.environ
    assert _read_cache.cache_enabled()


def test_multi_worker_disables_cache(monkeypatch):
    monkeypatch.delenv("OPENVIKING_CONTENT_CACHE", raising=False)

    assert _read_cache.disable_for_multi_worker(4) is True
    assert os.environ["OPENVIKING_CONTENT_CACHE"] == "0"
    assert not _read_cache.cache_enabled()


def test_explicit_setting_wins(monkeypatch):
    monkeypatch.setenv("OPENVIKING_CONTENT_CACHE", "1")

    assert _read_cache.disable_for_multi_worker(4) is False
    assert _read_cache.cache_enabled()


# ---------------------------------------------------------------------------
# P2 consistency: mv 走 cp + 底层 rm，不经过 rm()，必须显式失效源与目标
# ---------------------------------------------------------------------------


class _MvAGFS:
    """支持 mv（cp + 底层 rm）的最小 AGFS。"""

    def __init__(self) -> None:
        self.paths = {"/local/acct/resources/source.md"}
        self.files = {"/local/acct/resources/source.md": b"before"}

    async def stat(self, path, fs_ctx=None):
        if path in self.paths:
            return {"isDir": False}
        if path == "/local/acct/resources":
            return {"isDir": True}
        raise FileNotFoundError(path)

    async def pathlock_acquire_batch(self, requests, timeout_secs=0.0, owner_lease_ref=None):
        return {"lease_ref": "test-lease"}

    async def pathlock_release(self, lease):
        return None

    async def cp(self, source, target, recursive=False, fs_ctx=None):
        self.paths.add(target)
        self.files[target] = self.files.get(source, b"")

    async def rm(self, path, recursive=False, fs_ctx=None):
        self.paths.discard(path)
        self.files.pop(path, None)


def _mv_fs(monkeypatch) -> VikingFS:
    fs = VikingFS.__new__(VikingFS)
    fs._async_agfs = _MvAGFS()
    fs.vector_store = None
    fs.acl_manager = None
    monkeypatch.setattr(fs, "_ensure_access", AsyncMock())
    monkeypatch.setattr(
        fs,
        "_uri_to_path",
        lambda uri, **_kwargs: f"/local/acct/{uri.removeprefix('viking://')}",
    )
    monkeypatch.setattr(
        fs,
        "_path_to_uri",
        lambda path, **_kwargs: f"viking://{path.removeprefix('/local/acct/')}",
    )
    monkeypatch.setattr(fs, "_update_vector_store_uris", AsyncMock())
    monkeypatch.setattr(fs, "_delete_from_vector_store", AsyncMock())
    return fs


async def test_mv_invalidates_source_and_target(monkeypatch):
    """mv 之后源与目标都不能再命中旧缓存。"""
    fs = _mv_fs(monkeypatch)
    ctx = _ctx()
    src = "viking://resources/source.md"
    dst = "viking://resources/target.md"
    src_key = _read_cache.make_key(ctx, src)
    dst_key = _read_cache.make_key(ctx, dst)
    READ_CACHE.put(src_key, "before", READ_CACHE.version(src))
    READ_CACHE.put(dst_key, "stale-target", READ_CACHE.version(dst))

    await fs.mv(src, dst, ctx=ctx)

    assert READ_CACHE.get(src_key, READ_CACHE.version(src)) is None
    assert READ_CACHE.get(dst_key, READ_CACHE.version(dst)) is None
