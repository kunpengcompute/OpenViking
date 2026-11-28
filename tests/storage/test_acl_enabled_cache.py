# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""账号级 ACL 开关缓存（P3）的单元测试。

被测对象：`openviking/storage/viking_fs/_acl_cache.py`

设计约束（也是这些用例要守住的东西）：
  1. 只缓存「这个账号有没有开启 ACL 权限体系」这一个布尔配置，
     不缓存任何 allow / deny 结论，因此不会出现「换个用户复用上次允许结果」；
  2. key 是账号 ID，不同账号之间互不影响；
  3. 命中缓存只省掉一次配置查询，真正的可访问性判定（_is_accessible）照常执行；
  4. 账号开关的切换最多延迟一个 TTL 生效；
  5. 关闭开关（OPENVIKING_ACL_ENABLED_CACHE=0）后完全透传，等价于改动前。
"""

import asyncio
import time
from typing import Any, Dict, List

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.viking_fs import VikingFS
from openviking.storage.viking_fs import _acl_cache as acl_cache
from openviking.storage.viking_fs._acl_cache import CACHE as ACL_CACHE
from openviking.storage.viking_fs._acl_cache import acl_enabled_cached
from openviking_cli.session.user_id import UserIdentifier


class FakeAclManager:
    """只实现被测代码会调用的 is_enabled，并记录调用次数与耗时。"""

    def __init__(self, enabled: bool = False, delay: float = 0.0) -> None:
        self.enabled = enabled
        self.delay = delay
        self.calls = 0
        self.resolve_calls = 0
        self.accounts: List[str] = []

    async def is_enabled(self, account_id: str) -> bool:
        self.calls += 1
        self.accounts.append(account_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.enabled

    async def resolve_many(self, uris, ctx) -> Dict[str, Any]:
        """真正的 ACL 判定入口：每次调用都必须被执行。"""
        self.resolve_calls += 1
        return {uri: {"acl": "fake-effective"} for uri in uris}


def _ctx(user_id: str = "default", account_id: str = "default") -> RequestContext:
    return RequestContext(
        user=UserIdentifier(account_id, user_id),
        role=Role.ROOT,
    )


@pytest.fixture(autouse=True)
def _reset_acl_cache(monkeypatch):
    monkeypatch.delenv("OPENVIKING_ACL_ENABLED_CACHE", raising=False)
    ACL_CACHE.clear()
    ACL_CACHE.reset_stats()
    yield
    ACL_CACHE.clear()
    ACL_CACHE.reset_stats()


# ---------------------------------------------------------------------------
# 缓存行为
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _enable_kunpeng_opt(monkeypatch):
    """单测在任意架构上开启鲲鹏优化总开关（CI 多为 x86）。"""
    monkeypatch.setattr("openviking._kunpeng_opt.enabled", lambda: True)


async def test_same_account_is_queried_once():
    manager = FakeAclManager(enabled=True)

    for _ in range(5):
        assert await acl_enabled_cached(manager, "acct-1") is True

    assert manager.calls == 1
    assert ACL_CACHE.hits == 4
    assert ACL_CACHE.misses == 1


async def test_different_accounts_do_not_share_entries():
    manager = FakeAclManager(enabled=False)

    assert await acl_enabled_cached(manager, "acct-a") is False
    assert await acl_enabled_cached(manager, "acct-b") is False
    assert manager.calls == 2
    assert manager.accounts == ["acct-a", "acct-b"]


async def test_entry_expires_after_ttl(monkeypatch):
    """TTL 到期后必须重新查询，保证账号开关变更最多延迟一个 TTL。"""
    manager = FakeAclManager(enabled=True)
    fake_now = {"t": 1000.0}
    monkeypatch.setattr(acl_cache.time, "monotonic", lambda: fake_now["t"])

    assert await acl_enabled_cached(manager, "acct-1") is True
    assert manager.calls == 1

    # TTL 内命中
    fake_now["t"] += acl_cache._TTL_SECONDS - 0.1
    assert await acl_enabled_cached(manager, "acct-1") is True
    assert manager.calls == 1

    # 超过 TTL 后重新查询，并看到新的开关值
    manager.enabled = False
    fake_now["t"] += 1.0
    assert await acl_enabled_cached(manager, "acct-1") is False
    assert manager.calls == 2


async def test_invalidate_forces_refresh():
    manager = FakeAclManager(enabled=False)
    assert await acl_enabled_cached(manager, "acct-1") is False

    ACL_CACHE.invalidate("acct-1")
    manager.enabled = True

    assert await acl_enabled_cached(manager, "acct-1") is True
    assert manager.calls == 2


async def test_disabled_env_passes_through(monkeypatch):
    monkeypatch.setenv("OPENVIKING_ACL_ENABLED_CACHE", "0")
    manager = FakeAclManager(enabled=True)

    for _ in range(3):
        assert await acl_enabled_cached(manager, "acct-1") is True

    assert manager.calls == 3
    assert len(ACL_CACHE._data) == 0


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "disable", "disabled"])
async def test_off_values_all_disable_cache(monkeypatch, value):
    monkeypatch.setenv("OPENVIKING_ACL_ENABLED_CACHE", value)
    assert acl_cache.cache_enabled() is False


async def test_default_is_enabled(monkeypatch):
    monkeypatch.delenv("OPENVIKING_ACL_ENABLED_CACHE", raising=False)
    assert acl_cache.cache_enabled() is True


async def test_stats_report_hit_rate():
    manager = FakeAclManager(enabled=False)
    await acl_enabled_cached(manager, "acct-1")
    for _ in range(3):
        await acl_enabled_cached(manager, "acct-1")

    stats = ACL_CACHE.stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.75)
    assert stats["entries"] == 1
    assert stats["ttl_s"] == acl_cache._TTL_SECONDS


# ---------------------------------------------------------------------------
# 与 _can_access_many 的联动：只省配置查询，不省权限判定
# ---------------------------------------------------------------------------


def _make_fs(manager: FakeAclManager) -> VikingFS:
    fs = VikingFS(agfs=object(), acl_manager=manager)
    fs._async_agfs = object()
    return fs


async def test_can_access_many_queries_config_once_per_account():
    manager = FakeAclManager(enabled=False)
    fs = _make_fs(manager)
    ctx = _ctx()
    uris = [f"viking://resources/file{i}.md" for i in range(10)]

    for _ in range(5):
        result = await fs._can_access_many(uris, ctx)
        assert all(result[uri] is True for uri in uris)

    # 10 个 URI × 5 轮，只查一次账号配置
    assert manager.calls == 1


async def test_acl_enabled_still_uses_real_access_check(monkeypatch):
    """账号开启 ACL 时，配置缓存命中不能跳过真正的 ACL 判定。"""
    from openviking.storage.viking_fs import _access as access_mod

    monkeypatch.setattr(access_mod, "acl_allows", lambda effective, ctx, action: True)
    manager = FakeAclManager(enabled=True)
    fs = _make_fs(manager)
    ctx = _ctx()
    uris = ["viking://resources/a.md", "viking://resources/b.md"]

    first = await fs._can_access_many(uris, ctx)
    second = await fs._can_access_many(uris, ctx)

    assert first == second == dict.fromkeys(uris, True)
    assert manager.calls == 1  # 账号配置只查一次（缓存命中）
    assert manager.resolve_calls == 2  # ACL 判定每次都真的执行


async def test_other_user_does_not_reuse_denied_decision():
    """换用户时，即使配置缓存命中，可见性判定也必须重新执行。"""
    manager = FakeAclManager(enabled=False)
    fs = _make_fs(manager)

    root_ctx = RequestContext(user=UserIdentifier("default", "default"), role=Role.ROOT)
    user_ctx = RequestContext(user=UserIdentifier("default", "alice"), role=Role.USER)
    hidden = "viking://_system/secret.md"

    assert await fs._can_access_many([hidden], root_ctx) == {hidden: True}
    # _system 命名空间对普通用户不可见，说明结论没有被 ROOT 的缓存复用
    assert await fs._can_access_many([hidden], user_ctx) == {hidden: False}
    assert manager.calls == 1


# ---------------------------------------------------------------------------
# 时延：缓存 vs 不走缓存
# ---------------------------------------------------------------------------


async def test_cached_config_lookup_is_faster(monkeypatch):
    manager = FakeAclManager(enabled=False, delay=0.0003)  # 模拟 0.3ms 配置查询
    rounds = 50

    monkeypatch.setenv("OPENVIKING_ACL_ENABLED_CACHE", "0")
    started = time.perf_counter()
    for _ in range(rounds):
        await acl_enabled_cached(manager, "acct-1")
    uncached_ms = (time.perf_counter() - started) * 1000

    ACL_CACHE.clear()
    monkeypatch.delenv("OPENVIKING_ACL_ENABLED_CACHE")
    await acl_enabled_cached(manager, "acct-1")  # warm
    started = time.perf_counter()
    for _ in range(rounds):
        await acl_enabled_cached(manager, "acct-1")
    cached_ms = (time.perf_counter() - started) * 1000

    assert cached_ms < uncached_ms * 0.2
    print(f"\nP3: uncached={uncached_ms:.2f}ms cached={cached_ms:.2f}ms")
