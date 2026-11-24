# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P3：缓存「账号是否开启 ACL」这一配置查询（TTL 默认 3 秒）。

为什么只缓存这一个字段
  1. 它读的是账号级配置（acl.enabled），本身不授予任何权限；
  2. 命中缓存后，真正的可访问性判定（_is_accessible / resolve_many）照常执行，
     所以不会出现「换个用户也复用上次允许结果」的问题；
  3. 每次调用 is_enabled 都要经过配置管理器的异步锁与
     run_to_completion 调度，实测约 0.2~0.3 ms，是读路径上最贵的一段非 IO 开销。

代价
  账号级 ACL 开关的切换（开启/关闭）最多延迟 TTL 生效（默认 3 秒）。
  文件级权限判定不受影响，缓存不涉及 allow/deny 结论。

关闭：OPENVIKING_ACL_ENABLED_CACHE=0（false/off/no/disable 同样生效）
TTL：  OPENVIKING_ACL_ENABLED_CACHE_TTL，默认 3.0 秒（模块导入时读取）
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple
from openviking import _kunpeng_opt


_ENV = "OPENVIKING_ACL_ENABLED_CACHE"
_OFF_VALUES = {"0", "false", "off", "no", "disable", "disabled"}
_TTL_SECONDS = float(os.getenv("OPENVIKING_ACL_ENABLED_CACHE_TTL", "3.0"))


def cache_enabled() -> bool:
    if not _kunpeng_opt.enabled():
        return False
    return os.getenv(_ENV, "").strip().lower() not in _OFF_VALUES


class AclEnabledCache:
    """按 account_id 缓存 is_enabled 的布尔结果。"""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[bool, float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, account_id: str) -> Optional[bool]:
        item = self._data.get(account_id)
        if item is None:
            return None
        value, expire_at = item
        if expire_at <= time.monotonic():
            self._data.pop(account_id, None)
            return None
        return value

    def put(self, account_id: str, value: bool) -> None:
        self._data[account_id] = (value, time.monotonic() + _TTL_SECONDS)

    def invalidate(self, account_id: str) -> None:
        self._data.pop(account_id, None)

    def clear(self) -> None:
        self._data.clear()

    def reset_stats(self) -> None:
        self.hits = self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": cache_enabled(),
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "ttl_s": _TTL_SECONDS,
        }


CACHE = AclEnabledCache()


async def acl_enabled_cached(manager: Any, account_id: str) -> bool:
    """带缓存的 is_enabled；缓存关闭时直接透传。"""
    if not cache_enabled():
        return await manager.is_enabled(account_id)
    cached = CACHE.get(account_id)
    if cached is not None:
        CACHE.hits += 1
        return cached
    CACHE.misses += 1
    value = bool(await manager.is_enabled(account_id))
    CACHE.put(account_id, value)
    return value
