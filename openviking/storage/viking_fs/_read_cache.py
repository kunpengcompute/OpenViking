# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P2：VikingFS 读路径的只读内容缓存（内存 LRU + 版本号 + TTL 兜底）。

设计要点
  1. 只缓存 read_file 读出的文本；URI 解析与权限检查（_ensure_access）在
     每次读时照常执行，只有鉴权通过后才查缓存，命中时跳过路径映射与 AGFS 读盘；
  2. key = (account, user, uri)，天然按用户隔离，不会串数据；
  3. 写入 / 追加 / 删除 / 移动都主动失效：失效不再只删条目，而是把该 URI 的
     版本号 +1；缓存条目记录「读之前」的版本，读时版本不一致直接当 miss。
     这样「读 miss → 并发写 → 读线程回填旧内容」的竞态不会留下长期脏数据；
  4. 只保证单进程内一致：多 worker（workers > 1）启动时自动关闭本缓存，
     进程之间不共享失效通知，跨进程共享请改用上游 CacheFS（storage.agfs.cachefs）；
  5. stats() 暴露命中率，评测报告需要注明「缓存口径」。

关闭方式：OPENVIKING_CONTENT_CACHE=0（false/off/no/disable 同样生效）。
TTL 可用 OPENVIKING_CONTENT_CACHE_TTL 覆盖（模块导入时读取，改后需重启）。
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from openviking import _kunpeng_opt

_ENV = "OPENVIKING_CONTENT_CACHE"
_OFF_VALUES = {"0", "false", "off", "no", "disable", "disabled"}
_TTL_SECONDS = float(os.getenv("OPENVIKING_CONTENT_CACHE_TTL", "600.0"))
_MAX_ENTRIES = int(os.getenv("OPENVIKING_CONTENT_CACHE_MAX_ENTRIES", "4096"))
_MAX_ENTRY_BYTES = int(os.getenv("OPENVIKING_CONTENT_CACHE_MAX_ENTRY_BYTES", str(256 * 1024)))
_MAX_VERSION_ENTRIES = 65536

CacheKey = Tuple[str, str, str]
CacheVersion = Tuple[int, int]  # (uri 版本号, 前缀代次)


def cache_enabled() -> bool:
    """缓存开关：默认开启，显式设置关闭值才走原生路径。"""
    if not _kunpeng_opt.enabled():
        return False
    return os.getenv(_ENV, "").strip().lower() not in _OFF_VALUES


def disable_for_multi_worker(workers: int) -> bool:
    """多 worker 启动时自动关闭内容缓存。

    每个 worker 是独立进程、各自一份内存缓存，进程之间感知不到对方的写入，
    因此默认退回无缓存路径，避免读到旧内容。单进程（workers <= 1）不做任何事；
    部署方显式设置过 OPENVIKING_CONTENT_CACHE 时以显式值为准。

    返回 True 表示本次做了自动关闭。
    """
    if workers <= 1:
        return False
    if os.getenv(_ENV) is not None:
        return False
    os.environ[_ENV] = "0"
    return True


def make_key(ctx: Any, uri: str) -> CacheKey:
    """按 (账号, 用户, URI) 生成 key，避免跨用户命中。"""
    user = getattr(ctx, "user", None)
    account = str(getattr(user, "account_id", None) or getattr(ctx, "account_id", "") or "")
    uid = str(getattr(user, "user_id", None) or getattr(user, "uid", None) or user or "")
    return (account, uid, uri)


class ReadContentCache:
    """带 TTL 的 LRU 文本缓存；条目带版本号，失效通过版本递增实现。"""

    def __init__(self) -> None:
        self._data: "OrderedDict[CacheKey, Tuple[str, float, CacheVersion]]" = OrderedDict()
        self._by_uri: Dict[str, set] = {}
        self._ver: Dict[str, int] = {}
        self._prefix_epoch = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.stores = 0
        self.version_misses = 0
        self.bumps = 0

    def __len__(self) -> int:
        return len(self._data)

    # ---------- 版本 ----------

    def version(self, uri: str) -> CacheVersion:
        """读之前取版本；必须早于底层 read，写入后再 bump。"""
        with self._lock:
            return (self._ver.get(uri, 0), self._prefix_epoch)

    def bump(self, uri: str) -> None:
        """写成功后调用：让该 URI 的所有已缓存/在途回填条目失效。"""
        with self._lock:
            self._ver[uri] = self._ver.get(uri, 0) + 1
            self.bumps += 1
            if len(self._ver) > _MAX_VERSION_ENTRIES:
                # 版本表本身也是内存；清空并推进前缀代次，保证旧条目全部失配
                self._ver.clear()
                self._prefix_epoch += 1

    # ---------- 读写 ----------

    def get(self, key: CacheKey, version: Optional[CacheVersion] = None) -> Optional[str]:
        if not cache_enabled():
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            text, expire_at, ver = item
            if version is not None and ver != version:
                self._drop(key)
                self.misses += 1
                self.version_misses += 1
                return None
            if expire_at <= time.monotonic():
                self._drop(key)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return text

    def put(
        self,
        key: CacheKey,
        text: str,
        version: Optional[CacheVersion] = None,
    ) -> None:
        if not cache_enabled() or not text or len(text) > _MAX_ENTRY_BYTES:
            return
        with self._lock:
            self._data[key] = (
                text,
                time.monotonic() + _TTL_SECONDS,
                version if version is not None else (0, 0),
            )
            self._by_uri.setdefault(key[2], set()).add(key)
            self._data.move_to_end(key)
            self.stores += 1
            while len(self._data) > _MAX_ENTRIES:
                oldest, _ = self._data.popitem(last=False)
                self._discard_index(oldest)

    def _discard_index(self, key: CacheKey) -> None:
        keys = self._by_uri.get(key[2])
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._by_uri.pop(key[2], None)

    def _drop(self, key: CacheKey) -> None:
        self._data.pop(key, None)
        self._discard_index(key)

    # ---------- 失效 ----------

    def invalidate(self, uri: str) -> int:
        """精确失效单个 URI（所有用户）：先递增版本，再清理已缓存条目。"""
        with self._lock:
            self.bump(uri)
            keys = self._by_uri.pop(uri, set())
            for k in keys:
                self._data.pop(k, None)
            self.invalidations += len(keys)
            return len(keys)

    def invalidate_prefix(self, prefix: str) -> int:
        """按目录前缀失效（删除/移动整棵子树时使用）：递增前缀代次。"""
        with self._lock:
            self._prefix_epoch += 1
            keys = [k for k in self._data if k[2].startswith(prefix)]
            for k in keys:
                self._data.pop(k, None)
                self._discard_index(k)
            self.invalidations += len(keys)
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._by_uri.clear()

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = self.misses = self.invalidations = self.stores = 0
            self.version_misses = self.bumps = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": cache_enabled(),
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "invalidations": self.invalidations,
                "version_misses": self.version_misses,
                "bumps": self.bumps,
                "hit_rate": (self.hits / total) if total else 0.0,
                "ttl_s": _TTL_SECONDS,
                "max_entries": _MAX_ENTRIES,
            }


CACHE = ReadContentCache()
