# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One-request ownership of in-flight and completed query embeddings."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from openviking.models.embedder.base import EmbedResult


@dataclass
class _SharedQueryEmbedding:
    task: "asyncio.Task[EmbedResult]"
    waiters: int = 0


class QueryEmbeddingCache(dict[Any, _SharedQueryEmbedding]):
    """Own in-flight query work for one request and track its waiters."""

    def __init__(self) -> None:
        super().__init__()
        self._pending_tasks: set[asyncio.Task] = set()
        self.computations = 0
        self.hits = 0

    async def run(
        self,
        key: Any,
        factory: Callable[[], Awaitable["EmbedResult"]],
    ) -> "EmbedResult":
        entry = self.get(key)
        if entry is None:
            self.computations += 1
            entry = _SharedQueryEmbedding(asyncio.create_task(factory()))
            self[key] = entry
            entry.task.add_done_callback(lambda done: self._discard_unsuccessful(key, entry, done))

        else:
            self.hits += 1
        entry.waiters += 1
        try:
            return await asyncio.shield(entry.task)
        finally:
            entry.waiters -= 1
            if entry.waiters == 0 and not entry.task.done():
                if self.get(key) is entry:
                    self.pop(key, None)
                self._pending_tasks.add(entry.task)
                entry.task.cancel()

    def _discard_unsuccessful(
        self,
        key: Any,
        entry: _SharedQueryEmbedding,
        done: "asyncio.Task[EmbedResult]",
    ) -> None:
        self._pending_tasks.discard(done)
        if (done.cancelled() or done.exception() is not None) and self.get(key) is entry:
            self.pop(key, None)

    async def close(self) -> None:
        pending = {entry.task for entry in self.values() if not entry.task.done()} | {
            task for task in self._pending_tasks if not task.done()
        }
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.clear()
        self._pending_tasks.clear()


# Request handlers install one cache around the fan-out that issues sibling
# finds. Context copying gives every child task the same cache instance.
query_embed_cache_var: contextvars.ContextVar[Optional[QueryEmbeddingCache]] = (
    contextvars.ContextVar("ov_query_embed_cache", default=None)
)


@asynccontextmanager
async def query_embed_cache_scope() -> AsyncIterator[QueryEmbeddingCache]:
    """Install and clean up the request-scoped query embedding cache."""
    cache = QueryEmbeddingCache()
    token = query_embed_cache_var.set(cache)
    try:
        yield cache
    finally:
        await cache.close()
        query_embed_cache_var.reset(token)
