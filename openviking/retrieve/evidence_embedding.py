# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded, restart-safe repair of failed raw-evidence embeddings only."""

import asyncio
import logging
import time
from contextlib import suppress
from uuid import uuid4

from openviking.retrieve.evidence_retriever import evidence_index_lock, load_index
from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext, Role
from openviking.utils.model_retry import classify_api_error, extract_metric_error_code
from openviking_cli.session.user_id import UserIdentifier

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 4
SCAN_INTERVAL = 30
REPAIR_BATCH_SIZE = 16


async def embed_turn(embedder, text, *, attempts, timeout=None):
    """Keep sanitized durable state; do not store provider messages or credentials."""
    from openviking.models.embedder.base import embed_compat

    try:
        result = await asyncio.wait_for(
            embed_compat(embedder, text, is_query=False), timeout=timeout
        )
        vector = getattr(result, "dense_vector", None)
        if not vector:
            raise ValueError("Embedding provider returned no dense vector")
        return list(vector), {"status": "ready", "attempts": attempts}
    except Exception as exc:
        error_class = classify_api_error(exc)
        if isinstance(exc, ValueError):
            error_class = "permanent"
        code = extract_metric_error_code(exc)
        if code in {"401", "403"}:
            error_class = "auth"
        if isinstance(exc, (TimeoutError, ConnectionError)) or code in {
            "408",
            "500",
            "502",
            "503",
            "504",
        }:
            error_class = "transient"
        retry = error_class == "transient" and attempts < MAX_ATTEMPTS
        return None, {
            "status": "pending" if retry else "failed",
            "attempts": attempts,
            "error_class": error_class,
            "error_type": type(exc).__name__,
            "next_retry_at": time.time() + min(30 * 2 ** (attempts - 1), 300) if retry else None,
        }


async def repair_embeddings(fs, uri, ctx):
    """Claim under the existing lease, call models outside it, then check the source again."""
    snapshot = await load_index(fs, uri, ctx)
    due = [
        tid
        for tid, turn in snapshot._turn_by_id.items()
        if not turn.get("emb")
        and turn.get("embedding_state", {}).get("status") == "pending"
        and turn["embedding_state"].get("next_retry_at", 0) <= time.time()
    ][:REPAIR_BATCH_SIZE]
    if not due:
        return
    embedder = fs._get_embedder(ctx)
    if embedder is None:
        return
    for tid in due:
        async with evidence_index_lock(fs, uri, ctx) as lease:
            index = await load_index(fs, uri, ctx)
            turn = index._turn_by_id.get(tid)
            state = (turn or {}).get("embedding_state", {})
            if (
                not turn
                or turn.get("emb")
                or state.get("status") != "pending"
                or state.get("next_retry_at", 0) > time.time()
            ):
                continue
            if state["attempts"] >= MAX_ATTEMPTS:
                state.update(status="failed", next_retry_at=None)
                index._dirty = True
                await index.save(fs, uri, ctx, lease_ref=lease)
                continue
            claim = dict(
                state,
                attempts=state["attempts"] + 1,
                attempt_id=uuid4().hex,
                next_retry_at=time.time() + 120,
            )
            turn["embedding_state"] = claim
            text, source_id = turn["text"], turn.get("source_id")
            index._dirty = True
            await index.save(fs, uri, ctx, lease_ref=lease)
        vector, result = await embed_turn(embedder, text, attempts=claim["attempts"], timeout=20)
        async with evidence_index_lock(fs, uri, ctx) as lease:
            index = await load_index(fs, uri, ctx)
            turn = index._turn_by_id.get(tid)
            if (
                not turn
                or turn["text"] != text
                or turn.get("source_id") != source_id
                or turn.get("embedding_state") != claim
            ):
                continue  # A deletion, correction or another worker superseded this attempt.
            turn["embedding_state"] = result
            if vector:
                turn["emb"] = vector
            index._dirty = True
            await index.save(fs, uri, ctx, lease_ref=lease)


class EvidenceEmbeddingRepair:
    """Discover user indexes as the existing session scheduler does; no queue changes."""

    def __init__(self, fs):
        self.fs = fs
        self.task = None

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None

    async def scan_once(self):
        agfs = self.fs._async_agfs
        for account in await agfs.ls("/local"):
            account_id = account.get("name", "")
            if not account.get("isDir") or account_id == "_system":
                continue
            try:
                users = await agfs.ls(f"/local/{account_id}/user")
            except Exception as exc:
                if not is_not_found_error(exc):
                    logger.warning("[evidence] cannot scan account: %s", type(exc).__name__)
                continue
            for user in users:
                user_id = user.get("name", "")
                if not user.get("isDir"):
                    continue
                ctx = RequestContext(
                    UserIdentifier(account_id, user_id), Role.USER, bypass_acl=True
                )
                try:
                    await repair_embeddings(
                        self.fs, f"viking://user/{user_id}/.index/evidence.json", ctx
                    )
                except Exception as exc:
                    logger.warning("[evidence] repair skipped: %s", type(exc).__name__)

    async def _run(self):
        while True:
            try:
                await self.scan_once()
            except Exception as exc:
                logger.warning("[evidence] repair scan failed: %s", type(exc).__name__)
            await asyncio.sleep(SCAN_INTERVAL)
