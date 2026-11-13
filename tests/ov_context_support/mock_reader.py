# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fake-server snapshots with the production reader's recall and configuration."""

from __future__ import annotations

from typing import Any

from openviking.integrations.langchain.ov_context.reader import (
    ContextSnapshot,
    MemoryReaderConfig,
    OVContextReader,
    RecallResult,
)
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking.integrations.langchain.ov_context.state import OVContext
from openviking_cli.client.identity import OVIdentity
from openviking_cli.retrieve.types import FindResult
from tests.ov_context_support.mock_context import SnapshotAdapter


class _MemoryClient:
    def __init__(self, handle: AsyncClientHandle) -> None:
        self.handle = handle

    async def find(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        score_threshold: float | None = None,
        identity: OVIdentity | None = None,
    ) -> FindResult:
        result = await self.handle.call(
            "find",
            identity=identity,
            query=query,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
        )
        return result if isinstance(result, FindResult) else FindResult.from_dict(result)

    async def read(self, uri: str, *, identity: OVIdentity | None = None) -> str:
        return await self.handle.call("read", identity=identity, uri=uri)

    async def get_session_context(
        self,
        session_id: str,
        token_budget: int = 128_000,
        *,
        identity: OVIdentity | None = None,
    ) -> dict[str, Any]:
        return await self.handle.call(
            "get_session_context",
            identity=identity,
            session_id=session_id,
            token_budget=token_budget,
        )


class MockContextReader:
    def __init__(
        self,
        client: AsyncClientHandle,
        config: MemoryReaderConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        # This offline adapter emulates the legacy find wire contract.
        self.config = config or MemoryReaderConfig(search_mode="find")
        self.metadata = metadata if metadata is not None else {}
        self.reader = OVContextReader(_MemoryClient(client), self.config)

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity | None = None,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        snapshot = await SnapshotAdapter(self.client, self.config, self.metadata).read_context(
            OVContext(session_id),
            expected_archive_id=expected_archive_id,
            identity=identity,
            token_budget=token_budget,
        )
        if snapshot.archive_id is not None and not snapshot.view.get("latest_archive_overview"):
            raise ValueError("Completed archive snapshot is missing its summary")
        return ContextSnapshot(snapshot.messages, snapshot.archive_id)

    async def recall(
        self,
        query: str,
        *,
        identity: OVIdentity | None = None,
        session_id: str,
    ) -> RecallResult:
        return await self.reader.recall(query, identity=identity, session_id=session_id)
