# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Candidate extraction uses original sources and existing ownership policies."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message
from openviking.message.part import TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.memory import fact_records
from openviking.session.memory.deterministic_extractor import DeterministicMemoryExtractor
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext
from openviking_cli.session.user_id import UserIdentifier


@pytest.fixture
def setup_extractor(monkeypatch):
    monkeypatch.setattr(fact_records, "_parse", lambda text: None)
    monkeypatch.setattr(DeterministicMemoryExtractor, "_get_nlp", lambda self, lang: None)
    registry = MemoryTypeRegistry(load_schemas=False)
    templates = Path(__file__).parents[3] / "openviking/prompts/templates/memory"
    registry.load_from_directory(str(templates))
    ctx = RequestContext(
        user=UserIdentifier(account_id="account-a", user_id="alice"), role=Role.USER
    )
    fs = AsyncMock()
    fs.exists.return_value = False
    fs.read_file.side_effect = FileNotFoundError
    return registry, ctx, fs


def message(text, id="m1", peer=None, role="user"):
    return Message(
        id=id, role=role, peer_id=peer, parts=[TextPart(text)], created_at="2023-05-08T10:00:00Z"
    )


async def extract(setup, messages, types=None, **scope):
    registry, ctx, fs = setup
    context = ExtractContext(messages)
    handler = MemoryIsolationHandler(ctx, context, allowed_memory_types=types, **scope)
    extractor = DeterministicMemoryExtractor(registry, fs, isolation_handler=handler)
    result = await extractor.extract(context.messages, ctx, types, context)
    return result, context


@pytest.mark.asyncio
async def test_three_channels_and_stable_replay(setup_extractor):
    messages = [message("[Caroline]: 我昨天去了北京旅游。")]
    result, context = await extract(setup_extractor, messages)
    assert result.errors == []
    assert {op.memory_type for op in result.upsert_operations} == {"facts", "entities", "events"}
    event = next(op for op in result.upsert_operations if op.memory_type == "events")
    assert event.memory_fields["fact_records"][0]["time"] == "2023-05-07"
    assert "/2023/05/08/" in event.uris[0]
    assert event.memory_fields["ranges"] == "0"
    again, _ = await extract(setup_extractor, messages)
    assert [op.uris for op in again.upsert_operations] == [
        op.uris for op in result.upsert_operations
    ]
    for op in result.upsert_operations:
        assert op.uris and all("/alice/" in uri for uri in op.uris)
        assert all(row["source_turn_ids"] == ["m1"] for row in op.memory_fields["fact_records"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "[Caroline]: 我明天去北京旅游。",
        "[Caroline]: 我去过北京旅游吗？",
        "[Caroline]: 我没有去北京旅游。",
    ],
)
async def test_non_occurrences_are_not_events(setup_extractor, text):
    result, _ = await extract(setup_extractor, [message(text)], {"events"})
    assert not result.upsert_operations


@pytest.mark.asyncio
async def test_disabled_types_and_self_scope(setup_extractor):
    msg = message("[Caroline]: 我昨天去了北京旅游。")
    result, _ = await extract(setup_extractor, [msg], set())
    assert not result.upsert_operations
    result, _ = await extract(setup_extractor, [msg], allow_self=False)
    assert not result.upsert_operations
    registry, _, _ = setup_extractor
    registry.get("facts").enabled = False
    result, _ = await extract(setup_extractor, [msg], {"facts"})
    assert not result.upsert_operations


@pytest.mark.asyncio
async def test_peers_are_partitioned_before_entity_attachment(setup_extractor):
    messages = [
        message("[Caroline]: 我昨天去了北京旅游。", "a", "peer-a"),
        message("[Caroline]: 我今天去了上海旅游。", "b", "peer-b"),
    ]
    result, _ = await extract(setup_extractor, messages, allowed_peer_ids={"peer-a", "peer-b"})
    assert not result.errors
    entities = [op for op in result.upsert_operations if op.memory_type == "entities"]
    assert len(entities) == 2
    for op in entities:
        rows = op.memory_fields["fact_records"]
        assert len(rows) == 1
        source = rows[0]["source_turn_ids"][0]
        assert f"/peers/peer-{source}/" in op.uris[0]
        assert ("北京" in op.memory_fields["content"]) == (source == "a")
    denied, _ = await extract(setup_extractor, messages, allowed_peer_ids={"peer-a"})
    assert all("/peer-b/" not in uri for op in denied.upsert_operations for uri in op.uris)


@pytest.mark.asyncio
async def test_chunks_and_all_text_parts_keep_original_source(setup_extractor):
    msg = message("[Caroline]: " + "我昨天去了北京旅游。" * 80)
    msg.parts.append(TextPart("我目前已完成7篇文章。"))
    result, context = await extract(setup_extractor, [msg], {"facts"})
    assert len(context.messages) > 1
    assert result.upsert_operations
    for op in result.upsert_operations:
        assert op.memory_fields["source_turn_ids"] == ["m1"]
        assert "#chunk" not in str(op.memory_fields["fact_records"])
    prepared = DeterministicMemoryExtractor(
        setup_extractor[0], setup_extractor[2]
    )._prepare_messages(context.messages, context)
    assert "已完成7篇文章" in prepared[0].text


@pytest.mark.asyncio
async def test_entity_read_error_is_reported(setup_extractor):
    setup_extractor[2].read_file.side_effect = TimeoutError("backend unavailable")
    result, _ = await extract(
        setup_extractor, [message("[Caroline]: 我昨天去了北京旅游。")], {"entities"}
    )
    assert result.errors and "backend unavailable" in result.errors[0]
    assert not result.upsert_operations
