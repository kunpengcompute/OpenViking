# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Execution evidence may belong to peers; resulting memories belong to the session user."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import (
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdateResult
from openviking_cli.session.user_id import UserIdentifier


def context() -> RequestContext:
    return RequestContext(user=UserIdentifier(account_id="acct", user_id="alice"), role=Role.USER)


def schema(kind: str = "trajectories") -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type=kind,
        directory="viking://user/{{ user_space }}/memories/" + kind,
        filename_template="pdf.md",
        agent_only=True,
    )


def operation(kind: str, ranges: str | None = "0") -> ResolvedOperation:
    fields = {"peer_id": "other-peer", "user_id": "other-user"}
    if ranges is not None:
        fields["ranges"] = ranges
    return ResolvedOperation(memory_type=kind, memory_fields=fields, uris=[])


@pytest.mark.parametrize("kind", ["trajectories", "experiences"])
@pytest.mark.parametrize("ranges", ["0", None])
def test_execution_memories_use_session_user_without_rewriting_peer_evidence(
    kind: str,
    ranges: str | None,
) -> None:
    messages = [Message(id="a1", role="assistant", peer_id="mail", parts=[TextPart("Read PDF")])]
    extract = ExtractContext(messages)
    handler = MemoryIsolationHandler(context(), extract, session_owned=True)
    op = operation(kind, ranges)
    assert handler.calculate_memory_uris(schema(kind), op, extract) == [
        f"viking://user/alice/memories/{kind}/pdf.md"
    ]
    assert messages[0].peer_id == "mail"
    assert op.memory_fields["user_id"] == "alice"
    assert "peer_id" not in op.memory_fields
    assert handler.get_read_scope().peer_ids == []
    assert handler.render_schema_directories(schema(kind)) == [
        f"viking://user/alice/memories/{kind}"
    ]


@pytest.mark.parametrize("ranges", ["0", "not-a-range", "99", None])
def test_execution_scope_does_not_bypass_disabled_self_memory(ranges: str | None) -> None:
    extract = ExtractContext(
        [Message(id="a1", role="assistant", peer_id="mail", parts=[TextPart("Read PDF")])]
    )
    handler = MemoryIsolationHandler(
        context(), extract, allow_self=False, allowed_peer_ids={"mail"}, session_owned=True
    )
    assert handler.calculate_memory_uris(schema(), operation("trajectories", ranges), extract) == []


@pytest.mark.parametrize("ranges", ["not-a-range", "99", ""])
def test_execution_scope_requires_valid_nonempty_evidence(ranges: str) -> None:
    extract = ExtractContext(
        [Message(id="a1", role="assistant", peer_id="mail", parts=[TextPart("Read PDF")])]
    )
    handler = MemoryIsolationHandler(context(), extract, session_owned=True)
    assert handler.calculate_memory_uris(schema(), operation("trajectories", ranges), extract) == []


def test_participant_memory_keeps_existing_peer_policy() -> None:
    extract = ExtractContext(
        [Message(id="a1", role="assistant", peer_id="mail", parts=[TextPart("Read PDF")])]
    )
    op = operation("trajectories")
    handler = MemoryIsolationHandler(context(), extract)
    assert handler.calculate_memory_uris(schema(), op, extract) == []
    handler = MemoryIsolationHandler(
        context(), extract, allow_self=False, allowed_peer_ids={"mail"}
    )
    assert handler.calculate_memory_uris(schema(), op, extract) == [
        "viking://user/alice/peers/mail/memories/trajectories/pdf.md"
    ]


async def test_compressor_phase_resolves_peer_only_ranges_before_applying(monkeypatch) -> None:
    """Real phase + URI resolution: catches forgetting to enable session ownership."""
    messages = [Message(id="a1", role="assistant", peer_id="mail", parts=[TextPart("Read PDF")])]
    extract = ExtractContext(messages)
    provider = SimpleNamespace(
        prepare_extraction_messages=AsyncMock(),
        get_extract_context=lambda: extract,
        _get_registry=lambda: object(),
        get_memory_schemas=lambda _: [schema()],
    )
    config = SimpleNamespace(vlm=SimpleNamespace(get_vlm_instance=lambda: object()))
    monkeypatch.setattr("openviking.session.compressor_v2.get_openviking_config", lambda: config)
    monkeypatch.setattr("openviking.session.compressor_v2.get_viking_fs", lambda: None)

    class Loop:
        def __init__(self, **kwargs):
            self.handler = kwargs["isolation_handler"]

        async def run(self):
            op = operation("trajectories")
            op.uris = self.handler.calculate_memory_uris(schema(), op, extract)
            return ResolvedOperations(
                upsert_operations=[op], delete_file_contents=[], errors=[]
            ), []

    async def apply(ops, ctx, **kwargs):
        assert ctx.user.user_id == "alice"
        assert ops.upsert_operations[0].uris == ["viking://user/alice/memories/trajectories/pdf.md"]
        result = MemoryUpdateResult()
        result.written_uris = list(ops.upsert_operations[0].uris)
        return result

    monkeypatch.setattr("openviking.session.compressor_v2.ExtractLoop", Loop)
    compressor = SessionCompressorV2(vikingdb=None)
    monkeypatch.setattr(compressor, "_resolve_supersedes", AsyncMock(return_value={}))
    monkeypatch.setattr(
        compressor, "_get_or_create_updater", lambda *_: SimpleNamespace(apply_operations=apply)
    )
    result = await compressor._run_extract_phase(
        provider, messages, context(), True, "trajectory", allowed_memory_types={"trajectories"}
    )
    assert result[0] == ["viking://user/alice/memories/trajectories/pdf.md"]
