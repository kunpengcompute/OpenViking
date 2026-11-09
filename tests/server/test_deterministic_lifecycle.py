"""Source lifecycle integration against the service's real filesystem adapter."""

import asyncio
import json

import pytest

from openviking.message import TextPart
from openviking.retrieve.evidence_retriever import load_index
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config import get_openviking_config


@pytest.mark.asyncio
async def test_deterministic_extract_correct_and_delete_with_service_leases(service, monkeypatch):
    from openviking.utils.spacy_utils import load_spacy_model

    if load_spacy_model("en_core_web_sm") is None:
        pytest.skip("install en_core_web_sm for the model lifecycle integration test")
    monkeypatch.setattr(
        service.sessions._session_compressor,
        "_get_or_create_react",
        lambda **kwargs: pytest.fail("deterministic candidates must not invoke ReAct"),
    )
    config = get_openviking_config()
    monkeypatch.setattr(config.memory, "deterministic_extract_enabled", True)
    ctx = RequestContext(UserIdentifier.the_default_user("test_user"), Role.USER)
    session = service.sessions.session(ctx, "deterministic-lifecycle")
    await session.ensure_exists()
    message = await session.add_message_async(
        "user", [TextPart("I completed 4 articles this year.")]
    )
    archive = session.uri + "/history/archive_001"
    fs = service.viking_fs
    await fs.write_file(archive + "/messages.jsonl", message.to_jsonl() + "\n", ctx=ctx)
    await session._append_to_evidence_index([message], archive)
    await asyncio.wait_for(
        session._session_compressor._extract_user_memories(
            messages=[message],
            ctx=ctx,
            session_id=session.session_id,
            archive_uri=archive,
            allowed_memory_types={"facts", "entities", "events"},
            strict_extract_errors=True,
        ),
        timeout=30,
    )
    result = await service.search.find_evidence(
        "What articles did I complete?",
        ctx,
        strategy="coverage",
        use_memories=True,
        max_supplementary_queries=0,
    )
    assert result["facts"], json.dumps(result, ensure_ascii=False)
    assert result["facts"][0]["source_turn_ids"] == [message.id]
    correction = await asyncio.wait_for(
        service.sessions.correct_message(
            session.session_id, message.id, "I completed 7 articles this year.", ctx
        ),
        timeout=30,
    )
    assert correction["status"] == "completed"
    result = await service.search.find_evidence(
        "What articles did I complete?",
        ctx,
        strategy="coverage",
        use_memories=True,
        max_supplementary_queries=0,
    )
    assert result["facts"] and all("4 articles" not in fact["object"] for fact in result["facts"])
    await asyncio.wait_for(service.sessions.delete(session.session_id, ctx), timeout=30)
    index = await load_index(fs, "viking://user/test_user/.index/evidence.json", ctx)
    assert not index.resolve_sources([message.id])
    result = await service.search.find_evidence(
        "What articles did I complete?",
        ctx,
        strategy="coverage",
        use_memories=True,
        max_supplementary_queries=0,
    )
    assert result["evidence"] == [] and result["facts"] == []


@pytest.mark.asyncio
async def test_evidence_and_correction_routes_forward_authorized_requests(
    client, service, monkeypatch
):
    from unittest.mock import AsyncMock

    search = AsyncMock(return_value={"evidence": [], "facts": [], "diagnostics": {}})
    correction = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(service.search, "find_evidence", search)
    monkeypatch.setattr(service.sessions, "correct_message", correction)
    response = await client.post(
        "/api/v1/search/find_evidence",
        json={"query": "articles", "strategy": "coverage", "use_memories": True},
    )
    assert response.status_code == 200
    assert response.json()["result"]["evidence"] == []
    assert search.await_args.kwargs["strategy"] == "coverage"
    assert search.await_args.kwargs["ctx"].account_id
    response = await client.post(
        "/api/v1/search/find_evidence", json={"query": "articles", "use_memories": True}
    )
    assert response.status_code == 400
    assert search.await_count == 1
    response = await client.put(
        "/api/v1/sessions/chat/messages/m1", json={"content": "corrected text"}
    )
    assert response.status_code == 200
    assert correction.await_args.args[:3] == ("chat", "m1", "corrected text")
    assert correction.await_args.args[3].account_id
