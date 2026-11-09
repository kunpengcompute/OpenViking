import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.retrieve.evidence_retriever import EvidenceIndex
from openviking.server.identity import RequestContext, Role
from openviking.server.routers.search import FindEvidenceRequest
from openviking.service.search_service import SearchService
from openviking_cli.exceptions import ConflictError, InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier


@pytest.fixture
def setup():
    ctx = RequestContext(UserIdentifier("tenant", "alice"), Role.USER)
    fs = SimpleNamespace(
        read_file=AsyncMock(return_value=json.dumps(EvidenceIndex().to_dict())),
        _get_embedder=Mock(return_value=None),
    )
    return SearchService(fs), fs, ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["legacy", "coverage"])
async def test_pending_source_correction_blocks_search(setup, strategy):
    service, fs, ctx = setup
    index = EvidenceIndex()
    index.pending_source_updates = ["m1"]
    fs.read_file.return_value = json.dumps(index.to_dict())
    with pytest.raises(ConflictError):
        await service.find_evidence("question", ctx, strategy=strategy)


@pytest.mark.asyncio
async def test_authorized_load_is_repeated_and_uses_request_context(setup):
    service, fs, ctx = setup
    assert (await service.find_evidence("question", ctx, strategy="coverage"))["evidence"] == []
    await service.find_evidence("question", ctx)
    assert fs.read_file.await_count == 2
    assert fs.read_file.call_args.kwargs["ctx"] is ctx
    assert fs.read_file.call_args.kwargs["uri"] == "viking://user/alice/.index/evidence.json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"candidate_limit": 100},
        {"strategy": "unknown"},
        {"use_graph": True},
        {"user_uri": "viking://user/alice/memories"},
    ],
)
async def test_service_validates_non_http_calls(setup, kwargs):
    service, _, ctx = setup
    with pytest.raises(InvalidArgumentError):
        await service.find_evidence("question", ctx, **kwargs)


def test_request_budget_and_auxiliary_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FindEvidenceRequest(candidate_limit=51)
    with pytest.raises(ValidationError):
        FindEvidenceRequest(use_events=True)
    assert FindEvidenceRequest(strategy="coverage", use_events=True).use_events
