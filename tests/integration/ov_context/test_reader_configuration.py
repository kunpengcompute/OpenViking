"""Reader configuration drives recalled model input without middleware overrides."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import Metrics, OVContextConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from tests.ov_context_support.fixtures import counter, mock_io
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_session_hooks import call, runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "expected_count", "expected_searches"),
    [
        pytest.param({}, 2, 1, id="reader-limits-override-sync-defaults"),
        pytest.param({"enabled": False}, 0, 0, id="ordinary-disabled"),
        pytest.param({"recall_limit": 1}, 1, 1, id="final-count"),
        pytest.param({"candidate_limit": 0}, 0, 0, id="no-candidates"),
        pytest.param({"score_threshold": 0.95}, 0, 1, id="score-filter"),
        pytest.param({"max_injected_chars": 0}, 0, 0, id="zero-budget"),
        pytest.param({"max_injected_chars": 10}, 0, 1, id="insufficient-budget"),
        pytest.param({"prefer_abstract": False}, 1, 1, id="identical-full-bodies-deduplicated"),
        pytest.param(
            {"enabled": False, "experience_enabled": True},
            1,
            1,
            id="independent-experience-config",
        ),
    ],
)
async def test_reader_config_controls_recall_and_model_input(
    env: tuple[Any, ...], changes: dict[str, Any], expected_count: int, expected_searches: int
) -> None:
    server, transport, _, _, request, _ = env
    server.recalled = [
        {
            **server.memory,
            "uri": f"viking://user/u/memories/preferences/{name}.md",
            "abstract": f"Preference {name}",
            "score": score,
        }
        for name, score in (("first", 0.9), ("second", 0.7), ("low", 0.05))
    ] + [
        {
            **server.memory,
            "uri": "viking://user/u/memories/experiences/fix.md",
            "abstract": "Repair workflow",
            "category": "experiences",
            "score": 0.95,
        }
    ]
    config = replace(
        MemoryReaderConfig(
            search_mode="find",
            target_uris=("viking://user/u/memories",),
            candidate_limit=3,
            recall_limit=2,
            score_threshold=0.2,
            ranking="score",
            context_token_budget=741,
            experience_target_uris=("viking://user/u/memories/experiences",),
        ),
        **changes,
    )
    client = TransportSDKAdapter(transport, "p")
    samples: list[tuple[str, dict[str, Any]]] = []
    middleware = OpenVikingContextMiddleware(
        client=client,
        commit_config=None,
        # These options belong to the synchronous interface, not the injected reader.
        limit=1,
        score_threshold=0.99,
        target_uri="viking://wrong-sync-scope",
        context_config=OVContextConfig(context_window=100000, output_reserve=1000),
        token_counter=counter,
        metrics=Metrics(lambda event, **data: samples.append((event, data))),
        **mock_io(client, reader_config=config, peer_id="p"),
    )
    state = {"messages": [HumanMessage(id="current", content="Fix the error in repo service.py")]}
    original = deepcopy(state["messages"])
    rt = runtime(env)
    state.update(await middleware.abefore_agent(state, rt))
    assembled = await call(middleware, state, request, rt)
    await call(middleware, state, request, rt)

    recalls = [data for event, data in samples if event == "memory.recall"]
    assert len(recalls) == 1 and recalls[0]["count"] == expected_count
    assert server.counts["find"] == expected_searches
    context_reads = [req for req, _ in server.requests if req.url.path.endswith("/context")]
    assert len(context_reads) == 1
    assert context_reads[0].url.params["token_budget"] == "741"
    assert state["messages"] == original
    assert state["ov_context"]["recall_done"]
    content = str(assembled[0].content)
    assert ("[参考记忆]" in content) == bool(expected_count)
    assert state["ov_context"]["recall_status"] == ("success" if expected_count else "empty")
    assert len(content) <= len(original[0].content) + config.max_injected_chars + 100
    searches = [body for req, body in server.requests if req.url.path.endswith("/find")]
    for body in searches:
        experience = config.experience_enabled
        assert body["target_uri"] == list(
            config.experience_target_uris if experience else config.target_uris
        )
        assert body["limit"] == (
            config.experience_candidate_limit if experience else config.candidate_limit
        )
    if expected_count and not config.prefer_abstract:
        assert "Full preference text" in content
        # Both candidates must be read to establish that their complete bodies match.
        assert server.counts["read"] == 2
