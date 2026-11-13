"""Default recall includes only the caller user and current Agent peer."""

import json
from dataclasses import replace

import httpx
import pytest

from examples.deepagents_ov_context.chat.config import (
    DEFAULT_FIND_READER_CONFIG,
    ChatOVPolicy,
    load_ov_policy,
)
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile, ObservedContextReader
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking_cli.client.identity import OVIdentity
from openviking_cli.retrieve.context import QUOTA_CATEGORIES
from openviking_cli.retrieve.types import ContextType, MatchedContext

from .test_chat_api import parse_events


@pytest.mark.parametrize("scope", ["default", "user_only", "explicit", "empty"])
@pytest.mark.parametrize("search_mode", ["find", "context"])
async def test_recall_uses_session_agent_identity_and_configured_roots(
    resilient_chat,
    monkeypatch,
    scope,
    search_mode,
):
    client, runtime, server = resilient_chat
    runtime.settings = replace(
        runtime.settings,
        reader_config=(
            DEFAULT_FIND_READER_CONFIG if search_mode == "find" else runtime.settings.reader_config
        ),
    )
    if scope == "user_only":
        runtime.settings = replace(
            runtime.settings, ov_policy=ChatOVPolicy(recall_agent_memories=False)
        )
    original = server.handle

    endpoint = "/api/v1/search/find"

    def query_targets(query):
        target = query["target_uri"]
        return target if isinstance(target, list) else [target]

    async def scoped_memory(request):
        if request.url.path != endpoint:
            return await original(request)
        body = json.loads(request.content)
        server.requests.append((request, body))
        user = request.headers["X-OpenViking-User"]
        agent = request.headers["X-OpenViking-Actor-Peer"]

        def result(query):
            return {
                "memories": [
                    {
                        **server.memory,
                        "uri": target.replace("viking://user/", f"viking://user/{user}/", 1)
                        + "/events/record.md",
                        "level": 2,
                        "category": "events",
                        "abstract": f"{agent}: {target}",
                    }
                    for target in query_targets(query)
                ],
                "resources": [],
                "skills": [],
            }

        return httpx.Response(200, json={"status": "ok", "result": result(body)})

    monkeypatch.setattr(server, "handle", scoped_memory)
    for agent in ("mail-agent-a", "mail-agent-b"):
        execution = {"mode": "current"}
        if scope == "explicit":
            execution["target_uris"] = ["viking://user/memories/preferences"]
        if scope == "empty":
            execution["target_uris"] = []
        session = (
            await client.post(
                "/api/sessions",
                json={
                    "user_id": "same-user",
                    "agent_id": agent,
                    "execution": execution,
                },
            )
        ).json()
        assert session["agent_id"] == agent
        start = len(server.requests)
        response = await client.post(
            "/api/chat",
            json={
                "session_id": session["session_id"],
                "message": "继续处理巡检登记",
            },
        )
        events = parse_events(response.text)
        assert events[-1][0] == "done", response.text
        finds = [(req, body) for req, body in server.requests[start:] if req.url.path == endpoint]
        expected = {"viking://user/memories"}
        if scope == "default":
            expected.add(f"viking://user/peers/{agent}/memories")
        elif scope == "explicit":
            expected = {"viking://user/memories/preferences"}
        elif scope == "empty":
            expected = set()
        queries = [body for _, body in finds]
        actual = {target for query in queries for target in query_targets(query)}
        assert actual == expected
        assert len(finds) == int(bool(expected))
        for query in queries:
            assert query["limit"] == 25
            assert query["score_threshold"] == (
                runtime.settings.reader_config.score_threshold if search_mode == "find" else 0.0
            )
            assert not {"recall_policy", "max_tokens", "detail", "quotas"}.intersection(query)
        assert all(
            req.headers["X-OpenViking-User"] == "same-user"
            and req.headers["X-OpenViking-Actor-Peer"] == agent
            and req.headers["X-OpenViking-Account"] == runtime.settings.account_id
            for req, _ in finds
        )
        if not expected:
            continue
        recall = next(
            data
            for event, data in events
            if event == "stage" and data.get("name") == "memory_recall_result"
        )
        assert recall["count"] == len(expected)
        assert all(target in recall["content"] for target in expected)


def test_policy_environment_and_explicit_custom_roots(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OV_CHAT__IO_TIMEOUT=3\nOV_CHAT__RECALL_AGENT_MEMORIES=false\n")
    policy = load_ov_policy(env)
    assert policy.io_timeout == 3 and not policy.recall_agent_memories
    monkeypatch.setenv("OV_CHAT__IO_TIMEOUT", "4")
    assert load_ov_policy(env).io_timeout == 4
    custom = ("viking://user/memories/preferences",)
    assert ChatOVPolicy().recall_targets(custom, "agent-a") == custom


@pytest.mark.parametrize("agent_memories", [True, False])
@pytest.mark.parametrize("mode", ["current", "ordinary", "experience", "context_only"])
async def test_chat_quotas_through_real_reader_sdk_and_search_service(api, agent_memories, mode):
    fs, _, sdk = api
    fs.extra_items = [
        MatchedContext(
            uri=uri, context_type=kind, score=0.99, abstract=f"fixture fact for {kind.value}"
        )
        for uri, kind in (
            ("viking://user/alice/peers/assistant/memories/events/peer.md", ContextType.MEMORY),
            ("viking://resources/manual.md", ContextType.RESOURCE),
            ("viking://user/alice/skills/check/SKILL.md", ContextType.SKILL),
        )
    ]
    defaults = MemoryReaderConfig(
        search_mode="context", recall_quotas=dict.fromkeys(QUOTA_CATEGORIES, 1)
    )
    observations = []
    reader = ObservedContextReader(
        sdk,
        ExecutionProfile(mode=mode).memory_config(defaults),
        scope_policy=ChatOVPolicy(recall_agent_memories=agent_memories),
    )
    reader.observer = lambda event, **fields: observations.append(fields)
    result = await reader.recall(
        "continue the task", identity=OVIdentity("tenant", "alice", "assistant")
    )
    expected = set(QUOTA_CATEGORIES)
    if mode == "ordinary":
        expected.remove("experiences")
    elif mode == "experience":
        expected = {"experiences"}
    elif mode == "context_only":
        expected = set()
    assert result.status == ("success" if expected else "empty")
    assert result.count == len(expected)
    assert all(f'type="{category}"' in result.content for category in expected)
    assert all(
        f'type="{category}"' not in result.content for category in QUOTA_CATEGORIES - expected
    )
    peer = "viking://user/alice/peers/assistant/memories/events/peer.md"
    assert (peer in result.content) == (agent_memories and mode in {"current", "ordinary"})
    assert observations[0]["status"] == result.status
    if mode == "context_only":
        assert not fs.calls
    assert defaults.target_uris is None and defaults.recall_quotas == dict.fromkeys(
        QUOTA_CATEGORIES, 1
    )
