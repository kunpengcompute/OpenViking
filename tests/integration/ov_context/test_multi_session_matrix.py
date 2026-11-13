"""Cross-product isolation and assembly contracts; OV transport is offline."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from typing import Any

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import OVContextConfig, OVIdentity, RunIdentity
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot

from .test_request_identity import FakeOVService, Reader, Writer, build, runtime

pytestmark = pytest.mark.asyncio


async def capture(
    middleware: OpenVikingContextMiddleware,
    state: dict[str, Any],
    identity: RunIdentity,
    template: ModelRequest[Any],
) -> list[BaseMessage]:
    seen: list[BaseMessage] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.extend(request.messages)
        assert "POISON-REQUEST" not in str(request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    response = await middleware.awrap_model_call(
        template.override(
            state=state,
            runtime=runtime(identity),
            messages=[HumanMessage(id="poison", content="POISON-REQUEST")],
        ),
        handler,
    )
    assert response.command is not None
    state.update(response.command.update)
    return seen


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "concurrent"])
@pytest.mark.parametrize("blocks", [False, True], ids=["text", "multimodal"])
async def test_owner_cross_product_two_runs_and_checkpoint_restore(
    env: tuple[Any, ...], parallel: bool, blocks: bool
) -> None:
    service = FakeOVService()
    identities = [
        RunIdentity(account, user, peer, "same-session", "r1")
        for account in ("a", "b")
        for user in ("alice", "bob")
        for peer in ("mail", "doc")
    ]
    identities.append(replace(identities[0], session_id="second-session"))
    markers = {i.session_key: "MARK-" + "-".join(i.session_key) for i in identities}
    states: dict[tuple[str, ...], dict[str, Any]] = {
        i.session_key: {"messages": []} for i in identities
    }
    original_headers = dict(service.http.headers)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[])
    try:
        for round_no in (1, 2):
            # Discard middleware per round; the shared SDK remains borrowed.
            mw, _, _ = build(service)

            async def run(
                base: RunIdentity,
                round_no: int = round_no,
                mw: OpenVikingContextMiddleware = mw,
            ) -> None:
                identity = replace(base, run_id=f"r{round_no}")
                state, marker = states[base.session_key], markers[base.session_key]
                text = f"{marker} round {round_no}"
                user = HumanMessage(
                    id=f"u{round_no}",
                    content=[
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": "https://image.test/a.png"}},
                    ]
                    if blocks
                    else text,
                    additional_kwargs={"trace": marker},
                )
                state["messages"].append(user)
                state.update(await mw.abefore_agent(state, runtime(identity)))
                first = await capture(mw, state, identity, env[4])
                assert [m.id for m in first].count(user.id) == 1
                injected = first[-1]
                prefix = f"[参考记忆]\nreference:{identity.account_id}/{identity.user_id}/{identity.actor_peer_id}"
                if blocks:
                    assert injected.content[0]["text"].startswith(prefix)
                    assert injected.content[1:] == user.content
                else:
                    assert injected.content.startswith(prefix)
                    assert injected.content.endswith(text)
                assert injected.additional_kwargs["trace"] == marker
                assistant = AIMessage(
                    id=f"a{round_no}",
                    content="",
                    tool_calls=[
                        {"id": f"c{round_no}", "name": "lookup", "args": {"marker": marker}}
                    ],
                )
                tool = ToolMessage(
                    id=f"t{round_no}",
                    tool_call_id=f"c{round_no}",
                    content=[{"type": "text", "text": marker}],
                    status="error" if round_no == 2 else "success",
                    artifact={"local": marker},
                )
                state["messages"].extend([assistant, tool])
                raw_before = deepcopy(state["messages"])
                state.update(await mw.aafter_model(state, runtime(identity)))
                second = await capture(mw, state, identity, env[4])
                assert second[-2].model_dump() == assistant.model_dump()
                assert second[-1].model_dump() == tool.model_dump()
                assert state["messages"] == raw_before
                assert str(second).count("[参考记忆]") == 1
                for other in set(markers.values()) - {marker}:
                    assert other not in str(second)
                state["messages"].append(AIMessage(id=f"f{round_no}", content=marker))
                state.update(await mw.aafter_agent(state, runtime(identity)))
                ctx = state["ov_context"]
                assert tuple(ctx["owner"]) == identity.session_key
                assert ctx["write_index"] == len(state["messages"]) == 4 * round_no
                assert ctx["completed_runs"] == round_no
                assert ctx["run_id"] is None and not ctx["recall_done"]
                assert "[参考记忆]" not in str(ctx["messages"])
                assert ctx["messages"] == state["messages"]
                state["ov_context"] = serde.loads_typed(serde.dumps_typed(ctx))

            if parallel:
                await asyncio.gather(*(run(i) for i in identities))
            else:
                for identity in identities:
                    await run(identity)
            assert not mw._ov_session_engine.active
            await mw.aclose()
            assert not service.http.is_closed
        caches = [state["ov_context"]["messages"] for state in states.values()]
        assert len({id(cache) for cache in caches}) == len(identities)
        assert len({id(cache[0]) for cache in caches}) == len(identities)
        assert len(service.sessions) == len(identities)
        # One initial load per session, then checkpoint reuse. Recall runs per Run.
        assert sum(r.url.path.endswith("/context") for r in service.requests) == len(identities)
        assert sum(r.url.path == "/api/v1/search/search" for r in service.requests) == 2 * len(
            identities
        )
        assert dict(service.http.headers) == original_headers
    finally:
        await service.http.aclose()


class VersionedService(FakeOVService):
    def __init__(self) -> None:
        super().__init__()
        self.archives: dict[tuple[str, ...], str] = {}

    async def handle(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle(request)
        if response.is_success and request.method == "GET" and "/sessions/" in request.url.path:
            key = (
                *(
                    request.headers[n]
                    for n in (
                        "X-OpenViking-Account",
                        "X-OpenViking-User",
                        "X-OpenViking-Actor-Peer",
                    )
                ),
                request.url.path.split("/")[4],
            )
            payload = response.json()
            payload["result"]["latest_finished_archive_id"] = self.archives.get(key)
            return httpx.Response(200, json=payload)
        return response


class SnapshotReader(Reader):
    def __init__(self, service: VersionedService) -> None:
        super().__init__(service)
        self.snapshots: dict[tuple[str, ...], ContextSnapshot] = {}

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity: OVIdentity,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        snapshot = self.snapshots.get(identity.session_key(session_id))
        return (
            snapshot
            if snapshot is not None
            else await super().exchange_context(session_id, expected_archive_id, identity=identity)
        )


@pytest.mark.parametrize(
    "bad",
    [None, "missing-user", "changed-user", "duplicate-id", "orphan-tool", "unsupported-system"],
)
async def test_archive_switch_is_atomic_and_does_not_affect_another_user(
    env: tuple[Any, ...], bad: str | None
) -> None:
    from langchain_core.messages import SystemMessage

    service = VersionedService()
    reader = SnapshotReader(service)
    mw = OpenVikingContextMiddleware(
        client=service.client,
        message_writer=Writer(service),
        context_reader=reader,
        context_config=OVContextConfig(context_window=100000),
        token_counter=lambda request: 10,
    )
    alice = RunIdentity("a", "alice", "mail", "same-session", "r")
    bob = replace(alice, user_id="bob")
    states: dict[tuple[str, ...], dict[str, Any]] = {}
    try:
        for identity in (alice, bob):
            state: dict[str, Any] = {
                "messages": [
                    HumanMessage(id="old", content="OLD-RAW"),
                    HumanMessage(id="now", content=f"QUERY-{identity.user_id}"),
                ]
            }
            state.update(await mw.abefore_agent(state, runtime(identity)))
            await capture(mw, state, identity, env[4])
            states[identity.session_key] = state
        state = states[alice.session_key]
        baseline = deepcopy(state)
        bob_baseline = deepcopy(states[bob.session_key])
        user = state["messages"][-1]
        summary = HumanMessage(id="ov_summary:archive_001", content="ALICE-SUMMARY")
        messages: list[BaseMessage] = [summary, user]
        if bad == "missing-user":
            messages = [summary]
        elif bad == "changed-user":
            messages = [summary, HumanMessage(id="now", content="WRONG-QUERY")]
        elif bad == "duplicate-id":
            messages.append(user)
        elif bad == "orphan-tool":
            messages.append(ToolMessage(id="t", content="x", tool_call_id="missing"))
        elif bad == "unsupported-system":
            messages.insert(0, SystemMessage(id="sys", content="unexpected"))
        service.archives[alice.session_key] = "archive_001"
        reader.snapshots[alice.session_key] = ContextSnapshot(messages, "archive_001")
        if bad not in (None, "missing-user"):
            with pytest.raises(ContextReadError):
                await capture(mw, state, alice, env[4])
            assert state == baseline  # No unpublished cache or cursor mutation.
            reader.snapshots[alice.session_key] = ContextSnapshot([summary, user], "archive_001")
        assembled = await capture(mw, state, alice, env[4])
        assert [m.id for m in assembled] == [summary.id, user.id]
        assert "OLD-RAW" not in str(assembled)  # Raw history must not be re-appended.
        assert str(assembled).count("[参考记忆]") == 1
        assert state["ov_context"]["archive_id"] == "archive_001"
        assert state["messages"] == baseline["messages"]
        await capture(mw, states[bob.session_key], bob, env[4])
        assert states[bob.session_key] == bob_baseline
    finally:
        await mw.aclose()
        await service.http.aclose()
