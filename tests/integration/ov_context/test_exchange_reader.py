"""Run the production exchange reader through Middleware's snapshot contract."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy
from openviking.integrations.langchain.ov_context import (
    ContextBudgetExceeded,
    OVContext,
    OVContextConfig,
    OVIdentity,
)
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle
from openviking_cli.retrieve.types import FindResult
from tests.ov_context_support.fixtures import counter, publish
from tests.ov_context_support.mock_writer import MockMessageWriter
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_graph import CallContext, ToolLoopModel, read_mail
from .test_session_hooks import call, runtime


class TypedTransportClient(TransportSDKAdapter):
    async def find(self, query, target_uri="", limit=10, score_threshold=None, *, identity=None):
        result = await super().find(query, target_uri, limit, score_threshold, identity=identity)
        return FindResult.from_dict(result)


@pytest_asyncio.fixture
async def exchange_env(env):
    server, transport, _, _, _, _ = env
    controls = {}

    async def handle(request):
        response = await server.handle(request)
        if not response.is_success:
            return response
        data = response.json()
        result = data["result"]
        event = request.url.path.rsplit("/", 1)[-1]
        if event == "context":
            # Emulate the documented terminal-version and native-ID server contract.
            result["latest_finished_archive_id"] = controls.get(
                "context_boundary", server.archive_id
            )
            for record in result["messages"]:
                ids = server.native_host_ids.get(record["id"])
                if ids:
                    record["id"] = ids[0]
        elif event == "commit" and result.get("archived") is True:
            controls["commit_count"] = (
                controls.get(
                    "commit_count", int(server.archive_id.split("_")[1]) if server.archive_id else 0
                )
                + 1
            )
        elif request.url.path.startswith("/api/v1/tasks/"):
            result["status"] = controls.get("task_status", "completed")
            result["result"] = controls.get("task_result", {})
        elif request.method == "GET" and isinstance(result, dict) and result.get("session_id"):
            if not controls.get("omit_terminal_metadata"):
                result["latest_finished_archive_id"] = controls.get(
                    "metadata_boundary", server.archive_id
                )
            if not controls.get("omit_pending_tokens"):
                result["pending_tokens"] = controls.get("pending_tokens", 0)
            else:
                result.pop("pending_tokens", None)
            result["commit_count"] = controls.get(
                "commit_count", int(server.archive_id.split("_")[1]) if server.archive_id else 0
            )
        return httpx.Response(response.status_code, json=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport.http = http
        client = TypedTransportClient(transport, "p")
        config = OVContextConfig(context_window=100000, output_reserve=1000)
        writer = MockMessageWriter(
            AsyncClientHandle(SimpleNamespace(client=client)), "p", server.native_host_ids
        )
        reader = OVContextReader(client, MemoryReaderConfig(search_mode="find"))
        middleware = OpenVikingContextMiddleware(
            client=client,
            commit_config=None,  # The tests below explicitly select legacy policies.
            message_writer=writer,
            context_reader=reader,
            context_config=config,
            session_id_resolver=lambda state, rt: rt.context.identity.session_id,
            peer_id="p",
            token_counter=counter,
        )
        yield middleware, env, controls


async def start(exchange_env, history=False):
    middleware, env, controls = exchange_env
    raw = [HumanMessage(id="current", content="Read the mail", additional_kwargs={"source": "app"})]
    if history:
        raw[:0] = [
            HumanMessage(id="past", content="Open mailbox"),
            AIMessage(id="past-a", content="Opened"),
        ]
    state = {"messages": raw}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    return middleware, env, controls, state


async def test_real_reader_initial_load_loop_reuse_and_new_snapshot(exchange_env):
    middleware, env, _, state = await start(exchange_env, history=True)
    assert not hasattr(middleware._ov_session_engine.context_reader, "refresh")
    assert env[0].counts["context"] == 1
    raw = deepcopy(state["messages"])
    await call(middleware, state, env[4], runtime(env))
    view = state["ov_context"]["messages"][-1]
    await call(middleware, state, env[4], runtime(env))
    assert env[0].counts["context"] == 1 and env[0].counts["find"] == 1
    publish(env[0], env[3], {"past", "past-a"}, "Mailbox is open")
    output = await call(middleware, state, env[4], runtime(env))
    assert [m.id for m in output] == ["ov_summary:archive_001", "current"]
    assert output[-1] == view and output[-1] is not view
    assert state["ov_context"]["messages"][-1] == view
    assert state["ov_context"]["messages"][-1] is view
    assert view.additional_kwargs["source"] == "app"
    assert state["messages"] == raw and state["ov_context"]["write_index"] == 3
    assert state["ov_context"]["archive_id"] == "archive_001"
    await call(middleware, state, env[4], runtime(env))
    assert env[0].counts["context"] == 2 and env[0].counts["find"] == 1


async def test_snapshot_boundary_mismatch_keeps_checkpoint(exchange_env):
    middleware, env, controls, state = await start(exchange_env)
    await call(middleware, state, env[4], runtime(env))
    previous = deepcopy(state)
    controls.update(metadata_boundary="archive_002", context_boundary="archive_001")
    with pytest.raises(ContextReadError, match="regressed"):
        await call(middleware, state, env[4], runtime(env))
    assert state == previous


@pytest.mark.parametrize("boundary", [None, "archive_000"])
async def test_regressed_terminal_metadata_is_rejected_before_context_read(exchange_env, boundary):
    middleware, env, controls, state = await start(exchange_env)
    env[0].archive_id = "archive_001"
    env[0].overview = "History"
    await call(middleware, state, env[4], runtime(env))
    previous = deepcopy(state)
    reads = env[0].counts["context"]
    controls["metadata_boundary"] = boundary
    with pytest.raises(ContextReadError, match="regressed"):
        await call(middleware, state, env[4], runtime(env))
    assert state == previous and env[0].counts["context"] == reads


async def test_pending_token_policy_and_running_commit_do_not_read_context(exchange_env):
    middleware, env, controls, state = await start(exchange_env, history=True)
    controls.update(
        omit_terminal_metadata=True, pending_tokens=100, commit_count=99, task_status="running"
    )
    middleware.commit_policy = OpenVikingCommitPolicy(
        mode="pending_tokens", pending_token_threshold=50
    )
    await call(middleware, state, env[4], runtime(env))
    assert env[0].counts["commit"] == 1 and state["ov_context"]["pending_commit"]
    await call(middleware, state, env[4], runtime(env))
    assert env[0].counts["context"] == 1
    assert env[0].counts["commit"] == 1
    assert state["ov_context"]["archive_id"] is None


async def test_finished_task_triggers_exchange_without_terminal_metadata(exchange_env):
    middleware, env, controls, state = await start(exchange_env, history=True)
    await call(middleware, state, env[4], runtime(env))
    state["ov_context"]["pending_commit"] = {"task_id": "t1", "archive_id": None}
    controls.update(
        omit_terminal_metadata=True,
        task_status="completed",
        task_result={"archive_uri": "viking://s/history/archive_001"},
    )
    publish(env[0], env[3], {"past", "past-a"})
    await call(middleware, state, env[4], runtime(env))
    assert state["ov_context"]["archive_id"] == "archive_001"
    assert state["ov_context"]["pending_commit"] is None
    assert env[0].counts["context"] == 2


async def test_failed_archive_keeps_applied_cache_and_records_failed_submission(exchange_env):
    middleware, env, controls, state = await start(exchange_env)
    await call(middleware, state, env[4], runtime(env))
    state["ov_context"]["pending_commit"] = {
        "task_id": "t1",
        "archive_id": None,
        "target_archive_id": "archive_001",
    }
    controls.update(task_status="failed")
    env[0].archive_id = "archive_001"
    env[0].overview = "Previous successful overview"
    previous = deepcopy(state["ov_context"])
    with pytest.raises(ContextBudgetExceeded, match="compression task did not complete"):
        await call(middleware, state, env[4], runtime(env))
    assert state["ov_context"]["messages"] == previous["messages"]
    assert state["ov_context"]["archive_id"] is None
    recovered = middleware._context(state, runtime(env))
    assert recovered.pending_commit is None


async def test_snapshot_user_validation_failure_does_not_clear_pending_commit(exchange_env):
    middleware, env, controls, state = await start(exchange_env)
    await call(middleware, state, env[4], runtime(env))
    state["ov_context"]["pending_commit"] = {"task_id": "t1", "archive_id": None}
    previous = deepcopy(state)
    controls["task_status"] = "completed"
    env[0].archive_id = "archive_001"
    env[0].session_messages["s"][0]["parts"][0]["text"] = "Wrong user input"
    with pytest.raises(ContextBudgetExceeded, match="ContextReadError") as raised:
        await call(middleware, state, env[4], runtime(env))
    assert isinstance(raised.value.__cause__, ContextReadError)
    assert "does not match" in str(raised.value.__cause__)
    assert state == previous


async def test_commit_tracks_target_and_excludes_summary_from_retained_count(exchange_env):
    middleware, env, _, state = await start(exchange_env, history=True)
    context = OVContext.from_state(state["ov_context"])
    context.messages.insert(0, HumanMessage(id="ov_summary:archive_001", content="History"))
    context.archive_id = "archive_001"
    context.completed_runs = 4
    client = middleware._connection.client
    client.commit_session = AsyncMock(
        return_value={
            "archived": True,
            "task_id": "t1",
            "archive_uri": "viking://s/history/archive_002",
        }
    )
    assert middleware._ov_session_engine.retained_tail(context) == (1, 2)
    await middleware._ov_session_engine.commit(context, identity=context.identity)
    client.commit_session.assert_awaited_once_with(
        session_id="s", keep_recent_count=1, identity=OVIdentity("a", "u", "p")
    )
    assert context.pending_commit == {
        "task_id": "t1",
        "archive_id": "archive_001",
        "target_archive_id": "archive_002",
    }
    assert context.completed_runs == 0


async def test_real_agent_graph_with_exchange_reader_does_not_leak_injection(exchange_env):
    middleware, env, _ = exchange_env
    model = ToolLoopModel()
    agent = create_agent(
        model=model, tools=[read_mail], middleware=[middleware], context_schema=CallContext
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(id="m1", content="Read the mail")]},
        context=CallContext(env[3]),
    )
    assert result["messages"][-1].content == "Mail summarized"
    assert len(model.inputs) == 2 and env[0].counts["context"] == 1
    assert env[0].counts["find"] == 1
    assert not any(m.additional_kwargs.get("ov_context_injected") for m in result["messages"])
    assert not middleware._ov_session_engine.active


async def test_missing_pending_tokens_remains_unknown_for_injected_policy(
    exchange_env: tuple[Any, ...],
) -> None:
    from openviking.integrations.langchain.ov_context.commit import CommitConfig

    middleware, env, controls, state = await start(exchange_env, history=True)
    controls["omit_pending_tokens"] = True
    engine = middleware._ov_session_engine
    engine.session_commit_policy = OpenVikingCommitPolicy(
        CommitConfig(100, 0.5, 1, 1), middleware._connection.client
    )
    context = OVContext.from_state(state["ov_context"])
    context.completed_runs = 5
    result = await engine.check_summary(context, identity=context.identity)
    assert result.pending_tokens is None
    assert not await engine.commit(
        context, pending_tokens=result.pending_tokens, identity=context.identity
    )
    assert env[0].counts["commit"] == 0


async def test_exchange_reader_preserves_commit_progress_across_model_failure(
    exchange_env: tuple[Any, ...],
) -> None:
    middleware, env, controls, state = await start(exchange_env, history=True)
    controls.update(omit_terminal_metadata=True, pending_tokens=100, task_status="running")
    middleware.commit_policy = OpenVikingCommitPolicy(
        mode="pending_tokens", pending_token_threshold=50
    )
    before = deepcopy(state)

    async def failing(request: ModelRequest[Any]) -> ModelResponse[Any]:
        raise RuntimeError("model failed after commit")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="model failed after commit"):
            await middleware.awrap_model_call(
                env[4].override(state=state, runtime=runtime(env)), failing
            )
        assert state == before
    assert env[0].counts["commit"] == 1
    await call(middleware, state, env[4], runtime(env))
    assert state["ov_context"]["pending_commit"]["task_id"] == "t1"
    await call(middleware, state, env[4], runtime(env))
    assert env[0].counts["commit"] == 1
    assert not middleware._ov_session_engine._commit_progress
