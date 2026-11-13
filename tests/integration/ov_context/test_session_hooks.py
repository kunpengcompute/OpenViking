"""Reference-design examples: state cache, synchronous writes and summary versions."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from openviking.integrations._message_content import LANGCHAIN_CONTENT_MIME, encode_tool_content
from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ContextAssembler,
    OVContext,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking.message import Message
from openviking.message.part import part_from_dict
from tests.ov_context_support.fixtures import counter, mock_io, publish
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter
from tests.ov_context_support.session_view import _native_messages
from tests.ov_context_support.transport import OVHTTPError

from .test_graph import CallContext, ToolLoopModel, read_mail

pytestmark = pytest.mark.asyncio


def middleware(
    env: tuple[Any, ...], *, reader_config: MemoryReaderConfig | None = None, **kwargs: Any
) -> OpenVikingContextMiddleware:
    _, transport, _, _, _, _ = env
    from openviking.integrations.langchain.ov_context.commit import CommitConfig

    keep = kwargs.pop("keep_recent_count", 0)
    runs = kwargs.pop("commit_every_runs", None)
    commit = CommitConfig(100000, 0.7, runs, keep) if runs is not None else None
    if runs is not None:
        env[0].session_meta_hook = lambda sid: {"pending_tokens": 100}
    config = OVContextConfig(context_window=100000, output_reserve=1000, **kwargs)
    client = TransportSDKAdapter(transport, "p")
    return OpenVikingContextMiddleware(
        client=client,
        commit_config=commit,
        context_config=config,
        peer_id="p",
        token_counter=counter,
        **mock_io(client, reader_config=reader_config, peer_id="p"),
    )


def runtime(env, run_id="r"):
    return SimpleNamespace(context=CallContext(replace(env[3], run_id=run_id)))


async def start(
    env: tuple[Any, ...], messages: list[BaseMessage] | None = None, **config: Any
) -> tuple[OpenVikingContextMiddleware, dict[str, Any]]:
    mw = middleware(env, **config)
    state = {"messages": messages or env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    return mw, state


async def call(mw, state, request, rt):
    captured = []

    async def handler(updated):
        captured.extend(updated.messages)
        return ModelResponse(result=[AIMessage(id="response", content="ok")])

    result = await mw.awrap_model_call(request.override(state=state, runtime=rt), handler)
    state.update(result.command.update)
    return captured


async def test_cache_instantiated_in_state_and_no_objects_or_clients_in_checkpoint(env):
    mw, state = await start(env)
    context = OVContext.from_state(state["ov_context"])
    assert context.session_id == "s" and context.run_id == "r"
    assert context.write_index == 1 and context.messages == state["messages"]
    serde = JsonPlusSerializer(allowed_msgpack_modules=[])
    restored = serde.loads_typed(serde.dumps_typed(state["ov_context"]))
    assert restored["messages"] == context.messages
    assert isinstance(restored["messages"][0], HumanMessage)
    assert not any("client" in key or "lock" in key for key in restored)


async def test_model_assembler_does_not_read_raw_state_or_request_messages(env):
    mw, state = await start(env, reader_config=MemoryReaderConfig(enabled=False))
    poisoned = env[4].override(messages=[HumanMessage(id="poison", content="Must not be read")])
    output = await call(mw, state, poisoned, runtime(env))
    assert [m.id for m in output] == ["m1"]
    assert "Must not be read" not in str(output)


async def test_ten_model_calls_use_one_context_read_and_one_recall_attempt(env):
    mw, state = await start(env)
    for _ in range(10):
        await call(mw, state, env[4], runtime(env))
    assert env[0].counts["context"] == 1
    assert env[0].counts["find"] == 1
    assert state["ov_context"]["recall_done"]
    assert state["messages"][0].content == "Read the mail"


@pytest.mark.parametrize("result", ["success", "empty", "timeout", "error"])
async def test_recall_all_outcomes_are_attempted_once_per_run(env, result):
    server = env[0]
    config = {"reader_config": MemoryReaderConfig(search_mode="find", recall_timeout=0.01)}
    if result == "empty":
        server.recalled = []
    elif result == "error":
        server.error["find"] = 503
    elif result == "timeout":

        async def slow():
            await asyncio.sleep(0.05)

        server.search_hook = slow
    mw, state = await start(env, **config)
    for _ in range(3):
        await call(mw, state, env[4], runtime(env))
    assert state["ov_context"]["recall_status"] == result
    assert server.counts["find"] == 1


async def test_two_runs_keep_session_cache_cursor_but_reset_recall(env):
    mw, state = await start(env)
    await call(mw, state, env[4], runtime(env))
    state.update(await mw.aafter_agent(state, runtime(env)))
    assert state["ov_context"]["run_id"] is None
    assert state["ov_context"]["messages"][0] == state["messages"][0]
    first = state["ov_context"]["messages"][0]
    state["messages"].append(HumanMessage(id="u2", content="Follow-up"))
    state.update(await mw.abefore_agent(state, runtime(env, "r2")))
    assert state["ov_context"]["messages"][0] is first
    await call(mw, state, env[4], runtime(env, "r2"))
    assert env[0].counts["find"] == 2
    assert env[0].counts["context"] == 1
    assert state["ov_context"]["write_index"] == 2
    assert len(env[0].messages) == 2


async def test_native_message_format_metadata_multimodal_and_tool_pair_are_preserved(env):
    messages = [
        HumanMessage(
            id="u",
            name="user",
            content=[
                {"type": "text", "text": "Read mail"},
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            ],
            additional_kwargs={"trace": "u"},
            response_metadata={"source": "user"},
        ),
        AIMessage(
            id="a",
            name="agent",
            content="Reading",
            additional_kwargs={"trace": "a"},
            response_metadata={"model": "offline"},
            tool_calls=[{"id": "c1", "name": "read_mail", "args": {}}],
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        ),
        ToolMessage(
            id="t",
            name="read_mail",
            tool_call_id="c1",
            status="success",
            content=[
                {"type": "text", "text": "Mail"},
                {"type": "image_url", "image_url": {"url": "https://example.test/b.png"}},
            ],
            artifact={"file": "mail.md"},
            additional_kwargs={"trace": "t"},
        ),
    ]
    original = deepcopy(messages)
    mw, state = await start(env, messages, reader_config=MemoryReaderConfig(enabled=False))
    output = await call(mw, state, env[4], runtime(env))
    assert [type(m) for m in output] == [HumanMessage, AIMessage, ToolMessage]
    assert [m.model_dump() for m in output] == [m.model_dump() for m in original]
    assert state["messages"] == original
    assert output[1].tool_calls[0]["id"] == output[2].tool_call_id
    # Force an actual OV context refresh and retain native metadata from the cache.
    refreshed = OVContext.from_state(state["ov_context"])
    await mw._ov_session_engine.refresh(refreshed, identity=refreshed.identity)
    assert [m.model_dump() for m in refreshed.messages] == [m.model_dump() for m in original]


async def test_multimodal_tool_payload_roundtrips_real_message_storage():
    content = [
        {"type": "text", "text": "截图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    encoded, mime = encode_tool_content(content)
    assert mime == LANGCHAIN_CONTENT_MIME
    stored = Message(
        id="native",
        role="user",
        parts=[
            part_from_dict(
                {
                    "type": "tool",
                    "tool_id": "call",
                    "tool_name": "browser",
                    "tool_status": "completed",
                    "tool_output": encoded,
                    "tool_output_mime_type": mime,
                }
            )
        ],
    )
    restored = _native_messages(Message.from_dict(json.loads(stored.to_jsonl())).to_dict())[0]
    assert isinstance(restored, ToolMessage) and restored.content == content


async def test_synchronous_write_completes_before_cache_and_cursor_advance(env):
    server = env[0]
    entered, release = asyncio.Event(), asyncio.Event()

    async def block():
        entered.set()
        await release.wait()

    server.batch_hook = block
    mw = middleware(env)
    state = {"messages": env[4].messages}
    task = asyncio.create_task(mw.abefore_agent(state, runtime(env)))
    await entered.wait()
    assert "ov_context" not in state
    release.set()
    state.update(await task)
    assert state["ov_context"]["write_index"] == 1
    assert len(state["ov_context"]["messages"]) == 1
    assert server.counts["tasks"] == 0  # No task discovery or write-confirmation polling.


async def test_native_synchronous_batch_response_is_supported(env):
    mw = middleware(env)
    adapter = mw._connection.client
    original = adapter.batch_add_messages

    async def native(session_id, messages, *, identity=None):
        result = await original(session_id, messages, identity=identity)
        return {"added": result["added"], "message_count": result["message_count"]}

    adapter.batch_add_messages = native
    state = {"messages": env[4].messages}
    state.update(await mw.abefore_agent(state, runtime(env)))
    assert state["ov_context"]["write_index"] == 1
    assert len(state["ov_context"]["messages"]) == 1


async def test_failed_write_does_not_publish_cursor_or_cache(env):
    mw, state = await start(env)
    before = deepcopy(state["ov_context"])
    state["messages"].append(AIMessage(id="new", content="answer"))
    env[0].error["batch"] = 503
    with pytest.raises(OVHTTPError):
        await mw.aafter_model(state, runtime(env))
    assert state["ov_context"] == before


async def test_archive_version_replaces_covered_originals_without_host_reinsertion(env):
    raw = [
        HumanMessage(id="h0", content="Open mailbox"),
        AIMessage(id="a0", content="Mail opened"),
        HumanMessage(id="u1", content="Read target mail"),
    ]
    mw, state = await start(env, raw)
    await call(mw, state, env[4], runtime(env))
    publish(env[0], env[3], {"h0", "a0"}, "Mailbox is open")
    before = deepcopy(state["messages"])
    output = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in output] == ["ov:summary", "u1"]
    assert state["ov_context"]["archive_id"] == "archive_001"
    assert state["ov_context"]["write_index"] == 3
    assert state["messages"] == before
    assert "Open mailbox" not in str(output)
    assert "参考记忆" in output[-1].content
    await call(mw, state, env[4], runtime(env))
    assert env[0].counts["context"] == 2 and env[0].counts["find"] == 1


async def test_version_regression_or_incomplete_summary_keeps_prior_cache(env):
    mw, state = await start(env)
    before = deepcopy(state["ov_context"])
    env[0].overview = "Invalid snapshot"
    env[0].history = [
        {
            "id": "orphan",
            "role": "user",
            "parts": [
                {
                    "type": "tool",
                    "tool_id": "orphan",
                    "tool_name": "read",
                    "tool_status": "completed",
                }
            ],
        }
    ]
    env[0].archive_id = "archive_001"
    with pytest.raises(ValueError):
        await call(mw, state, env[4], runtime(env))
    assert state["ov_context"] == before


@pytest.mark.parametrize("status", [401, 403, 503])
async def test_initial_context_failure_policy_and_auth_fail_closed(env, status):
    env[0].error["context"] = status
    mw = middleware(env, context_failure="host", reader_config=MemoryReaderConfig(enabled=False))
    state = {"messages": env[4].messages}
    if status in (401, 403):
        with pytest.raises(OVHTTPError):
            await mw.abefore_agent(state, runtime(env))
        assert not env[0].messages
    else:
        state.update(await mw.abefore_agent(state, runtime(env)))
        output = await call(mw, state, env[4], runtime(env))
        assert [m.id for m in output] == ["m1"]
        assert state["ov_context"]["context_status"] == "reload_required"


async def test_real_graph_checkpoint_hides_private_state_and_preserves_native_cache(env):
    mw = middleware(env)
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[]))
    model = ToolLoopModel()
    agent = create_agent(
        model=model,
        tools=[read_mail],
        middleware=[mw],
        checkpointer=saver,
        context_schema=CallContext,
    )
    config = {"configurable": {"thread_id": "s"}}
    output = await agent.ainvoke(
        {"messages": env[4].messages}, config=config, context=CallContext(env[3])
    )
    snapshot = (await agent.aget_state(config)).values
    assert "ov_context" not in output
    cached = snapshot["ov_context"]
    assert cached["write_index"] == len(snapshot["messages"]) == 4
    assert cached["completed_runs"] == 1 and cached["run_id"] is None
    assert [m.model_dump() for m in cached["messages"]] == [
        m.model_dump() for m in snapshot["messages"]
    ]
    assert env[0].counts["context"] == 1


async def test_commit_protects_recent_tool_groups_and_pins_current_user_separately(env):
    mw = middleware(env)
    ctx = OVContext(
        session_id="s",
        messages=[
            HumanMessage(id="old", content="old"),
            AIMessage(
                id="call", content="", tool_calls=[{"id": "c", "name": "read_mail", "args": {}}]
            ),
            ToolMessage(id="tool", tool_call_id="c", content="done"),
            HumanMessage(id="current", content="current"),
        ],
        owner=("a", "u", "p", "s"),
    )
    ctx.begin_run("r", ctx.messages[-1])
    assert mw._ov_session_engine.retained_tail(ctx) == (1, 3)
    ctx.original_user = ctx.messages[0]
    assert mw._ov_session_engine.retained_tail(ctx) == (3, 1)


async def test_same_text_different_ids_remains_two_native_messages(env):
    raw = [HumanMessage(id="one", content="same"), HumanMessage(id="two", content="same")]
    mw, state = await start(env, raw, reader_config=MemoryReaderConfig(enabled=False))
    output = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in output] == ["one", "two"]


async def test_assembler_preserves_system_metadata_and_injects_guide_once(env):
    assembler = ContextAssembler(OVContextConfig(context_window=100000), counter)
    ctx = OVContext(session_id="s", messages=env[4].messages, owner=("a", "u", "p", "s"))
    request = env[4].override(
        system_message=SystemMessage(id="sys", content="rules", additional_kwargs={"cache": True})
    )
    first = assembler.assemble(request, ctx)
    second = assembler.assemble(first, ctx)
    assert second.system_message.id == "sys"
    assert set(second.system_message.additional_kwargs) == {"cache", "ov_context_system_guide"}
    assert second.system_message.additional_kwargs["cache"] is True
    assert second.system_message.additional_kwargs == first.system_message.additional_kwargs
    assert request.system_message.additional_kwargs == {"cache": True}
    assert second.system_message.content.count("[OpenViking Context Guide]") == 1


async def test_parallel_tool_views_replace_only_matching_content_and_keep_args_metadata(env):
    raw = [
        HumanMessage(id="u", content="Read two messages"),
        AIMessage(
            id="a",
            content="",
            tool_calls=[
                {"id": "c1", "name": "browser", "args": {"url": "https://example.test"}},
                {"id": "c2", "name": "browser", "args": {"tab": 2}},
            ],
        ),
        ToolMessage(
            id="t2",
            name="browser",
            tool_call_id="c2",
            content="original two",
            artifact={"tab": 2},
            additional_kwargs={"trace": 2},
        ),
        ToolMessage(id="t1", name="browser", tool_call_id="c1", content="original one"),
    ]
    mw = middleware(env, reader_config=MemoryReaderConfig(enabled=False))
    writer = mw._ov_session_engine.writer
    original = writer.write

    async def short(*args, **kwargs):
        result = await original(*args, **kwargs)
        # External Writer owns short views; the native batch response is unchanged.
        messages = [
            message.model_copy(update={"content": "short two"})
            if isinstance(message, ToolMessage) and message.tool_call_id == "c2"
            else message
            for message in result.model_messages
        ]
        return replace(result, model_messages=messages)

    writer.write = short
    state = {"messages": raw}
    state.update(await mw.abefore_agent(state, runtime(env)))
    output = await call(mw, state, env[4], runtime(env))
    assert output[2].content == "short two" and output[3].content == "original one"
    assert output[2].artifact == {"tab": 2} and output[2].additional_kwargs == {"trace": 2}
    assert raw[2].content == "original two"


async def test_capacity_commit_waits_for_new_summary_then_retries_assembly_once(env):
    mw = middleware(
        env,
        reader_config=MemoryReaderConfig(enabled=False),
        commit_wait_timeout=0.2,
        poll_interval=0.001,
    )
    engine = mw._ov_session_engine
    engine.config = replace(engine.config, context_window=100, output_reserve=0, safety_margin=0)
    engine.assembler.config = engine.config
    engine.assembler.token_counter = lambda request: (
        20 if any(m.id == "ov:summary" for m in request.messages) else 200
    )
    raw = [
        HumanMessage(id="past", content="old history"),
        AIMessage(id="past-a", content="old answer"),
        HumanMessage(id="current", content="new question"),
    ]
    state = {"messages": raw}
    state.update(await mw.abefore_agent(state, runtime(env)))

    async def commit(session_id, keep_recent_count, *, identity=None):
        assert keep_recent_count == 1  # Entire current turn is protected.
        publish(env[0], env[3], {"past", "past-a"})
        return {"archived": True, "status": "accepted", "task_id": "background"}

    mw._connection.client.commit_session = commit
    output = await call(mw, state, env[4], runtime(env))
    assert [m.id for m in output] == ["ov:summary", "current"]
    assert state["ov_context"]["pending_commit"] is None
    assert env[0].counts["context"] == 2


async def test_capacity_error_never_removes_current_turn_without_safe_history(env):
    from openviking.integrations.langchain.ov_context import ContextBudgetExceeded

    mw, state = await start(env, reader_config=MemoryReaderConfig(enabled=False))
    engine = mw._ov_session_engine
    engine.assembler.token_counter = lambda request: 1_000_000
    with pytest.raises(ContextBudgetExceeded, match="no safe history"):
        await call(mw, state, env[4], runtime(env))
    assert env[0].counts["commit"] == 0
    assert state["ov_context"]["messages"] == state["messages"]


async def test_tool_backtracking_is_session_bound_and_schema_hides_runtime(env):
    mw = middleware(env)
    tools = {tool.name: tool for tool in mw.tools}
    read = tools["openviking_tool_result_read"]
    assert "runtime" not in read.tool_call_schema.model_json_schema()["properties"]
    rt = SimpleNamespace(
        state={"ov_context": {"session_id": "s", "owner": env[3].session_key, "run_id": "r"}},
        context=runtime(env).context,
    )
    with pytest.raises(ValueError, match="different session"):
        await read.coroutine(ref="viking://user/u/sessions/other/tool-results/x", runtime=rt)


async def test_original_guide_rules_for_list_system_content_are_idempotent(env):
    assembler = ContextAssembler(OVContextConfig(context_window=100000), counter)
    ctx = OVContext(session_id="s", messages=env[4].messages, owner=("a", "u", "p", "s"))
    request = env[4].override(
        system_message=SystemMessage(content=[{"type": "text", "text": "rules"}])
    )
    second = assembler.assemble(assembler.assemble(request, ctx), ctx)
    assert str(second.system_message.content).count("[OpenViking Context Guide]") == 1


async def test_first_context_load_after_compression_never_reimports_covered_graph_history(env):
    raw = [
        HumanMessage(id="old", content="old question"),
        AIMessage(id="old-a", content="old answer"),
        HumanMessage(id="new", content="new question"),
    ]
    first, state = await start(env, raw, reader_config=MemoryReaderConfig(enabled=False))
    state.update(await first.aafter_agent(state, runtime(env)))
    publish(env[0], env[3], {"old", "old-a"}, "old exchange summarized")
    # Resume with the persisted write cursor and force a fresh external context load.
    fresh = middleware(env, reader_config=MemoryReaderConfig(enabled=False))
    restored = deepcopy(state)
    restored["ov_context"]["context_status"] = "uninitialized"
    restored.update(await fresh.abefore_agent(restored, runtime(env, "new-run")))
    output = await call(fresh, restored, env[4], runtime(env, "new-run"))
    assert [m.id for m in output] == ["ov:summary", "new"]
    assert "old question" not in str(output)
    assert restored["ov_context"]["write_index"] == 3
    assert len(env[0].messages) == 3  # The persisted cursor avoids replay.


async def test_original_entry_requires_complete_runtime_identity(env):
    mw = middleware(env)
    rt = SimpleNamespace(context=None, execution_info=SimpleNamespace(thread_id="s", run_id="r"))
    with pytest.raises(ValueError, match="complete RunIdentity"):
        await mw.abefore_agent({"messages": env[4].messages}, rt)
    assert not env[0].requests


async def test_official_async_sdk_uses_native_context_and_tool_endpoints():
    import httpx

    from openviking_cli.client.http import AsyncHTTPClient

    observed = []

    async def handle(request):
        observed.append(request)
        return httpx.Response(200, json={"status": "ok", "result": {"ok": True}})

    client = AsyncHTTPClient(url="http://test", account="a", user="u", actor_peer_id="p")
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handle)
    ) as http:
        client._http = http
        await client.get_session_context("s")
        assert dict(observed[-1].url.params) == {
            "token_budget": "128000",
        }
        await client.read_tool_result("s", "r", offset=3, limit=7)
        assert observed[-1].url.path == "/api/v1/sessions/s/tool-results/r"
        assert dict(observed[-1].url.params) == {"offset": "3", "limit": "7"}
        await client.search_tool_result("s", "r", query="mail")
        assert observed[-1].url.path.endswith("/tool-results/r/search")
        await client.list_tool_results("s")
        assert observed[-1].url.path.endswith("/s/tool-results")


async def test_first_load_with_existing_parallel_tool_results_reuses_one_native_pair(env):
    raw = [
        HumanMessage(id="u", content="Read mail"),
        AIMessage(
            id="a",
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_mail", "args": {}},
                {"id": "c2", "name": "read_mail", "args": {}},
            ],
        ),
        ToolMessage(id="t2", tool_call_id="c2", content="two", artifact={"n": 2}),
        ToolMessage(id="t1", tool_call_id="c1", content="one", additional_kwargs={"n": 1}),
    ]
    first, state = await start(env, raw, reader_config=MemoryReaderConfig(enabled=False))
    state.update(await first.aafter_agent(state, runtime(env)))
    fresh = middleware(env, reader_config=MemoryReaderConfig(enabled=False))
    state.update(await fresh.abefore_agent(state, runtime(env, "r2")))
    output = await call(fresh, state, env[4], runtime(env, "r2"))
    assert [m.model_dump() for m in output] == [m.model_dump() for m in raw]
    assert len(env[0].messages) == 3  # One user, assistant, aggregate; replay adds none.


async def test_original_peer_resolver_is_used_by_async_writer(env):
    mw, state = await start(env)
    mw.peer_id_resolver = lambda state, runtime: "resolved-peer"
    state["messages"].append(AIMessage(id="answer", content="done"))
    state.update(await mw.aafter_model(state, runtime(env)))
    batches = [body for request, body in env[0].requests if request.url.path.endswith("/batch")]
    assert batches[-1]["messages"][0]["peer_id"] == "resolved-peer"


async def test_after_agent_failed_write_releases_active_marker_without_publishing_state(env):
    mw, state = await start(env)
    before = deepcopy(state["ov_context"])
    state["messages"].append(AIMessage(id="final", content="done"))
    env[0].error["batch"] = 503
    with pytest.raises(OVHTTPError):
        await mw.aafter_agent(state, runtime(env))
    assert mw._ov_session_engine.active == {}
    assert state["ov_context"] == before


@pytest.mark.parametrize("archive_id", [None, "archive_001"])
async def test_commit_records_applied_archive_as_wait_baseline(env, archive_id: str | None) -> None:
    env[0].sessions.add("s")
    engine = middleware(env)._ov_session_engine
    user = HumanMessage(id="current", content="Read mail")
    context = OVContext(
        "s",
        messages=[
            HumanMessage(id="old", content="Open mailbox"),
            AIMessage(id="a", content="Opened"),
            user,
        ],
        archive_id=archive_id,
        owner=("a", "u", "p", "s"),
    )
    context.begin_run("r", user)
    assert await engine.commit(context, identity=context.identity)
    assert context.pending_commit == {"task_id": "t1", "archive_id": archive_id}
    assert context.archive_id == archive_id


@pytest.mark.parametrize("blocks", [False, True], ids=["text", "multimodal"])
async def test_archive_refresh_reconciles_unique_current_user_with_server_assigned_id(
    env: tuple[Any, ...], blocks: bool
) -> None:
    from .test_hook_design import Reader, make

    content = (
        [
            {"type": "text", "text": "Current question"},
            {"type": "image_url", "image_url": {"url": "https://image.test/a.png"}},
        ]
        if blocks
        else "Current question"
    )
    user = HumanMessage(id="graph-user", content=content, additional_kwargs={"trace": "keep"})
    reader = Reader()
    mw = make(env, reader)
    state = {"messages": [HumanMessage(id="old", content="Old raw history"), user]}
    state.update(await mw.abefore_agent(state, runtime(env)))
    await call(mw, state, env[4], runtime(env))
    summary = HumanMessage(id="ov_summary:archive_001", content="Archived history")
    reader.snapshots["s"] = (
        "archive_001",
        [summary, HumanMessage(id="ov-generated-id", content=deepcopy(content))],
    )
    output = await call(mw, state, env[4], runtime(env))
    assert [message.id for message in output] == [summary.id, user.id]
    assert output[-1].additional_kwargs["trace"] == "keep"
    assert str(output).count("[参考记忆]") == 1
    assert "Old raw history" not in str(output)
    assert state["messages"][-1] is user
    assert reader.snapshots["s"][1][-1].id == "ov-generated-id"


@pytest.mark.parametrize("bad", ["ambiguous", "changed", "id-collision", "summary-only"])
async def test_changed_ov_user_id_requires_unique_original_content_and_no_collision(
    bad: str,
) -> None:
    from openviking.integrations.langchain.ov_context.errors import ContextReadError

    user = HumanMessage(id="graph-user", content="query")
    ctx = OVContext(session_id="s", messages=[user])
    ctx.begin_run("r", user)
    snapshot = [HumanMessage(id="ov-user", content="query")]
    if bad == "ambiguous":
        snapshot.append(HumanMessage(id="ov-other", content="query"))
    elif bad == "changed":
        snapshot[0] = HumanMessage(id="ov-user", content="changed")
    elif bad == "id-collision":
        snapshot.insert(0, AIMessage(id=user.id, content="answer"))
    else:
        snapshot = [HumanMessage(id="ov_summary:archive_001", content="query")]
    before = deepcopy(ctx.to_state())
    with pytest.raises(ContextReadError):
        ctx.replace(snapshot, "archive_001")
    assert ctx.to_state() == before
