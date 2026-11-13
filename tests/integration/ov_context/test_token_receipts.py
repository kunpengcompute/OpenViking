"""Run-scoped receipts across hooks, fallback, recovery and uncertain writes."""

import asyncio
from dataclasses import dataclass, replace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain.ov_context import OVContext, WriteResult
from openviking.integrations.langchain.ov_context.commit import CommitConfig, CommitResult
from openviking.integrations.langchain.ov_context.writer import (
    IncompleteWriteError,
    OVMessageWriter,
    WriterConfig,
)

from .test_hook_design import Reader, make
from .test_io_contracts import SuppliedWriter
from .test_session_hooks import call, runtime


@dataclass(frozen=True)
class ReceiptResult(WriteResult):
    pending_tokens: int | None = None


class ReceiptWriter(SuppliedWriter):
    def __init__(self, tokens=17):
        super().__init__()
        self.tokens = tokens

    async def write(self, *args, **kwargs):
        result = await super().write(*args, **kwargs)
        return ReceiptResult(
            result.model_messages, result.write_index, result.written_count, self.tokens
        )


def reads(env):
    return sum(
        request.method == "GET" and request.url.path.endswith("/sessions/s")
        for request, _ in env[0].requests
    )


async def setup(env, tokens=17, **kwargs):
    writer, reader = ReceiptWriter(tokens), Reader()
    reader.tokens = 99  # A fallback has a distinguishable result.
    middleware = make(env, reader, writer, **kwargs)
    state = {"messages": [HumanMessage(id="u", content="hello")]}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    return middleware, writer, reader, state


async def test_hello_reuses_receipt_for_both_hooks_but_keeps_summary_read(env):
    mw, writer, reader, state = await setup(env, commit_config=CommitConfig(100000, 0.7, 100, 0))
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before + 1  # check_summary still queries the boundary.
    state["messages"].append(AIMessage(id="a", content="Hello"))
    before = reads(env)
    state.update(await mw.aafter_model(state, runtime(env)))
    state.update(await mw.aafter_agent(state, runtime(env)))
    assert reads(env) == before
    assert state["ov_context"]["completed_runs"] == 1
    assert not any("receipt" in key or "pending_tokens" in key for key in state["ov_context"])


@pytest.mark.parametrize("tokens", [None, -1, True, "17", 1.5])
async def test_missing_invalid_receipt_falls_back_without_rejecting_write(env, tokens):
    mw, writer, reader, state = await setup(env, tokens)
    context = OVContext.from_state(state["ov_context"])
    before = reads(env)
    assert await mw._ov_session_engine.pending_tokens(context, identity=(context).identity) == 99
    assert reads(env) == before + 1
    assert context.write_index == 1


async def test_new_write_replaces_receipt_and_failed_write_invalidates_it(env):
    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    assert await engine.pending_tokens(context, identity=(context).identity) == 17
    writer.tokens = 23
    state["messages"].append(AIMessage(id="a", content="Hello"))
    await engine.write(context, state["messages"], identity=(context).identity)
    assert await engine.pending_tokens(context, identity=(context).identity) == 23
    writer.error = RuntimeError("uncertain write")
    state["messages"].append(HumanMessage(id="u2", content="again"))
    with pytest.raises(RuntimeError, match="uncertain"):
        await engine.write(context, state["messages"], identity=(context).identity)
    assert context.write_index == 2
    assert await engine.pending_tokens(context, identity=(context).identity) == 99


@pytest.mark.parametrize("outcome", ["skipped", "not_archived", "archived", "error"])
async def test_only_policy_skip_preserves_receipt(env, outcome):
    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    policy = AsyncMock()
    if outcome == "error":
        policy.commit.side_effect = RuntimeError("lost commit response")
        with pytest.raises(RuntimeError):
            await engine._send_commit(
                context,
                pending_tokens=17,
                trigger="policy",
                policy=policy,
                keep=0,
                identity=(context).identity,
            )
    else:
        policy.commit.return_value = CommitResult(status=outcome, archived=outcome == "archived")
        await engine._send_commit(
            context,
            pending_tokens=17,
            trigger="policy",
            policy=policy,
            keep=0,
            identity=(context).identity,
        )
    assert await engine.pending_tokens(context, identity=(context).identity) == (
        17 if outcome == "skipped" else 99
    )


async def test_invalidated_or_new_engine_never_trusts_checkpoint_receipt(env):
    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    engine.invalidate_tokens(context)
    assert await engine.pending_tokens(context, identity=(context).identity) == 99
    writer.tokens = 25
    state["messages"].append(AIMessage(id="a", content="Hello"))
    await engine.write(context, state["messages"], identity=(context).identity)
    assert await engine.pending_tokens(context, identity=(context).identity) == 25
    fresh = make(env, reader, writer)
    assert (
        await fresh._ov_session_engine.pending_tokens(
            OVContext.from_state(context.to_state()),
            identity=(OVContext.from_state(context.to_state())).identity,
        )
        == 99
    )
    other_run = replace(context, run_id="different-run")
    assert await engine.pending_tokens(other_run, identity=(other_run).identity) == 99


@pytest.mark.parametrize("change", ["identity", "session"])
async def test_receipt_does_not_cross_identity_or_session(env, change):
    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    if change == "identity":
        other = replace(context, owner=("a", "another-user", "p", "s"))
    else:
        other = replace(context, session_id="other", owner=("a", "u", "p", "other"))
    engine._call_read = AsyncMock(
        return_value={"session_id": other.session_id, "pending_tokens": 99}
    )
    assert await engine.pending_tokens(other, identity=(other).identity) == 99
    engine._call_read.assert_awaited_once_with(
        "get_session", identity=other.identity, session_id=other.session_id, auto_create=False
    )


async def test_cancelled_write_invalidates_receipt_without_advancing_cursor(env):
    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    context = OVContext.from_state(state["ov_context"])
    writer.error = asyncio.CancelledError()
    state["messages"].append(AIMessage(id="a", content="Hello"))
    with pytest.raises(asyncio.CancelledError):
        await engine.write(context, state["messages"], identity=(context).identity)
    assert context.write_index == 1
    assert await engine.pending_tokens(context, identity=(context).identity) == 99


async def test_tools_and_multiple_model_calls_replace_receipts_keep_boundary_queries(env):
    mw, writer, reader, state = await setup(env, commit_config=CommitConfig(100000, 0.7, 100, 0))
    for index in range(2):
        await call(mw, state, env[4], runtime(env))
        writer.tokens = 30 + index
        state["messages"].append(
            AIMessage(
                id=f"call-{index}",
                content="",
                tool_calls=[{"id": f"tool-{index}", "name": "lookup", "args": {}}],
            )
        )
        state.update(await mw.aafter_model(state, runtime(env)))
        writer.tokens = 40 + index
        state["messages"].append(
            ToolMessage(
                id=f"result-{index}",
                tool_call_id=f"tool-{index}",
                content="result",
            )
        )
        before = reads(env)
        state.update(await mw.abefore_model(state, runtime(env)))
        assert reads(env) == before  # Tool write supplied its own updated receipt.
        assert (
            await mw._ov_session_engine.pending_tokens(
                OVContext.from_state(state["ov_context"]),
                identity=(OVContext.from_state(state["ov_context"])).identity,
            )
            == 40 + index
        )
    before = reads(env)
    await call(mw, state, env[4], runtime(env))
    assert reads(env) == before + 1


@pytest.mark.parametrize("last", [42, None, -1, True, "42"])
async def test_writer_uses_last_batch_receipt_or_none(last):
    client = AsyncMock()
    client.batch_add_messages.side_effect = [
        {"session_id": "s", "message_count": 1, "added": 1, "pending_tokens": 10},
        {"session_id": "s", "message_count": 2, "added": 1, "pending_tokens": last},
    ]
    result = await OVMessageWriter(client, WriterConfig(batch_size=1)).write(
        "s", [HumanMessage(content="hello"), AIMessage(content="Hello")], 0
    )
    assert getattr(result, "pending_tokens", None) == (
        42 if type(last) is int and last >= 0 else None
    )


async def test_partial_batch_exposes_confirmed_snapshot_only_as_diagnostics():
    client = AsyncMock()
    client.batch_add_messages.side_effect = [
        {"session_id": "s", "message_count": 1, "added": 1, "pending_tokens": 10},
        RuntimeError("second batch uncertain"),
    ]
    with pytest.raises(IncompleteWriteError) as caught:
        await OVMessageWriter(client, WriterConfig(batch_size=1)).write(
            "s", [HumanMessage(content="hello"), AIMessage(content="Hello")], 0
        )
    assert caught.value.confirmed_result.write_index == 1
    assert caught.value.confirmed_result.written_count == 1
    assert getattr(caught.value.confirmed_result, "pending_tokens", None) == 10


async def test_overridden_policy_keeps_read_before_decision(env):
    from openviking.integrations.langchain.middleware import OpenVikingCommitPolicy

    observed = []

    class CustomPolicy(OpenVikingCommitPolicy):
        async def commit(self, session_id, pending_tokens, runs, **kwargs):
            observed.append(pending_tokens)
            return CommitResult(status="skipped")

    mw, writer, reader, state = await setup(env)
    engine = mw._ov_session_engine
    engine.session_commit_policy = CustomPolicy(CommitConfig(100000, 0.7, 100, 0))
    context = OVContext.from_state(state["ov_context"])
    context.end_run()
    before = reads(env)
    await engine.commit(
        context,
        pending_tokens=await engine.pending_tokens(context, identity=(context).identity),
        identity=(context).identity,
    )
    assert observed == [99]
    assert reads(env) == before + 1
