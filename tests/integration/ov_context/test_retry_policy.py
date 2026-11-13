"""Bounded retry policy for transient OV failures.

``retry_attempts=0`` must remain byte-for-byte equivalent to the pre-retry
implementation: one call, no sleep, no reconciliation, no swallowed error.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from openviking.integrations.langchain.ov_context import (
    ContextReadResult,
    OVContext,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.errors import IncompleteWriteError
from openviking.integrations.langchain.ov_context.retry import (
    retryable_commit,
    retryable_read,
    retryable_write,
)
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.writer import WriteResult
from openviking_cli.exceptions import (
    ConflictError,
    InternalError,
    UnavailableError,
)
from tests.integration.ov_context.test_hook_design import Reader, make, runtime
from tests.integration.ov_context.test_io_contracts import SuppliedWriter
from tests.ov_context_support.commit_settings import EXAMPLE_COMMIT_CONFIG

pytestmark = pytest.mark.asyncio


class FakeConnection:
    path = None

    def __init__(self, client: Any) -> None:
        self.client = client
        if hasattr(client, "__dict__"):
            client._initialized = True  # The fake SDK has no deferred resources.


def engine(client: Any, *, retries: int, writer: Any = None) -> SessionEngine:
    return SessionEngine(
        FakeConnection(client),
        OVContextConfig(context_window=100000, retry_attempts=retries),
        None,
        None,
        message_writer=writer,
    )


def context(messages: list | None = None) -> OVContext:
    return OVContext(
        session_id="s",
        messages=messages or [],
        owner=("a", "u", "p", "s"),
    )


class FlakyGet:
    def __init__(self, failures: int, value: dict[str, Any] | None = None) -> None:
        self.failures = failures
        self.value = value or {"session_id": "s", "pending_tokens": 7}
        self.calls = 0

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.ConnectError("down")
        return self.value


async def test_zero_retries_calls_once_and_raises() -> None:
    client = FlakyGet(1)
    with pytest.raises(httpx.ConnectError):
        await engine(client, retries=0).pending_tokens(context(), identity=context().identity)
    assert client.calls == 1


async def test_read_retries_transient_failure() -> None:
    client = FlakyGet(2)
    value = await engine(client, retries=2).pending_tokens(context(), identity=context().identity)
    assert value == 7
    assert client.calls == 3


async def test_read_does_not_retry_non_transient_error() -> None:
    class BadValue:
        calls = 0

        async def get_session(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise ValueError("bad response")

    client = BadValue()
    with pytest.raises(ValueError):
        await engine(client, retries=2).pending_tokens(context(), identity=context().identity)
    assert client.calls == 1


class FlakyEnsure:
    def __init__(self) -> None:
        self.calls = 0

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("down")
        return {"session_id": "s"}


async def test_ensure_session_retries_whole_sequence() -> None:
    client = FlakyEnsure()
    ctx = context()
    result = await engine(client, retries=2).ensure_session("s", identity=ctx.identity)
    assert result["session_id"] == "s"
    assert client.calls == 2


class FlakyWriter:
    def __init__(self, *, written_count: int, cause: Exception | None = None) -> None:
        self.written_count = written_count
        self.cause = cause or httpx.ConnectError("down")
        self.calls = 0

    async def write(
        self,
        session_id: str,
        raw_messages: list,
        write_index: int,
        run_id: str,
        *,
        identity: Any = None,
        peer_id: str | None = None,
    ) -> WriteResult:
        self.calls += 1
        if self.calls == 1:
            try:
                raise self.cause
            except Exception as cause:
                raise IncompleteWriteError(
                    WriteResult([], write_index, self.written_count)
                ) from cause
        return WriteResult([], len(raw_messages), len(raw_messages) - write_index)


async def test_write_retries_only_pre_send_with_empty_prefix() -> None:
    writer = FlakyWriter(written_count=0)
    ctx = context([HumanMessage(content="hi", id="u1")])
    await engine(object(), retries=2, writer=writer).write(ctx, ctx.messages, identity=ctx.identity)
    assert writer.calls == 2
    assert ctx.write_index == 1


@pytest.mark.parametrize(
    "writer",
    [
        FlakyWriter(written_count=1),
        FlakyWriter(written_count=0, cause=httpx.ReadTimeout("late")),
    ],
)
async def test_write_does_not_retry_ambiguous_failure(writer: FlakyWriter) -> None:
    ctx = context([HumanMessage(content="hi", id="u1")])
    with pytest.raises(IncompleteWriteError):
        await engine(object(), retries=2, writer=writer).write(
            ctx, ctx.messages, identity=ctx.identity
        )
    assert writer.calls == 1
    assert ctx.write_index == 0


def _history() -> list:
    return [
        HumanMessage(content=f"u{i}", id=f"u{i}")
        if i % 2 == 0
        else AIMessage(content=f"a{i}", id=f"a{i}")
        for i in range(12)
    ]


class CommitConflict:
    def __init__(self) -> None:
        self.attempted = False
        self.commits = 0
        self.gets = 0

    async def commit_session(self, **kwargs: Any) -> dict[str, Any]:
        self.commits += 1
        self.attempted = True
        raise ConflictError("path locked")

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        self.gets += 1
        return {"session_id": "s", "commit_count": 1 if self.attempted else 0}

    async def list_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.attempted:
            return [{"task_id": "t", "resource_id": "s", "task_type": "session_commit"}]
        return []


async def test_commit_conflict_propagates_without_discovery_or_resending() -> None:
    client = CommitConflict()
    ctx = context(_history())
    with pytest.raises(ConflictError):
        await engine(client, retries=2).commit(ctx, identity=ctx.identity)
    assert client.commits == 1
    assert client.gets == 0 and ctx.pending_commit is None


class CommitRetry:
    def __init__(self) -> None:
        self.commits = 0

    async def commit_session(self, **kwargs: Any) -> dict[str, Any]:
        self.commits += 1
        if self.commits == 1:
            raise httpx.ConnectError("not sent")
        return {"archived": False}

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        return {"session_id": "s", "commit_count": 0}

    async def list_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


async def test_commit_retries_only_known_pre_send_errors() -> None:
    client = CommitRetry()
    ctx = context(_history())
    assert await engine(client, retries=2).commit(ctx, identity=ctx.identity) is False
    assert client.commits == 2


class CommitAmbiguous:
    def __init__(self) -> None:
        self.commits = 0

    async def commit_session(self, **kwargs: Any) -> dict[str, Any]:
        self.commits += 1
        raise InternalError("outcome unknown")

    async def get_session(self, **kwargs: Any) -> dict[str, Any]:
        return {"session_id": "s", "commit_count": 0}

    async def list_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


async def test_commit_does_not_resend_ambiguous_failure() -> None:
    client = CommitAmbiguous()
    with pytest.raises(InternalError):
        await engine(client, retries=2).commit(
            context(_history()), identity=context(_history()).identity
        )
    assert client.commits == 1


async def test_wait_summary_tolerates_transient_poll_errors() -> None:
    eng = engine(object(), retries=2)
    calls = 0

    async def flaky(ctx: OVContext, **kwargs) -> ContextReadResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("down")
        ctx.pending_commit = None
        return ContextReadResult(None)

    eng.check_summary = flaky  # type: ignore[method-assign]
    ctx = context()
    ctx.pending_commit = {"task_id": "t", "archive_id": None}
    await eng.wait_summary(ctx, identity=ctx.identity)
    assert calls == 2


async def test_wait_summary_zero_retries_keeps_legacy_failure() -> None:
    eng = engine(object(), retries=0)

    async def failing(ctx: OVContext, **kwargs) -> ContextReadResult:
        raise httpx.ConnectError("down")

    eng.check_summary = failing  # type: ignore[method-assign]
    with pytest.raises(httpx.ConnectError):
        await eng.wait_summary(context(), identity=context().identity)


async def test_abefore_model_keeps_write_when_pending_tokens_is_transient(
    env: tuple[Any, ...],
) -> None:
    reader = Reader()
    writer = SuppliedWriter()
    mw = make(
        env,
        reader,
        writer,
        retry_attempts=2,
        commit_config=EXAMPLE_COMMIT_CONFIG,
    )
    state: dict[str, Any] = {"messages": list(env[4].messages)}
    state.update(await mw.abefore_agent(state, runtime(env)))
    next_user = HumanMessage(content="next", id="u-next")
    state["messages"] = [*state["messages"], next_user]

    async def down(ctx: OVContext, **kwargs) -> int | None:
        raise httpx.ConnectError("down")

    mw._ov_session_engine.pending_tokens = down  # type: ignore[method-assign]
    update = await mw.abefore_model(state, runtime(env))
    assert update["ov_context"]["write_index"] == len(state["messages"])


async def test_initial_refresh_retries_before_failing(env: tuple[Any, ...]) -> None:
    reader = Reader()
    reader.error = httpx.ConnectError("down")
    mw = make(env, reader, retry_attempts=2)
    with pytest.raises(httpx.ConnectError):
        await mw.abefore_agent({"messages": list(env[4].messages)}, runtime(env))
    assert len(reader.exchanges) == 3


async def test_pre_write_failure_is_marked_consistent(env: tuple[Any, ...]) -> None:
    reader = Reader()
    reader.error = httpx.ConnectError("down")
    mw = make(env, reader, retry_attempts=2)
    with pytest.raises(httpx.ConnectError) as caught:
        await mw.abefore_agent({"messages": list(env[4].messages)}, runtime(env))
    assert getattr(caught.value, "ov_state_consistent", False) is True
    assert not mw._ov_session_engine.active


async def test_zero_retries_does_not_mark_consistent(env: tuple[Any, ...]) -> None:
    reader = Reader()
    reader.error = httpx.ConnectError("down")
    mw = make(env, reader, retry_attempts=0)
    with pytest.raises(httpx.ConnectError) as caught:
        await mw.abefore_agent({"messages": list(env[4].messages)}, runtime(env))
    assert not getattr(caught.value, "ov_state_consistent", False)
    assert len(reader.exchanges) == 1


async def test_abefore_model_zero_retries_still_raises(env: tuple[Any, ...]) -> None:
    reader = Reader()
    writer = SuppliedWriter()
    mw = make(env, reader, writer, retry_attempts=0, commit_config=EXAMPLE_COMMIT_CONFIG)
    state: dict[str, Any] = {"messages": list(env[4].messages)}
    state.update(await mw.abefore_agent(state, runtime(env)))
    state["messages"] = [*state["messages"], HumanMessage(content="next", id="u-next")]

    async def down(ctx: OVContext, **kwargs) -> int | None:
        raise httpx.ConnectError("down")

    mw._ov_session_engine.pending_tokens = down  # type: ignore[method-assign]
    with pytest.raises(httpx.ConnectError):
        await mw.abefore_model(state, runtime(env))


async def test_classification_guards() -> None:
    assert retryable_read(httpx.ConnectError("down"))
    assert retryable_read(UnavailableError("busy"))
    assert not retryable_read(ValueError("bad"))
    assert retryable_commit(httpx.ConnectError("not sent"))
    assert not retryable_commit(httpx.ReadTimeout("late"))
    assert not retryable_commit(ConflictError("locked"))
    assert not retryable_commit(UnavailableError("unknown server outcome"))


async def test_write_classification_requires_pre_send_cause() -> None:
    from openviking.integrations.langchain.ov_context.writer import AmbiguousWriteError

    assert not retryable_write(AmbiguousWriteError("unknown"))
    assert not retryable_write(ValueError("bad"))
