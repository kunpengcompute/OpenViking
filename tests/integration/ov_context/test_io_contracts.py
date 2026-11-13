"""Verify host-supplied I/O contracts at the native Middleware boundary."""

from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ContextReaderBase,
    MessageWriterBase,
    OVContext,
    OVContextConfig,
    RecallResult,
    WriteResult,
)
from openviking.integrations.langchain.ov_context.reader import ContextSnapshot
from tests.ov_context_support.fixtures import counter
from tests.ov_context_support.sdk_adapter import TransportSDKAdapter

from .test_session_hooks import call, runtime

pytestmark = pytest.mark.asyncio


class SuppliedWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str, str | None]] = []
        self.error: Exception | None = None

    async def write(
        self,
        session_id: str,
        raw_messages: list[BaseMessage],
        write_index: int,
        run_id: str,
        *,
        identity=None,
        peer_id: str | None = None,
    ) -> WriteResult:
        self.calls.append((session_id, write_index, run_id, peer_id))
        if self.error:
            raise self.error
        delta = raw_messages[write_index:]
        return WriteResult(
            list(delta),
            len(raw_messages),
            len(delta),
        )


class SuppliedReader:
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []
        self.archive_id: str | None = None
        self.queries: list[str] = []
        self.context_calls = 0
        self.exchange_calls: list[tuple[str, str | None]] = []

    async def exchange_context(
        self,
        session_id: str,
        expected_archive_id: str | None = None,
        *,
        identity=None,
        token_budget: int | None = None,
    ) -> ContextSnapshot:
        self.exchange_calls.append((session_id, expected_archive_id))
        self.context_calls += 1
        return ContextSnapshot(list(self.messages), self.archive_id)

    async def recall(self, query: str, *, identity=None, session_id: str) -> RecallResult:
        self.queries.append(query)
        return RecallResult("Use concise answers", "success", 1)


def make(
    env: tuple[Any, ...], writer: MessageWriterBase | None, reader: ContextReaderBase | None
) -> OpenVikingContextMiddleware:
    if reader is not None:
        env[0].session_meta_hook = lambda sid: {
            "latest_finished_archive_id": getattr(reader, "archive_id", None)
        }
    return OpenVikingContextMiddleware(
        client=TransportSDKAdapter(env[1]),
        message_writer=writer,
        context_reader=reader,
        context_config=OVContextConfig(context_window=100000),
        session_id_resolver=lambda state, runtime: "s",
        peer_id="p",
        token_counter=counter,
    )


@pytest.mark.parametrize("missing", ["both", "writer", "reader"])
async def test_missing_io_fails_before_any_network_or_state_change(
    env: tuple[Any, ...], missing: str
) -> None:
    writer = None if missing in ("both", "writer") else SuppliedWriter()
    reader = None if missing in ("both", "reader") else SuppliedReader()
    middleware = make(env, writer, reader)
    state = {"messages": env[4].messages}
    with pytest.raises(ValueError, match="requires injected"):
        await middleware.abefore_agent(state, runtime(env))
    assert not env[0].requests
    assert "ov_context" not in state
    assert not middleware._ov_session_engine.active


async def test_supplied_io_drives_cache_recall_native_formats_and_summary_refresh(
    env: tuple[Any, ...],
) -> None:
    writer, reader = SuppliedWriter(), SuppliedReader()
    middleware = make(env, writer, reader)
    assert middleware._ov_session_engine.context_reader is reader
    user = HumanMessage(id="u", content="Read mail", additional_kwargs={"trace": "u"})
    assistant = AIMessage(id="a", content="", tool_calls=[{"id": "c", "name": "read", "args": {}}])
    tool = ToolMessage(id="t", tool_call_id="c", content=[{"type": "text", "text": "mail"}])
    state: dict[str, Any] = {"messages": [user, assistant, tool]}
    original = deepcopy(state)
    state.update(await middleware.abefore_agent(state, runtime(env)))
    assert writer.calls == [("s", 0, "r", "p")]
    requests_after_start = len(env[0].requests)
    assert state["ov_context"]["write_index"] == 3
    poisoned = env[4].override(messages=[HumanMessage(id="bad", content="must not be used")])
    output = await call(middleware, state, poisoned, runtime(env))
    assert [type(m) for m in output] == [HumanMessage, AIMessage, ToolMessage]
    assert output[0].additional_kwargs["trace"] == "u"
    assert "Use concise answers" in output[0].content
    assert output[1].model_dump() == assistant.model_dump()
    assert output[2].model_dump() == tool.model_dump()
    assert state["messages"] == original["messages"]
    await call(middleware, state, poisoned, runtime(env))
    assert reader.queries == ["Read mail"] and reader.context_calls == 1
    # The Reader returns a snapshot; SessionEngine owns cache publication.
    summary = HumanMessage(id="ov:summary", content="Summarized the completed tool exchange")
    reader.messages = [summary, user]
    reader.archive_id = "archive_001"
    output = await call(middleware, state, poisoned, runtime(env))
    assert [m.id for m in output] == ["ov:summary", "u"]
    assert reader.context_calls == 2 and reader.queries == ["Read mail"]
    assert state["messages"] == original["messages"]
    assert not env[0].counts["batch"] and not env[0].counts["find"]
    assert not env[0].counts["context"]  # The injected reader owns all context refreshes.
    assert reader.exchange_calls == [
        ("s", None),
        ("s", None),
    ]  # Always request the latest snapshot.
    assert len(env[0].requests) > requests_after_start


async def test_supplied_writer_failure_does_not_advance_cache_cursor(
    env: tuple[Any, ...],
) -> None:
    writer, reader = SuppliedWriter(), SuppliedReader()
    middleware = make(env, writer, reader)
    state = {"messages": env[4].messages}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    previous = deepcopy(state["ov_context"])
    state["messages"].append(AIMessage(id="new", content="Final answer"))
    writer.error = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        await middleware.aafter_model(state, runtime(env))
    assert state["ov_context"] == previous
    assert state["ov_context"]["write_index"] == 1


async def test_write_contract_and_checkpoint_round_trip() -> None:
    from dataclasses import fields

    assert {field.name for field in fields(WriteResult)} == {
        "model_messages",
        "write_index",
        "written_count",
        "pending_tokens",
    }
    context = OVContext("s", write_index=2, archive_id="archive_001", owner=("a", "u", "p", "s"))
    restored = OVContext.from_state(context.to_state())
    assert restored.write_index == 2 and restored.archive_id == "archive_001"


async def test_native_batch_uses_only_original_payload_and_response_fields(
    env: tuple[Any, ...],
) -> None:
    client = TransportSDKAdapter(env[1])
    await client.create_session("s")
    payload = {"role": "user", "content": "Read mail"}
    result = await client.batch_add_messages("s", [payload])
    assert result == {"session_id": "s", "added": 1, "message_count": 1}
    assert env[0].requests[-1][1] == {"messages": [payload]}


class UnscannableMessages(list[BaseMessage]):
    """Fail if a no-op path tries to scan or copy the message sequence."""

    def __iter__(self) -> Any:
        raise AssertionError("No-op path scanned the message list")

    def __getitem__(self, key: Any) -> Any:
        raise AssertionError("No-op path indexed the message list")


async def test_no_delta_hooks_skip_writer_cache_copy_and_state_publication(
    env: tuple[Any, ...],
) -> None:
    writer, reader = SuppliedWriter(), SuppliedReader()
    middleware = make(env, writer, reader)
    state = {"messages": env[4].messages}
    state.update(await middleware.abefore_agent(state, runtime(env)))
    state["messages"] = UnscannableMessages(state["messages"])
    cached = state["ov_context"]
    cached["messages"] = UnscannableMessages(cached["messages"])
    original_cache = cached["messages"]
    activity = cached["last_activity"]
    for hook in (middleware.abefore_model, middleware.aafter_model):
        assert await hook(state, runtime(env)) == {}
    assert writer.calls == [("s", 0, "r", "p")]
    assert cached["messages"] is original_cache
    assert cached["last_activity"] == activity
    # Skipping work must not skip session ownership validation.
    cached["session_id"] = "different-session"
    with pytest.raises(ValueError, match="different session/identity"):
        await middleware.abefore_model(state, runtime(env))


async def test_engine_no_delta_never_scans_messages(env: tuple[Any, ...]) -> None:
    writer = SuppliedWriter()
    engine = make(env, writer, SuppliedReader())._ov_session_engine
    raw = UnscannableMessages([HumanMessage(id="u", content="mail")])
    context = OVContext(
        "s", messages=raw, write_index=1, last_activity=123.0, owner=("a", "u", "p", "s")
    )
    await engine.write(context, raw, identity=context.identity)
    assert not writer.calls
    assert context.messages is raw and context.last_activity == 123.0


async def test_filtered_write_advances_cursor_without_scanning_cache(env: tuple[Any, ...]) -> None:
    class FilteredWriter(SuppliedWriter):
        async def write(
            self,
            session_id: str,
            raw_messages: list[BaseMessage],
            write_index: int,
            run_id: str,
            *,
            identity=None,
            peer_id: str | None = None,
        ) -> WriteResult:
            return WriteResult([], len(raw_messages), 0)

    engine = make(env, FilteredWriter(), SuppliedReader())._ov_session_engine
    cache = UnscannableMessages([HumanMessage(id="old", content="history")])
    context = OVContext("s", messages=cache, owner=("a", "u", "p", "s"))
    await engine.write(context, [AIMessage(id="filtered", content="")], identity=context.identity)
    assert context.write_index == 1
    assert context.messages is cache
    assert context.last_activity > 0


async def test_regressed_cursor_fails_before_writer(env: tuple[Any, ...]) -> None:
    writer = SuppliedWriter()
    engine = make(env, writer, SuppliedReader())._ov_session_engine
    context = OVContext("s", write_index=2, owner=("a", "u", "p", "s"))
    with pytest.raises(ValueError, match="cursor regressed"):
        await engine.write(
            context, [HumanMessage(id="u", content="mail")], identity=context.identity
        )
    assert not writer.calls and context.write_index == 2


async def test_host_shared_client_initialized_and_closed_once() -> None:
    from openviking.integrations.langchain.client import OpenVikingConnection
    from openviking.integrations.langchain.ov_context.sdk import AsyncClientHandle

    class HostClient:
        def __init__(self) -> None:
            self._initialized = False
            self.initializations = 0
            self.closes = 0

        async def initialize(self) -> None:
            self.initializations += 1
            self._initialized = True

        async def close(self) -> None:
            self.closes += 1

    class BorrowingWriter(SuppliedWriter):
        def __init__(self, client: HostClient) -> None:
            super().__init__()
            self.client = AsyncClientHandle(OpenVikingConnection(client=client))

    class BorrowingReader(SuppliedReader):
        def __init__(self, client: HostClient) -> None:
            super().__init__()
            self.client = AsyncClientHandle(OpenVikingConnection(client=client))

    client = HostClient()
    await client.initialize()
    writer, reader = BorrowingWriter(client), BorrowingReader(client)
    middleware = OpenVikingContextMiddleware(
        client=client, message_writer=writer, context_reader=reader
    )
    handles = (middleware._ov_session_engine.client, writer.client, reader.client)
    for handle in handles:
        assert await handle.get() is client
        assert not handle.owned
    assert client.initializations == 1
    await middleware.aclose()
    for handle in handles[1:]:
        await handle.close()
    assert client.closes == 0
    await client.close()
    assert client.closes == 1


@pytest.mark.parametrize("missing_method", ["exchange_context", "recall"])
@pytest.mark.parametrize("failure_policy", ["raise", "host"])
async def test_reader_contract_is_validated_before_network(
    env: tuple[Any, ...], missing_method: str, failure_policy: str
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    reader = SimpleNamespace(
        archive_id=None,
        exchange_context=AsyncMock(return_value=ContextSnapshot()),
        recall=AsyncMock(return_value=RecallResult("", "empty")),
        refresh=AsyncMock(),
    )
    setattr(reader, missing_method, None)
    writer = SuppliedWriter()
    mw = make(env, writer, reader)
    mw._ov_session_engine.config = replace(
        mw._ov_session_engine.config, context_failure=failure_policy
    )
    state = {"messages": env[4].messages}
    with pytest.raises(TypeError, match=f"must implement {missing_method}"):
        await mw.abefore_agent(state, runtime(env))
    assert not env[0].requests and not writer.calls
    assert "ov_context" not in state
    reader.refresh.assert_not_awaited()
