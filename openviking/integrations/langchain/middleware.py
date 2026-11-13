# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""LangGraph agent middleware for OpenViking recall and capture."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, cast

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        BaseMessageChunk,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.types import Command
except ImportError as exc:  # pragma: no cover - exercised by optional import path
    from openviking.integrations.langchain.client import missing_dependency

    raise missing_dependency("langgraph", "langchain/langgraph") from exc

from openviking.integrations.langchain.client import (
    OpenVikingConnection,
    apply_commit_policy,
    call_openviking,
    ensure_client,
    extract_message_text,
    get_latest_user_text,
)
from openviking.integrations.langchain.context import (
    OPENVIKING_CONTEXT_MARKER,
    OpenVikingSessionContextAssembler,
)
from openviking.integrations.langchain.history import langchain_message_to_openviking
from openviking.integrations.langchain.ov_context.assembler import INJECTED, query_text
from openviking.integrations.langchain.ov_context.budget import TokenCounter
from openviking.integrations.langchain.ov_context.commit import (
    CommitConfig,
)
from openviking.integrations.langchain.ov_context.commit import (
    OpenVikingCommitPolicy as OpenVikingCommitPolicy,
)
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.errors import (
    ContextBudgetExceeded,
    ContextHistoryBudgetError,
    IdentityMismatchError,
)
from openviking.integrations.langchain.ov_context.identity import OVIdentity, RunIdentity, valid_id
from openviking.integrations.langchain.ov_context.metrics import Metrics
from openviking.integrations.langchain.ov_context.reader import ContextReaderBase, ContextReadResult
from openviking.integrations.langchain.ov_context.retry import retryable_read
from openviking.integrations.langchain.ov_context.sdk import auth_error
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.state import (
    OVContext,
    OVContextState,
    OVPluginState,
)
from openviking.integrations.langchain.ov_context.writer import MessageWriterBase, WriteResult
from openviking.integrations.langchain.retrievers import OpenVikingRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openviking.integrations.langchain.ov_context.commit import OVCommitPolicy

IdentityResolver = Callable[[OVPluginState, Runtime[Any]], RunIdentity]


_SESSION_ID_ERROR = (
    "OpenVikingContextMiddleware requires a LangGraph session id. Pass "
    'config={"configurable": {"thread_id": "..."}}, set state["session_id"], '
    "or provide session_id_resolver."
)


class OpenVikingContextMiddleware(AgentMiddleware[OVPluginState]):
    """OV-managed session cache for asynchronous LangGraph/DeepAgents execution.

    Five async hooks maintain ``state['ov_context']`` and assemble only its
    native message cache. Legacy synchronous callbacks remain available.
    """

    state_schema = OVPluginState

    def __init__(
        self,
        *,
        client: Any = None,
        retriever: OpenVikingRetriever | None = None,
        url: str | None = None,
        api_key: str | None = None,
        auth_mode: str = "trusted",
        timeout: float = 60.0,
        http_client: Any = None,
        account: str | None = None,
        user: str | None = None,
        user_id: str | None = None,
        actor_peer_id: str | None = None,
        path: str | None = None,
        target_uri: str | list[str] = "",
        limit: int = 5,
        peer_id: str | None = None,
        score_threshold: float | None = None,
        token_budget: int = 128_000,
        identity_resolver: IdentityResolver | None = None,
        session_id_resolver: Callable[[Mapping[str, Any], Any], str] | None = None,
        peer_id_resolver: Callable[[Mapping[str, Any], Any], str | None] | None = None,
        capture_on_after_agent: bool = True,
        commit_on_after_agent: bool = False,
        commit_policy: OpenVikingCommitPolicy | None = None,
        commit_config: CommitConfig | None = None,
        recall_header: str = "Relevant OpenViking context:",
        include_active_messages: bool = False,
        context_config: OVContextConfig | None = None,
        token_counter: TokenCounter | None = None,
        memory_policy: dict[str, Any] | None = None,
        metrics: Metrics | None = None,
        message_writer: MessageWriterBase | None = None,
        context_reader: ContextReaderBase | None = None,
        session_commit_policy: OVCommitPolicy | None = None,
    ) -> None:
        super().__init__()
        if commit_policy is not None and not isinstance(commit_policy, OpenVikingCommitPolicy):
            raise TypeError("commit_policy must be an OpenVikingCommitPolicy")
        if commit_config is not None and not isinstance(commit_config, CommitConfig):
            raise TypeError("commit_config must be a CommitConfig")
        if commit_policy is not None and commit_policy.has_session_policy:
            if session_commit_policy is not None:
                raise ValueError(
                    "Pass OpenVikingCommitPolicy once, either as "
                    "commit_policy or session_commit_policy"
                )
            session_commit_policy = commit_policy
        if commit_policy is None and session_commit_policy is None:
            if commit_on_after_agent:
                commit_policy = OpenVikingCommitPolicy(mode="always")
            elif commit_config is not None:
                session_commit_policy = OpenVikingCommitPolicy(commit_config)
        self._client = client
        self._connection = OpenVikingConnection(
            client=client,
            url=url,
            api_key=api_key,
            auth_mode=auth_mode,
            timeout=timeout,
            http_client=http_client,
            account=account,
            user=user,
            user_id=user_id,
            actor_peer_id=actor_peer_id,
            path=path,
        )
        self.retriever = retriever or OpenVikingRetriever(
            client=client,
            url=url,
            api_key=api_key,
            account=account,
            user=user,
            user_id=user_id,
            actor_peer_id=actor_peer_id,
            path=path,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            search_mode="search",
        )
        self.assembler = OpenVikingSessionContextAssembler(
            client=client,
            retriever=self.retriever,
            url=url,
            api_key=api_key,
            account=account,
            user=user,
            user_id=user_id,
            actor_peer_id=actor_peer_id,
            path=path,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            token_budget=token_budget,
            include_session_context=True,
            include_active_messages=include_active_messages,
            include_recall=True,
            recall_header=recall_header,
        )
        self.identity_resolver = identity_resolver
        self.session_id_resolver = session_id_resolver
        self.peer_id = peer_id
        self.peer_id_resolver = peer_id_resolver
        self.capture_on_after_agent = capture_on_after_agent
        self.commit_policy = commit_policy
        self.recall_header = recall_header
        self._captured_signatures: dict[tuple[str, str], tuple[str, ...]] = {}
        self._pending_context_parts: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._ov_session_engine = SessionEngine(
            self._connection,
            context_config or OVContextConfig(context_window=token_budget),
            token_counter,
            memory_policy,
            message_writer=message_writer,
            context_reader=context_reader,
            session_commit_policy=session_commit_policy,
            metrics=metrics,
        )
        if isinstance(session_commit_policy, OpenVikingCommitPolicy):
            if not session_commit_policy.has_session_policy:
                raise ValueError("session_commit_policy requires a CommitConfig")
            session_commit_policy.bind_client(self._ov_session_engine.client)
        self._ov_metrics = self._ov_session_engine.metrics
        if self._ov_session_engine.config.tool_results:
            from openviking.integrations.langchain.ov_context.tool_readback import (
                make_session_tools,
            )

            self.tools = make_session_tools(self._ov_session_engine, self._identity)

    def _identity(self, state: Mapping[str, Any], runtime: Runtime[Any]) -> RunIdentity:
        if self.identity_resolver is not None:
            identity = self.identity_resolver(cast(OVPluginState, state), runtime)
        else:
            identity = _nested_get(getattr(runtime, "context", None), "identity")
        if not isinstance(identity, RunIdentity):
            raise ValueError(
                "Provide a complete RunIdentity through identity_resolver "
                "or Runtime.context.identity"
            )
        # Validate again at the trust boundary, including custom resolver results.
        for value in vars(identity).values():
            valid_id(value)
        return identity

    def _context(
        self,
        state: Mapping[str, Any],
        runtime: Runtime[Any],
        *,
        copy_messages: bool = True,
        identity: RunIdentity | None = None,
    ) -> OVContext:
        identity = identity if identity is not None else self._identity(state, runtime)
        value = state.get("ov_context")
        context = (
            OVContext.from_state(cast(OVContextState, value), copy_messages=copy_messages)
            if value
            else OVContext(session_id=identity.session_id, owner=identity.session_key)
        )
        if context.session_id != identity.session_id or context.owner != identity.session_key:
            raise IdentityMismatchError("OVContext belongs to a different session/identity")
        self._ov_session_engine.recover_commit(context, persisted=True)
        return context

    def _run_peer(self, state: OVPluginState, runtime: Runtime[Any], identity: RunIdentity) -> str:
        # An override is an explicit DeepAgent authorization decision, not state data.
        peer = self.peer_id_resolver(state, runtime) if self.peer_id_resolver else self.peer_id
        peer = identity.actor_peer_id if peer is None else peer
        valid_id(peer)
        return peer

    def _check_run(self, context: OVContext, identity: RunIdentity) -> None:
        if context.context_status == "uninitialized" or not context.run_id:
            raise ValueError("OVContext was not initialized by abefore_agent")
        if context.run_id != identity.run_id:
            raise ValueError("Request belongs to a different Run")

    async def abefore_agent(self, state: OVPluginState, runtime: Runtime[Any]) -> dict[str, Any]:
        engine = self._ov_session_engine
        with engine.observe("before_agent.total") as sample:
            engine.require_io()
            identity = self._identity(state, runtime)
            context = self._context(state, runtime, identity=identity)
            run_id = identity.run_id
            peer_id = self._run_peer(state, runtime, identity)
            sample.update(
                session_id=context.session_id,
                run_id=run_id,
                account_id=identity.account_id,
                user_id=identity.user_id,
                actor_peer_id=identity.actor_peer_id,
            )
            active = engine.active.get(context.session_key)
            if active and active != run_id:
                raise RuntimeError("Serialize runs using the same OV session")
            engine.active[context.session_key] = run_id
            wrote = False
            try:
                raw = state.get("messages") or []
                current = next(
                    (
                        m
                        for m in reversed(raw)
                        if isinstance(m, HumanMessage) and not m.additional_kwargs.get(INJECTED)
                    ),
                    None,
                )
                if current is None:
                    raise ValueError("Run requires a native HumanMessage")
                # A new Middleware may reuse a ready checkpoint. Its borrowed
                # SDK still needs initialization before external Writer/Reader I/O.
                await engine.client.get()
                engine.invalidate_tokens(context)
                if context.context_status == "uninitialized":
                    metadata = await engine.ensure_session(
                        context.session_id, identity=identity.sdk_identity, run_id=run_id
                    )
                    engine.observe_commit_count(
                        context, engine._session_commit_count(metadata, context)
                    )
                    engine._remember_auto_commit_policy(context, metadata)
                    candidate = await engine.refresh(
                        context, initial=True, run_id=run_id, identity=identity.sdk_identity
                    )
                    engine.apply_context(context, candidate)
                context.begin_run(run_id, current)
                wrote = True
                await engine.write(context, raw, peer_id, identity=identity.sdk_identity)
            except BaseException as exc:
                if not wrote and engine.config.retry_attempts > 0:
                    # No write was attempted, so OV and the checkpoint still
                    # agree. With retries enabled the host may retry without
                    # a restore; retry_attempts=0 keeps the legacy behaviour.
                    try:
                        exc.ov_state_consistent = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                engine.invalidate_tokens(context)
                engine.active.pop(context.session_key, None)
                raise
            return {"ov_context": context.to_state()}

    async def _write_hook(
        self,
        state: OVPluginState,
        runtime: Runtime[Any],
        *,
        after_write: Callable[[OVContext, WriteResult], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        engine = self._ov_session_engine
        raw = state.get("messages") or []
        value = state.get("ov_context")
        unchanged = value is not None and value.get("write_index", 0) == len(raw)
        identity = self._identity(state, runtime)
        context = self._context(state, runtime, copy_messages=not unchanged, identity=identity)
        self._check_run(context, identity)
        peer_id = self._run_peer(state, runtime, identity)
        if unchanged:
            return {}
        result = await engine.write(context, raw, peer_id, identity=identity.sdk_identity)
        if result is not None and after_write is not None:
            await after_write(context, result)
        return {"ov_context": context.to_state()}

    async def _pending_tokens_after_write(
        self, context: OVContext, identity: OVIdentity
    ) -> int | None:
        """An optional policy read must not roll back an acknowledged message write."""
        engine = self._ov_session_engine
        try:
            return await engine.pending_tokens(context, identity=identity)
        except Exception as exc:
            if engine.config.retry_attempts <= 0 or not retryable_read(exc):
                raise
            return None

    async def abefore_model(self, state: OVPluginState, runtime: Runtime[Any]) -> dict[str, Any]:
        engine = self._ov_session_engine
        identity = self._identity(state, runtime)
        update = await self._write_hook(state, runtime)
        if not update:
            return update
        context = OVContext.from_state(cast(OVContextState, update["ov_context"]))
        if engine.session_commit_policy is None:
            return update
        # Tool results have merged; the commit policy needs the session's live pending tokens.
        pending = await self._pending_tokens_after_write(context, identity.sdk_identity)
        await engine.commit(context, pending_tokens=pending, identity=identity.sdk_identity)
        return {"ov_context": context.to_state()}

    async def aafter_model(self, state: OVPluginState, runtime: Runtime[Any]) -> dict[str, Any]:
        engine = self._ov_session_engine
        identity = self._identity(state, runtime)
        if engine.config.batch_tool_messages:
            context = self._context(state, runtime, copy_messages=False, identity=identity)
            self._check_run(context, identity)
            raw = state.get("messages") or []
            if (
                len(raw) == context.write_index + 1
                and isinstance(raw[-1], AIMessage)
                and not isinstance(raw[-1], BaseMessageChunk)
                and raw[-1].tool_calls
            ):
                # Only an explicit local validation capability permits deferral.
                # A host writer without it retains write-before-tools behavior.
                if callable(inspect.getattr_static(engine.writer, "validate", None)):
                    validate = engine.writer.validate
                    if not inspect.iscoroutinefunction(validate):
                        validate(
                            [raw[-1]],
                            identity=identity.sdk_identity,
                            peer_id=self._run_peer(state, runtime, identity),
                        )
                        # The graph checkpoints the original AIMessage. Do not
                        # advance the confirmed cursor or model view until the
                        # call and results are actually persisted in before_model.
                        return {}
        if engine.session_commit_policy is None:
            return await self._write_hook(state, runtime)

        async def after_write(context: OVContext, result: WriteResult) -> None:
            # Tool calls execute after this Hook. Commit only after a final
            # model response, using the session's freshly persisted metrics.
            if result.model_messages and not getattr(result.model_messages[-1], "tool_calls", None):
                pending = await self._pending_tokens_after_write(context, identity.sdk_identity)
                await engine.commit(context, pending_tokens=pending, identity=identity.sdk_identity)

        return await self._write_hook(state, runtime, after_write=after_write)

    async def _compress_context(
        self,
        context: OVContext,
        *,
        identity: OVIdentity,
        pending_tokens: int | None,
        token_budget: int,
        request: ModelRequest[Any],
        probe_first: bool = False,
    ) -> None:
        """Complete required capacity repair before admitting a model request."""
        engine = self._ov_session_engine
        try:
            if probe_first:
                # Required probes share compression's error/deadline boundary;
                # they must never become optional-OV fallback model calls.
                meta = await engine.check_summary(
                    context, identity=identity, token_budget=token_budget, model_request=request
                )
                pending_tokens = meta.pending_tokens
                try:
                    prepared = engine.assembler.prepare(request, context)
                except ContextBudgetExceeded:
                    pass
                else:
                    if not prepared.capacity_reached:
                        return
            previous_archive = context.archive_id
            if not await engine.commit(
                context, identity=identity, pending_tokens=pending_tokens, trigger="capacity"
            ):
                raise ContextBudgetExceeded(
                    "Current turn exceeds capacity; no safe history to compress"
                )
            await engine.wait_summary(
                context, identity=identity, token_budget=token_budget, model_request=request
            )
            if context.archive_id == previous_archive:
                raise ContextBudgetExceeded("Compression did not produce a usable archive")
        except ContextBudgetExceeded:
            raise
        except Exception as exc:
            if isinstance(exc, IdentityMismatchError) or auth_error(exc):
                raise
            raise ContextBudgetExceeded(
                f"Capacity compression failed ({type(exc).__name__})"
            ) from exc

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ExtendedModelResponse[Any]:
        engine = self._ov_session_engine
        identity = self._identity(request.state, request.runtime)
        context = self._context(request.state, request.runtime, identity=identity)
        self._check_run(context, identity)
        history = engine.assembler.history_budget(request)
        try:
            meta = await engine.maybe_check_summary(
                context,
                identity=identity.sdk_identity,
                token_budget=history.history_target_tokens,
                model_request=request,
            )
        except Exception as exc:
            if (
                isinstance(exc, (IdentityMismatchError, ContextBudgetExceeded))
                or isinstance(exc, ContextHistoryBudgetError)
                or auth_error(exc)
                or engine.config.context_failure == "raise"
            ):
                raise
            context.context_status = "reload_required"
            meta = ContextReadResult(context.archive_id)
        if not context.recall_done:
            assert engine.context_reader is not None
            assert context.original_user is not None
            with engine.observe("memory.recall", context) as sample:
                recalled = await engine.context_reader.recall(
                    query_text(context.original_user),
                    identity=identity.sdk_identity,
                    session_id=context.session_id,
                )
                context.inject_recall(recalled.content, recalled.status)
                sample.update(status=recalled.status, count=recalled.count)
        prepared = None
        try:
            with engine.observe("context.assemble", context):
                prepared = engine.assembler.prepare(request, context, history_budget=history)
        except ContextBudgetExceeded:
            pass
        if prepared is None or prepared.capacity_reached:
            await self._compress_context(
                context,
                identity=identity.sdk_identity,
                pending_tokens=meta.pending_tokens,
                token_budget=history.history_target_tokens,
                request=request,
                probe_first=not meta.probe_performed,
            )
            with engine.observe("context.assemble", context):
                prepared = engine.assembler.prepare(request, context, history_budget=history)
        elif engine.session_commit_policy is not None:
            await engine.commit(
                context, identity=identity.sdk_identity, pending_tokens=meta.pending_tokens
            )
        elif (
            self.commit_policy is not None
            and self.commit_policy.mode == "pending_tokens"
            and meta.pending_tokens is not None
            and meta.pending_tokens >= self.commit_policy.pending_token_threshold
        ):
            await engine.commit(context, identity=identity.sdk_identity)
        updated = prepared.request
        budget_fields = {
            "estimated_input_tokens": prepared.input_tokens,
            "input_limit": prepared.budget.input_limit,
            "base_context_limit": prepared.budget.base_context_limit,
            "capacity_limit": prepared.budget.capacity_limit,
            "history_target_tokens": prepared.history_target_tokens,
            "fixed_tokens": prepared.fixed_tokens,
            "base_context_tokens": prepared.base_context_tokens,
            "counter": getattr(engine.assembler.token_counter, "name", "custom"),
        }
        describe = getattr(engine.assembler.token_counter, "diagnostics", None)
        if callable(describe):
            budget_fields["estimation"] = describe(updated)
        self._ov_metrics.emit(
            "context.estimate",
            session_id=context.session_id,
            run_id=context.run_id,
            account_id=identity.account_id,
            user_id=identity.user_id,
            actor_peer_id=identity.actor_peer_id,
            **budget_fields,
        )
        # Callback metadata follows this invocation's async context, including model retries.
        from langchain_core.runnables.config import ensure_config, var_child_runnable_config

        config = ensure_config()
        token = var_child_runnable_config.set(
            {**config, "metadata": {**config.get("metadata", {}), "ov_input_budget": budget_fields}}
        )
        try:
            with engine.observe("model.handler", context):
                response = await handler(updated)
        finally:
            var_child_runnable_config.reset(token)
        update = {"ov_context": context.to_state()}
        if isinstance(response, ExtendedModelResponse):
            command = response.command or Command()
            if command.update is not None and not isinstance(command.update, Mapping):
                raise TypeError("OV context requires mapping command updates")
            if command.update is not None and "ov_context" in command.update:
                raise ValueError("Conflicting ov_context update from model handler")
            command = replace(command, update={**(command.update or {}), **update})
            return ExtendedModelResponse(response.model_response, command)
        if isinstance(response, AIMessage):
            response = ModelResponse(result=[response])
        return ExtendedModelResponse(response, Command(update=update))

    async def aafter_agent(self, state: OVPluginState, runtime: Runtime[Any]) -> dict[str, Any]:
        engine = self._ov_session_engine
        identity = self._identity(state, runtime)
        context = self._context(state, runtime, identity=identity)
        if context.run_id is None and context.last_completed_run == identity.run_id:
            if context.write_index == len(state.get("messages") or []):
                return {}
        self._check_run(context, identity)
        peer_id = self._run_peer(state, runtime, identity)
        try:
            await engine.write(
                context, state.get("messages") or [], peer_id, identity=identity.sdk_identity
            )
            if context.run_id and context.last_completed_run != context.run_id:
                context.completed_runs += 1
                context.last_completed_run = context.run_id
            # The final write still needs the Run identity. Once it succeeds,
            # restore the user message and release the active-Run retention
            # anchor before evaluating the completed-Run commit policy.
            context.end_run()
            # A completed Run must archive its final outcome; retention is for in-Run compression.
            if engine.session_commit_policy is not None:
                await engine.commit(
                    context,
                    pending_tokens=await self._pending_tokens_after_write(
                        context, identity.sdk_identity
                    ),
                    identity=identity.sdk_identity,
                    keep_recent_count=0,
                )
            else:
                policy = self.commit_policy
                if policy and policy.mode == "always":
                    await engine.commit(
                        context, identity=identity.sdk_identity, keep_recent_count=0
                    )
        finally:
            engine.invalidate_tokens(context)
            engine.active.pop(context.session_key, None)
        return {"ov_context": context.to_state()}

    async def acleanup_run(
        self, graph: CompiledStateGraph[Any, Any, Any, Any], config: RunnableConfig
    ) -> None:
        """End an abandoned Run without retrying potentially ambiguous writes.

        Original messages and the confirmed cursor stay in the checkpoint,
        including deferred tool calls. Resume a paused/failed graph before
        cleanup; cleanup ends its Run identity and is not a flush operation.
        """
        engine = self._ov_session_engine
        state = (await graph.aget_state(config)).values
        if not state.get("ov_context"):
            return
        context = OVContext.from_state(state["ov_context"])
        engine.recover_commit(context)
        engine.invalidate_tokens(context)
        context.end_run()
        engine.active.pop(context.session_key, None)
        await graph.aupdate_state(config, {"ov_context": context.to_state()})
        engine.recover_commit(context, persisted=True)

    async def aclose(self) -> None:
        try:
            await self._ov_session_engine.client.close()
        finally:
            self._ov_session_engine._refresh_progress.clear()
            self._ov_session_engine._token_receipts.clear()

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        query = get_latest_user_text(request.messages)
        if not query:
            return handler(request)
        session_id = self._resolve_session_id(
            getattr(request, "state", {}) or {},
            getattr(request, "runtime", None),
        )
        peer_id = self._resolve_peer_id(
            getattr(request, "state", {}) or {},
            getattr(request, "runtime", None),
        )
        pending_key = _capture_key(session_id, peer_id)
        self._pending_context_parts.pop(pending_key, None)
        assembled = self.assembler.assemble(
            session_id=session_id,
            query=query,
        )
        context_block = assembled.block
        if not context_block:
            return handler(request)
        if assembled.context_parts:
            self._pending_context_parts[pending_key] = assembled.context_parts

        system_message = request.system_message
        if system_message is None:
            updated_system = SystemMessage(content=context_block)
        else:
            content = extract_message_text(system_message.content)
            updated_system = SystemMessage(content=f"{content}\n\n{context_block}".strip())
        try:
            return handler(request.override(system_message=updated_system))
        except Exception:
            self._pending_context_parts.pop(pending_key, None)
            raise

    def after_agent(self, state: OVPluginState, runtime: Any) -> dict[str, Any] | None:
        if not self.capture_on_after_agent:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        session_id = self._resolve_session_id(state, runtime)
        peer_id = self._resolve_peer_id(state, runtime)
        capture_key = _capture_key(session_id, peer_id)
        previous_signatures = self._captured_signatures.get(capture_key, ())
        current_signatures = tuple(_message_signature(message) for message in messages)

        if current_signatures == previous_signatures:
            self._pending_context_parts.pop(capture_key, None)
            return None
        start = 0
        if (
            previous_signatures
            and len(current_signatures) > len(previous_signatures)
            and current_signatures[: len(previous_signatures)] == previous_signatures
        ):
            start = len(previous_signatures)

        client = ensure_client(self._connection)
        added = 0
        pending_context_parts = list(self._pending_context_parts.pop(capture_key, []))
        for message in messages[start:]:
            if OPENVIKING_CONTEXT_MARKER in _message_content(message):
                continue
            payloads = langchain_message_to_openviking(message)
            for payload in payloads:
                if pending_context_parts and payload["role"] == "assistant":
                    payload["parts"].extend(pending_context_parts)
                    pending_context_parts = []
                call_openviking(
                    client,
                    "add_message",
                    session_id=session_id,
                    role=payload["role"],
                    parts=payload["parts"],
                    peer_id=peer_id,
                )
                added += 1
        self._captured_signatures[capture_key] = current_signatures
        if added:
            apply_commit_policy(client, session_id, self.commit_policy)
        return None

    def _resolve_session_id(self, state: Mapping[str, Any], runtime: Any) -> str:
        if self.session_id_resolver:
            resolved = _normalize_session_id(self.session_id_resolver(state, runtime))
            if resolved:
                return resolved
            raise ValueError(_SESSION_ID_ERROR)
        candidates = [
            state.get("thread_id"),
            state.get("session_id"),
            _nested_get(getattr(runtime, "context", None), "thread_id"),
            _nested_get(getattr(runtime, "config", None), "configurable", "thread_id"),
            _nested_get(getattr(runtime, "config", None), "configurable", "session_id"),
            _nested_get(getattr(runtime, "execution_info", None), "thread_id"),
        ]
        for candidate in candidates:
            resolved = _normalize_session_id(candidate)
            if resolved:
                return resolved
        raise ValueError(_SESSION_ID_ERROR)

    def _resolve_peer_id(self, state: Mapping[str, Any], runtime: Any) -> str | None:
        if self.peer_id_resolver:
            return _normalize_peer_id(self.peer_id_resolver(state, runtime))
        candidates = [
            state.get("peer_id"),
            state.get("peerId"),
            _nested_get(getattr(runtime, "context", None), "peer_id"),
            _nested_get(getattr(runtime, "context", None), "peerId"),
            _nested_get(getattr(runtime, "config", None), "configurable", "peer_id"),
            _nested_get(getattr(runtime, "config", None), "configurable", "peerId"),
            self.peer_id,
        ]
        for candidate in candidates:
            resolved = _normalize_peer_id(candidate)
            if resolved:
                return resolved
        return None

    def _ensure_session(self, client: Any, session_id: str) -> None:
        try:
            call_openviking(client, "create_session", session_id=session_id)
        except Exception:
            logger.debug("OpenViking LangGraph middleware session ensure failed", exc_info=True)
            pass


def _nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _normalize_session_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_peer_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _capture_key(session_id: str, peer_id: str | None) -> tuple[str, str]:
    return (session_id, peer_id or "")


def _message_role(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, BaseMessage):
        if message.type == "human":
            return "user"
        if message.type == "ai":
            return "assistant"
        return message.type
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "")
        return {"human": "user", "ai": "assistant"}.get(role, role)
    return str(getattr(message, "role", "") or getattr(message, "type", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return extract_message_text(message.get("content"))
    return extract_message_text(getattr(message, "content", ""))


def _message_stable_id(message: Any) -> str | None:
    if isinstance(message, dict):
        value = message.get("id")
    else:
        value = getattr(message, "id", None)
    return str(value) if value else None


def _message_signature(message: Any) -> str:
    return _stable_json(
        {
            "id": _message_stable_id(message),
            "role": _message_role(message),
            "content": _message_content(message),
            "tool_calls": _message_tool_calls(message),
            "tool_result": _message_tool_result(message),
        }
    )


def _message_tool_calls(message: Any) -> Any:
    if isinstance(message, AIMessage):
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            calls = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls") or []
        return calls
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def _message_tool_result(message: Any) -> dict[str, Any]:
    if isinstance(message, ToolMessage):
        return {
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None),
            "status": getattr(message, "status", None),
        }
    if isinstance(message, dict):
        return {
            "tool_call_id": message.get("tool_call_id") or message.get("tool_id"),
            "name": message.get("name") or message.get("tool_name"),
            "output": message.get("tool_output") or message.get("output"),
            "status": message.get("status") or message.get("tool_status"),
        }
    return {
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
        "status": getattr(message, "status", None),
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
