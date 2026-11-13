# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Initialize DeepAgents with OV-owned context and host-supplied I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import httpx
from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.client import OpenVikingConnection
from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    ContextReaderBase,
    MessageWriterBase,
    Metrics,
    OVContextConfig,
    RunIdentity,
    TokenCounter,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import ClientAuthMode


@dataclass(frozen=True)
class RequestContext:
    """Per-invocation identity; use a new run_id for each user request."""

    identity: RunIdentity


DemoAgent = CompiledStateGraph[
    AgentState[Any], RequestContext, InputAgentState, OutputAgentState[Any]
]


def create_demo_agent(
    *,
    model: str,
    connection: OpenVikingConnection,
    context_config: OVContextConfig,
    message_writer: MessageWriterBase,
    context_reader: ContextReaderBase,
    tools: Sequence[BaseTool] = (),
    checkpointer: BaseCheckpointSaver[Any],
    token_counter: TokenCounter | None = None,
    metrics: Metrics | None = None,
    commit_config: CommitConfig | None = None,
) -> tuple[DemoAgent, OpenVikingContextMiddleware]:
    """Return a compiled agent and its Middleware without sending an OV request.

    model is a provider:model spec so DeepAgents uses the registered profile.
    connection.client is this Run's initialized native SDK (or compatible client).
    The application owns the injected checkpoint store and HTTP connection pool.
    I/O implementations receive the authenticated identity on every invocation.
    Profile registration applies to other agents using this model in the process.
    commit_config selects the new thresholds; None explicitly selects legacy triggers.
    """
    if checkpointer is None:
        raise ValueError("Provide a checkpoint store shared across Run instances")
    if connection.client is None:
        raise ValueError("Provide one host-created client shared by Middleware, Writer and Reader")
    # Let OV own summaries. Register before create_deep_agent builds its stack.
    register_harness_profile(
        model,
        HarnessProfile(excluded_middleware=frozenset({"SummarizationMiddleware"})),
    )

    ov_middleware = OpenVikingContextMiddleware(
        client=connection.client,
        url=connection.url,
        api_key=connection.api_key,
        context_config=context_config,
        commit_config=commit_config,
        token_counter=token_counter or ApproximateTokenCounter(),
        message_writer=message_writer,
        context_reader=context_reader,
        metrics=metrics,
    )
    agent = create_deep_agent(
        model=model,
        tools=list(tools),
        middleware=[
            # Keep filesystem tools without an independent context-eviction path.
            FilesystemMiddleware(
                tool_token_limit_before_evict=None,
                human_message_token_limit_before_evict=None,
            ),
            ov_middleware,
        ],
        context_schema=RequestContext,
        checkpointer=checkpointer,
    )
    return agent, ov_middleware


async def invoke_demo_run(
    *,
    model: str,
    ov_url: str,
    api_key: str,
    http_client: httpx.AsyncClient,
    checkpointer: BaseCheckpointSaver[Any],
    identity: RunIdentity,
    messages: Sequence[AnyMessage],
    message_writer_factory: Callable[[AsyncHTTPClient], MessageWriterBase],
    context_reader_factory: Callable[[AsyncHTTPClient], ContextReaderBase],
    context_config: OVContextConfig,
    tools: Sequence[BaseTool] = (),
    token_counter: TokenCounter | None = None,
    auth_mode: ClientAuthMode = "api_key",
    commit_config: CommitConfig | None = None,
) -> dict[str, Any]:
    """Build one Run's graph, middleware and SDK over application-owned resources.

    The application serializes Runs for one session_key and keeps the supplied
    checkpoint store and HTTP client alive across calls. A failed Run may be
    retried with the same identity.run_id; no old Middleware is required.
    """
    sdk = AsyncHTTPClient(
        url=ov_url,
        api_key=api_key,
        profile_enabled=False,
        http_client=http_client,
        auth_mode=auth_mode,
    )
    await sdk.initialize()
    middleware: OpenVikingContextMiddleware | None = None
    try:
        agent, middleware = create_demo_agent(
            model=model,
            connection=OpenVikingConnection(client=sdk),
            context_config=context_config,
            commit_config=commit_config,
            message_writer=message_writer_factory(sdk),
            context_reader=context_reader_factory(sdk),
            checkpointer=checkpointer,
            tools=tools,
            token_counter=token_counter,
        )
        config: RunnableConfig = {"configurable": {"thread_id": json.dumps(identity.session_key)}}
        input_state: InputAgentState = {"messages": list(messages)}
        return cast(
            dict[str, Any],
            await agent.ainvoke(input_state, config=config, context=RequestContext(identity)),
        )
    finally:
        if middleware is not None:
            await middleware.aclose()
        await sdk.close()  # Detaches this Run's SDK; the shared pool stays open.
