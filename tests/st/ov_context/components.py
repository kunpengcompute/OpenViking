# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real OV/DeepAgents acceptance for Writer, context and tenant isolation.

Run artifacts are private and must stay outside the repository. Business data
is synthetic with known answers; SDK, storage, tools and model calls are real.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

import httpx
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import InputAgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import SecretStr

from examples.deepagents_ov_context.chat.agent import DemoAgent, RequestContext
from examples.deepagents_ov_context.chat.config import ModelConfig, load_service_api_key
from examples.deepagents_ov_context.chat.model import ArkChatOpenAI
from examples.deepagents_ov_context.chat.native_summary import DisabledNativeSummaryMiddleware
from openviking.integrations._message_content import comparable_content
from openviking.integrations.langchain.middleware import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    ApproximateTokenCounter,
    Metrics,
    OVContext,
    OVContextConfig,
    RunIdentity,
)
from openviking.integrations.langchain.ov_context.assembler import validate_tool_pairs
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking_cli.client.http import AsyncHTTPClient

ACTIVE: contextvars.ContextVar[RunIdentity | None] = contextvars.ContextVar(
    "acceptance_identity", default=None
)
PHASE: contextvars.ContextVar[str] = contextvars.ContextVar("acceptance_phase", default="setup")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def initial_memory_baseline(user_id: str) -> dict[str, str]:
    """Hashes of bundled initialization templates, not extracted memories."""
    from openviking.prompts.manager import PromptManager
    from openviking.session.memory.dataclass import MemoryFile
    from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
    from openviking.session.memory.utils.template_utils import TemplateUtils

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.load_from_directory(str(PromptManager._get_bundled_templates_dir() / "memory"))
    result = {}
    for schema in registry.list_all(include_disabled=False):
        if not schema.filename_template or "{{" in schema.filename_template:
            continue
        defaults = {f.name: f.init_value for f in schema.fields if f.init_value is not None}
        if not defaults or not schema.content_template:
            continue
        directory = TemplateUtils.render(schema.directory, {"user_space": user_id})
        uri = directory + "/" + schema.filename_template
        text = MemoryFileUtils.write(
            MemoryFile(uri=uri, memory_type=schema.memory_type, extra_fields=defaults),
            content_template=schema.content_template,
        )
        result[uri] = sha(MemoryFileUtils.read(text).content)
    return result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def signature(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Compare the persisted OV contract, without local-only native metadata/IDs."""
    values = []
    for m in messages:
        row: dict[str, Any] = {"type": m.type, "content": comparable_content(m.content)}
        if isinstance(m, AIMessage):
            row["tool_calls"] = m.tool_calls
        if isinstance(m, ToolMessage):
            row.update(tool_call_id=m.tool_call_id, name=m.name, status=m.status)
        values.append(row)
    return values


@dataclass(frozen=True)
class Tenant:
    account: str
    user: str
    peer: str
    session: str
    marker: str

    def identity(self, run: str) -> RunIdentity:
        return RunIdentity(self.account, self.user, self.peer, self.session, run)

    @property
    def thread(self) -> str:
        return json.dumps([self.account, self.user, self.peer, self.session])


def validate_model_input(sample: dict[str, Any], tenant: Tenant) -> None:
    system = sample.get("system")
    if not isinstance(system, dict):
        raise AssertionError("System message missing")
    body = str(system["content"])
    for prefix in (
        "SYS_HEADER:order-audit",
        f"TENANT:{tenant.account}/{tenant.user}/{tenant.peer}",
        "BASE_RULE:use_only_this_tenant_data",
        "[OpenViking Context Guide]",
    ):
        check(body.count(prefix) == 1, "System prefix missing/duplicated")
    check(sample["input_tokens"] <= sample["input_limit"], "Model input exceeded window")
    if "actual_input_tokens" in sample:
        check(
            sample["actual_input_tokens"] <= sample["input_limit"],
            "Provider-reported input exceeded window",
        )


class Capture(AgentMiddleware):
    def __init__(
        self, sink: list[dict[str, Any]], config: OVContextConfig, requests: list[ModelRequest[Any]]
    ) -> None:
        self.requests = requests
        self.sink = sink
        self.config = config

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        identity = request.runtime.context.identity
        window = self.config.context_window
        if window is None:
            raise ValueError("Validation requires an explicit context window")
        self.requests.append(request)
        count = ApproximateTokenCounter()(request)
        sample = {
            "owner": identity.session_key,
            "run_id": identity.run_id,
            "system": request.system_message.model_dump(mode="json")
            if request.system_message
            else None,
            "messages": [m.model_dump(mode="json") for m in request.messages],
            "input_tokens": count,
            "input_limit": window - self.config.output_reserve - self.config.safety_margin,
        }
        self.sink.append(sample)
        response = await handler(request)
        reported = []
        for message in response.result:
            if isinstance(message, AIMessage):
                check(not message.invalid_tool_calls, "Model returned invalid tool arguments")
                if message.usage_metadata:
                    reported.append(message.usage_metadata["input_tokens"])
        if reported:
            sample["actual_input_tokens"] = max(reported)
        return response


class Suite:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output
        self.requests: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.inputs: list[dict[str, Any]] = []
        self.model_requests: list[ModelRequest[Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.prefix = args.prefix
        self.cleanup = AsyncExitStack()

    async def __aenter__(self) -> Suite:
        await self.cleanup.__aenter__()
        self.http = await self.cleanup.enter_async_context(
            httpx.AsyncClient(
                timeout=180, trust_env=False, event_hooks={"request": [self.observe_request]}
            )
        )
        self.model_http = await self.cleanup.enter_async_context(
            httpx.AsyncClient(timeout=180, trust_env=False)
        )
        self.sdk = AsyncHTTPClient(
            url=self.args.ov_url,
            auth_mode="trusted",
            api_key=load_service_api_key(key_file=self.args.api_key_file),
            http_client=self.http,
            profile_enabled=False,
        )
        self.cleanup.push_async_callback(self.sdk.close)
        await self.sdk.initialize()
        self.defaults = dict(self.http.headers)
        cfg = ModelConfig.load(self.args.model_config)
        self.model = ArkChatOpenAI(
            model=cfg.chat_model,
            api_key=SecretStr(cfg.api_key),
            base_url=cfg.api_base,
            http_async_client=self.model_http,
            temperature=0,
            max_completion_tokens=8192,
            streaming=True,
            stream_usage=True,
            max_retries=0,
            timeout=180,
            extra_body={"thinking": {"type": "enabled"}},
        )
        save(
            self.output / f"conditions-{self.args.command}.json",
            {
                "source": self.args.source,
                "harness_sha256": sha(Path(__file__).read_text()),
                "python": sys.version,
                "platform": platform.platform(),
                "model": cfg.chat_model,
                "thinking": True,
                "temperature": 0,
                "max_tokens": 8192,
                "ov_url": self.args.ov_url,
                "parallelism": self.args.parallelism,
                "checkpoint": "AsyncSqliteSaver",
                "business_data": "synthetic order ledger, known answers",
                "tool_min_reduction": 0.8,
                "history_min_reduction": 0.7,
                "full_input_min_reduction": 0.5,
                "throughput_min_speedup": 1.5,
                "p95_max_slowdown": 2.5,
                "prefix": self.prefix,
            },
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        suffix = self.args.command + (
            "-" + self.args.worker_phase if self.args.command == "worker" else ""
        )
        save(self.output / f"private/{suffix}-requests.json", self.requests)
        save(self.output / f"private/{suffix}-metrics.json", self.metrics)
        save(self.output / f"private/{suffix}-model-inputs.json", self.inputs)
        save(self.output / f"private/{suffix}-tool-events.json", self.tools)
        await self.cleanup.__aexit__(*exc)

    async def observe_request(self, request: httpx.Request) -> None:
        identity = ACTIVE.get()
        self.requests.append(
            {
                "at": time.monotonic(),
                "phase": PHASE.get(),
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "owner": list(identity.session_key) if identity else None,
                "run_id": identity.run_id if identity else None,
                "headers": {
                    k: request.headers.get(k)
                    for k in (
                        "X-OpenViking-Account",
                        "X-OpenViking-User",
                        "X-OpenViking-Actor-Peer",
                    )
                },
            }
        )

    def metric(self, event: str, **fields: Any) -> None:
        self.metrics.append({"event": event, **fields, "at": time.monotonic()})

    async def case(self, name: str, action: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        start = time.monotonic()
        try:
            data = await asyncio.wait_for(action(), self.args.timeout)
            row = {
                "case": name,
                "status": "passed",
                "seconds": round(time.monotonic() - start, 3),
                "data": data,
            }
        except Exception as exc:
            save(
                self.output / f"private/{name}-error.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            row = {
                "case": name,
                "status": "failed",
                "seconds": round(time.monotonic() - start, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.results.append(row)
        save(self.output / f"results-{self.args.command}.json", self.results)
        print(
            json.dumps({k: v for k, v in row.items() if k != "data"}, ensure_ascii=False),
            flush=True,
        )

    async def writer(self) -> dict[str, Any]:
        tenant = Tenant(
            self.prefix + "-writer", "alice", "audit", "writer", "WRITER-" + self.prefix
        )
        identity = tenant.identity("write")
        token = ACTIVE.set(identity)
        try:
            await self.sdk.create_session(identity.session_id, identity=identity.sdk_identity)
            await self.sdk.add_message(
                identity.session_id,
                role="user",
                content="single add_message " + tenant.marker,
                identity=identity.sdk_identity,
            )
            initial = await self.sdk.get_session_context(
                identity.session_id, identity=identity.sdk_identity
            )
            check(tenant.marker in json.dumps(initial), "add_message not persisted")
            writer = OVMessageWriter(self.sdk)
            long = tenant.marker + "\n" + "order,amount,status\n" + "A-42,137,confirmed\n" * 5000
            raw: list[BaseMessage] = [
                HumanMessage(id=f"u-{i}", content=f"batch-{i}") for i in range(103)
            ]
            raw += [
                AIMessage(
                    id="parallel-calls",
                    content="",
                    tool_calls=[
                        {"id": f"c-{i}", "name": "load_ledger", "args": {"index": i}}
                        for i in range(3)
                    ],
                )
            ]
            raw += [
                ToolMessage(
                    id=f"t-{i}", name="load_ledger", tool_call_id=f"c-{i}", content=long + str(i)
                )
                for i in range(3)
            ]
            before = [m.model_dump(mode="json") for m in raw]
            result = await writer.write(identity.session_id, raw, 0, identity=identity.sdk_identity)
            check(result.write_index == len(raw), "write cursor incomplete")
            check(
                [m.model_dump(mode="json") for m in raw] == before,
                "Writer changed original messages",
            )
            reductions, refs = [], []
            for m in result.model_messages[-3:]:
                check(isinstance(m, ToolMessage), "Tool result missing")
                reductions.append(1 - len(str(m.content)) / len(long))
                check(reductions[-1] >= 0.8, "Tool reduction below 80%")
            remote = await self.sdk.get_session_context(
                identity.session_id, token_budget=500_000, identity=identity.sdk_identity
            )
            parts = [
                p
                for m in remote["messages"]
                for p in m["parts"]
                if p["type"] == "tool" and p.get("tool_output_ref")
            ]
            check(len(parts) == 3, "Parallel stored outputs missing")
            for p in parts:
                result_id = p["tool_output_ref"].rsplit("/", 1)[-1]
                chunks = []
                for offset in range(0, p["tool_output_original_chars"], 20_000):
                    body = await self.sdk.read_tool_result(
                        identity.session_id,
                        result_id,
                        offset=offset,
                        limit=20_000,
                        identity=identity.sdk_identity,
                    )
                    chunks.append(body["content"])
                original = "".join(chunks)
                check(sha(original) == p["tool_output_sha256"], "Stored original hash mismatch")
                refs.append(p["tool_output_ref"])
            request_count = len(self.requests)
            repeated = await writer.write(
                identity.session_id, raw, result.write_index, identity=identity.sdk_identity
            )
            check(
                not repeated.model_messages and len(self.requests) == request_count,
                "No-delta Writer sent duplicate request",
            )
            protected_body = await self.sdk.read_tool_result(
                identity.session_id,
                refs[0].rsplit("/", 1)[-1],
                limit=20_000,
                identity=identity.sdk_identity,
            )
            protected_text = json.dumps(protected_body, ensure_ascii=False)
            raw.extend(
                [
                    AIMessage(
                        id="read-call",
                        content="",
                        tool_calls=[
                            {
                                "id": "protected",
                                "name": "openviking_tool_result_read",
                                "args": {"ref": refs[0], "limit": 20_000},
                            }
                        ],
                    ),
                    ToolMessage(
                        id="read-result",
                        tool_call_id="protected",
                        name="openviking_tool_result_read",
                        content=protected_text,
                    ),
                ]
            )
            read_result = await writer.write(
                identity.session_id, raw, result.write_index, identity=identity.sdk_identity
            )
            check(read_result.model_messages[-1] is raw[-1], "Protected OV tool object replaced")
            check(
                read_result.model_messages[-1].content == protected_text,
                "Protected OV content compressed",
            )
            save(self.output / "private/writer-remote.json", remote)
            return {
                "raw_messages": len(raw),
                "cursor": read_result.write_index,
                "single_add": True,
                "tool_reductions": reductions,
                "protected_chars": len(protected_text),
                "original_hash_verified": 3,
            }
        finally:
            ACTIVE.reset(token)

    def graph(
        self,
        tenant: Tenant,
        saver: BaseCheckpointSaver[Any],
        *,
        commit: bool = False,
        small: bool = False,
    ) -> tuple[DemoAgent, OpenVikingContextMiddleware]:
        config = OVContextConfig(
            context_window=32_000 if small else 128_000,
            output_reserve=8192,
            safety_margin=512,
            capacity_ratio=0.7,
        )
        mw = OpenVikingContextMiddleware(
            client=self.sdk,
            message_writer=OVMessageWriter(self.sdk),
            context_reader=OVContextReader(
                self.sdk,
                MemoryReaderConfig(
                    recall_timeout=30,
                    context_token_budget=200_000,
                    enabled=False,
                    experience_enabled=False,
                ),
            ),
            context_config=config,
            metrics=Metrics(self.metric),
            commit_config=CommitConfig(128_000, 0.7, 100, 2) if commit else None,
        )

        @tool
        async def load_ledger(runtime: ToolRuntime[RequestContext, dict[str, Any]]) -> str:
            """Load this tenant's order ledger. No arguments or other tools needed."""
            if runtime.context is None:
                raise ValueError("Missing tool invocation context")
            identity = runtime.context.identity
            check(identity.session_key == tenant.identity("x").session_key, "Tool identity mixed")
            self.tools.append(
                {
                    "event": "start",
                    "tool": "load_ledger",
                    "owner": identity.session_key,
                    "run_id": identity.run_id,
                    "call_id": runtime.tool_call_id,
                    "at": time.monotonic(),
                }
            )
            await asyncio.sleep(0.4)
            value: str = (
                f"ledger_marker={tenant.marker}-LEDGER\norder_id=O-42\namount=137\n"
                + "A-42,137,confirmed\n" * 2000
            )
            self.tools.append(
                {
                    "event": "end",
                    "tool": "load_ledger",
                    "owner": identity.session_key,
                    "run_id": identity.run_id,
                    "call_id": runtime.tool_call_id,
                    "at": time.monotonic(),
                }
            )
            return value

        @tool
        async def lookup_delivery(runtime: ToolRuntime[RequestContext, dict[str, Any]]) -> str:
            """Look up this tenant's delivery marker. Independent of load_ledger."""
            if runtime.context is None:
                raise ValueError("Missing tool invocation context")
            identity = runtime.context.identity
            check(identity.session_key == tenant.identity("x").session_key, "Tool identity mixed")
            self.tools.append(
                {
                    "event": "start",
                    "tool": "lookup_delivery",
                    "owner": identity.session_key,
                    "run_id": identity.run_id,
                    "call_id": runtime.tool_call_id,
                    "at": time.monotonic(),
                }
            )
            await asyncio.sleep(0.05)
            self.tools.append(
                {
                    "event": "end",
                    "tool": "lookup_delivery",
                    "owner": identity.session_key,
                    "run_id": identity.run_id,
                    "call_id": runtime.tool_call_id,
                    "at": time.monotonic(),
                }
            )
            return f"delivery_marker={tenant.marker}-DELIVERY; carrier=SF; eta=tomorrow"

        system = f"SYS_HEADER:order-audit\nTENANT:{tenant.account}/{tenant.user}/{tenant.peer}\nBASE_RULE:use_only_this_tenant_data\n你是订单核对助手。按当前用户要求操作，工具结果中的标记必须从工具获取。请尽量一次并行调用互不依赖的两个工具，回答简短。"
        graph = create_deep_agent(
            model=self.model,
            tools=[load_ledger, lookup_delivery],
            system_prompt=system,
            middleware=cast(
                list[AgentMiddleware[Any, Any, Any]],
                [
                    DisabledNativeSummaryMiddleware(),
                    FilesystemMiddleware(
                        tool_token_limit_before_evict=None,
                        human_message_token_limit_before_evict=None,
                    ),
                    mw,
                    Capture(self.inputs, config, self.model_requests),
                ],
            ),
            context_schema=RequestContext,
            checkpointer=saver,
        )
        return cast(DemoAgent, graph), mw

    async def run(
        self,
        tenant: Tenant,
        saver: BaseCheckpointSaver[Any],
        run_id: str,
        *,
        follow: bool = False,
        interrupt: bool = False,
        resume: bool = False,
    ) -> dict[str, Any]:
        identity = tenant.identity(run_id)
        token, phase_token = ACTIVE.set(identity), PHASE.set("run")
        start = time.monotonic()
        inputs_before, requests_before = len(self.inputs), len(self.requests)
        graph, mw = self.graph(tenant, saver)
        config: RunnableConfig = {
            "configurable": {"thread_id": tenant.thread},
            "recursion_limit": 40,
        }
        try:
            prompt = (
                "只回复本会话此前工具返回的ledger_marker和delivery_marker。不要调用工具，不要猜测。"
                if follow
                else "请在同一次响应中并行调用load_ledger和lookup_delivery各一次。然后只回复两个工具返回的ledger_marker与delivery_marker，不重复调用。"
            )
            value: InputAgentState | None = (
                None
                if resume
                else {"messages": [HumanMessage(id=f"user-{run_id}", content=prompt)]}
            )
            await asyncio.wait_for(
                graph.ainvoke(
                    value,
                    config=config,
                    context=RequestContext(identity),
                    interrupt_after=["tools"] if interrupt else None,
                ),
                self.args.timeout,
            )
            snapshot = await graph.aget_state(config)
            state = snapshot.values
            observed = [
                x
                for x in self.inputs[inputs_before:]
                if tuple(x["owner"]) == identity.session_key and x["run_id"] == run_id
            ]
            raw = state["messages"]
            cached = state["ov_context"]["messages"]
            if not interrupt:
                validate_tool_pairs(cached)
                ids = [m.id for m in cached]
                check(all(ids) and len(ids) == len(set(ids)), "Cache message ID duplication")
                text = str(raw[-1].content)
                check(
                    tenant.marker + "-LEDGER" in text and tenant.marker + "-DELIVERY" in text,
                    "Model failed to recover own tool facts",
                )
                check(
                    state["ov_context"]["write_index"] == len(raw), "Checkpoint cursor incomplete"
                )
                check(
                    tuple(state["ov_context"]["owner"]) == identity.session_key,
                    "Checkpoint owner mixed",
                )
                for item in observed:
                    validate_model_input(item, tenant)
                if not follow and not resume:
                    batch = [m for m in raw if isinstance(m, AIMessage) and len(m.tool_calls) >= 2]
                    check(bool(batch), "Model did not request a parallel tool batch")
                    wanted = [c["id"] for c in batch[0].tool_calls]
                    got = [
                        m.tool_call_id
                        for m in raw
                        if isinstance(m, ToolMessage) and m.tool_call_id in wanted
                    ]
                    check(got == wanted, "Parallel tool results reordered/lost")
                    events = [
                        e
                        for e in self.tools
                        if tuple(e["owner"]) == identity.session_key and e["run_id"] == run_id
                    ]
                    starts = [e["at"] for e in events if e["event"] == "start"]
                    ends = [e["at"] for e in events if e["event"] == "end"]
                    check(
                        len(starts) == len(ends) == 2 and max(starts) < min(ends),
                        "Tools did not overlap or executed repeatedly",
                    )
                save(
                    self.output / f"private/{sha(tenant.thread)[:10]}-{run_id}-state.json",
                    {
                        "messages": [m.model_dump(mode="json") for m in raw],
                        "ov_context": {
                            k: [m.model_dump(mode="json") for m in v] if k == "messages" else v
                            for k, v in state["ov_context"].items()
                        },
                    },
                )
                phase_save = PHASE.set("compare")
                remote = await OVContextReader(self.sdk).exchange_context(
                    tenant.session, identity=identity.sdk_identity
                )
                PHASE.reset(phase_save)
                check(
                    signature(cached) == signature(remote.messages),
                    "Cache/OV message projection differs",
                )
            run_requests = [
                r
                for r in self.requests[requests_before:]
                if r["phase"] == "run"
                and r["owner"] == list(identity.session_key)
                and r["run_id"] == run_id
            ]
            for request in run_requests:
                check(
                    list(request["headers"].values()) == [tenant.account, tenant.user, tenant.peer],
                    "HTTP identity header mismatch",
                )
            check(dict(self.http.headers) == self.defaults, "Shared HTTP defaults mutated")
            return {
                "owner": list(identity.session_key),
                "run_id": run_id,
                "pid": os.getpid(),
                "seconds": round(time.monotonic() - start, 3),
                "model_inputs": len(observed),
                "context_reads": sum(r["path"].endswith("/context") for r in run_requests),
                "raw_messages": len(raw),
                "cached_messages": len(cached),
                "next": list(snapshot.next),
                "interrupted": interrupt,
                "parallel_tools": not follow and not resume,
                "message_ids": [m.id for m in cached],
                "raw_message_ids": [m.id for m in raw],
                "tool_result_ids": [m.tool_call_id for m in raw if isinstance(m, ToolMessage)],
                "executed_tools": [
                    e["call_id"]
                    for e in self.tools
                    if e["event"] == "end"
                    and tuple(e["owner"]) == identity.session_key
                    and e["run_id"] == run_id
                ],
            }
        finally:
            await mw.aclose()
            PHASE.reset(phase_token)
            ACTIVE.reset(token)

    async def matrix(self) -> dict[str, Any]:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        serial, concurrent, follow = [], [], []
        specs = [
            ("a", "alice", "mail"),
            ("a", "bob", "mail"),
            ("a", "alice", "doc"),
            ("b", "alice", "mail"),
        ]
        tenants = [
            Tenant(
                self.prefix + "-" + a,
                u,
                p,
                "shared-session" + ("-doc" if p == "doc" else ""),
                "MARK-" + self.prefix + f"-{i}",
            )
            for i, (a, u, p) in enumerate(specs)
        ]
        async with AsyncSqliteSaver.from_conn_string(
            str(self.output / "checkpoint.sqlite")
        ) as saver:
            # Warm a real model call before timing either workload.
            await self.model.ainvoke("只回复ready。")
            t = time.monotonic()
            for tenant in tenants:
                baseline = Tenant(
                    tenant.account,
                    tenant.user,
                    tenant.peer,
                    "serial-session-" + tenant.peer,
                    tenant.marker,
                )
                serial.append(await self.run(baseline, saver, "serial"))
            serial_total = time.monotonic() - t
            t = time.monotonic()
            sem = asyncio.Semaphore(self.args.parallelism)

            async def invoke(tenant: Tenant) -> dict[str, Any]:
                async with sem:
                    return await self.run(tenant, saver, "parallel")

            tasks = [asyncio.create_task(invoke(x)) for x in tenants]
            try:
                concurrent = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            parallel_total = time.monotonic() - t
            for tenant in tenants:
                follow.append(await self.run(tenant, saver, "follow", follow=True))
            check(
                all(x["context_reads"] == 0 for x in follow),
                "Follow-up reloaded working context despite stable version",
            )
            for tenant in tenants:
                matching = [
                    i for i in self.inputs if tuple(i["owner"]) == tenant.identity("x").session_key
                ]
                wire = json.dumps(matching, ensure_ascii=False)
                for other in tenants:
                    if other != tenant:
                        check(other.marker not in wire, "Foreign tenant marker in model inputs")
            durations = sorted(x["seconds"] for x in concurrent)
            p95 = durations[max(0, math.ceil(0.95 * len(durations)) - 1)]
            speedup = serial_total / parallel_total
            slowdown = p95 / statistics.median(x["seconds"] for x in serial)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            value = {
                "peak_process_rss_kib": usage.ru_maxrss,
                "cpu_user_seconds": usage.ru_utime,
                "cpu_system_seconds": usage.ru_stime,
                "serial": serial,
                "concurrent": concurrent,
                "follow": follow,
                "serial_total": serial_total,
                "parallel_total": parallel_total,
                "throughput_speedup": speedup,
                "p95_slowdown": slowdown,
                "performance_pass": speedup >= 1.5 and slowdown <= 2.5,
            }
            save(self.output / "matrix-summary.json", value)
            check(value["performance_pass"], "Concurrency performance outside declared envelope")
            return value

    async def compression(self) -> dict[str, Any]:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from benchmark.office.suites.common.memory_inventory import memory_leaves

        tenant = Tenant(
            self.prefix + "-history", "alice", "audit", "history", "BUDGET-" + self.prefix
        )
        identity = tenant.identity("history")
        token = ACTIVE.set(identity)
        try:
            await self.sdk.create_session(
                tenant.session,
                identity=identity.sdk_identity,
                memory_policy={"self": {"enabled": True}, "peer": {"enabled": False}},
            )
            baseline = initial_memory_baseline(identity.user_id)
            for leaf in await memory_leaves(self.sdk, identity.sdk_identity):
                baseline[leaf["uri"]] = sha(
                    await self.sdk.read(leaf["uri"], identity=identity.sdk_identity)
                )
            save(self.output / "memory-baseline.json", baseline)
            history = []
            for i in range(12):
                history.extend(
                    [
                        HumanMessage(
                            id=f"hu-{i}",
                            content=f"项目校验标记是{tenant.marker}，订单O-42预算137元。保留该事实。\n"
                            + "已核对订单，等待发货。" * 300,
                        ),
                        AIMessage(
                            id=f"ha-{i}",
                            content="确认预算137元，订单O-42。" + "重复日志：核对完成。" * 250,
                        ),
                    ]
                )
            writer = OVMessageWriter(self.sdk)
            result = await writer.write(tenant.session, history, 0, identity=identity.sdk_identity)
            cached = OVContext(
                tenant.session,
                messages=result.model_messages,
                write_index=result.write_index,
                owner=identity.session_key,
                context_status="ready",
            )
            before_chars = sum(len(str(m.content)) for m in cached.messages)
            receipt = await self.sdk.commit_session(
                tenant.session, keep_recent_count=2, identity=identity.sdk_identity
            )
            check(
                receipt.get("archived") is True and bool(receipt.get("task_id")),
                "Commit did not start archive",
            )
            task: dict[str, Any] = {}
            for _ in range(180):
                task = (
                    await self.sdk.get_task(receipt["task_id"], identity=identity.sdk_identity)
                    or {}
                )
                if task["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(1)
            check(task.get("status") == "completed", "Archive task did not complete")
            snapshot = await OVContextReader(
                self.sdk, MemoryReaderConfig(context_token_budget=200_000)
            ).exchange_context(tenant.session, identity=identity.sdk_identity)
            check(
                bool(snapshot.archive_id) and len(snapshot.messages) < len(history),
                "History not replaced by archived view",
            )
            after_chars = sum(len(str(m.content)) for m in snapshot.messages)
            reduction = 1 - after_chars / before_chars
            check(reduction >= 0.7, "History compression below 70%")
            check(
                tenant.marker in str([m.content for m in snapshot.messages]),
                "Key history fact lost",
            )
            async with AsyncSqliteSaver.from_conn_string(
                str(self.output / "history.sqlite")
            ) as saver:
                graph, mw = self.graph(tenant, saver, commit=True, small=True)
                config: RunnableConfig = {"configurable": {"thread_id": tenant.thread}}
                completed_node = next(
                    name
                    for name in graph.nodes
                    if name.startswith("OpenVikingContextMiddleware")
                    and name.endswith(".after_agent")
                )
                await graph.aupdate_state(
                    config,
                    {"messages": history, "ov_context": cached.to_state()},
                    as_node=completed_node,
                )
                seeded = (await graph.aget_state(config)).values
                check(
                    seeded.get("ov_context", {}).get("write_index") == len(history),
                    "Private OVContext not seeded into checkpoint",
                )
                check(
                    len(seeded.get("messages", [])) == len(history),
                    "Raw history not seeded into checkpoint",
                )
                phase = PHASE.set("run")
                try:
                    await graph.ainvoke(
                        {
                            "messages": [
                                HumanMessage(
                                    id="new-history-query",
                                    content="此前记录的项目校验标记、订单号和预算是多少？只返回三项，不调用工具。",
                                )
                            ]
                        },
                        config=config,
                        context=RequestContext(identity),
                    )
                finally:
                    PHASE.reset(phase)
                state = (await graph.aget_state(config)).values
                check(
                    tenant.marker in str(state["messages"][-1].content),
                    "Model could not recover archived fact",
                )
                assembled = [x for x in self.inputs if tuple(x["owner"]) == identity.session_key]
                check(
                    bool(assembled)
                    and all(x["input_tokens"] <= x["input_limit"] for x in assembled),
                    "Archived input over budget",
                )
                check(
                    sum(m.id.startswith("ov_summary:") for m in state["ov_context"]["messages"])
                    == 1,
                    "Summary duplication",
                )
                save(self.output / "private/archive-task.json", task)
                save(
                    self.output / "private/archive-snapshot.json",
                    [m.model_dump(mode="json") for m in snapshot.messages],
                )
                native = next(
                    r for r in self.model_requests if r.runtime.context.identity == identity
                )
                before_input = ApproximateTokenCounter()(
                    native.override(
                        messages=cast(list[AnyMessage], [*history, native.messages[-1]])
                    )
                )
                after_input = ApproximateTokenCounter()(native)
                save(
                    self.output / "compression-summary.json",
                    {
                        "before_chars": before_chars,
                        "after_chars": after_chars,
                        "body_reduction": reduction,
                        "before_input_tokens": before_input,
                        "after_input_tokens": after_input,
                        "input_reduction": 1 - after_input / before_input,
                    },
                )
                check(
                    1 - after_input / before_input >= 0.5,
                    "Complete model input reduction below 50%",
                )
                for item in assembled:
                    validate_model_input(item, tenant)
                return {
                    "before_input_tokens": before_input,
                    "after_input_tokens": after_input,
                    "input_reduction": 1 - after_input / before_input,
                    "before_chars": before_chars,
                    "after_chars": after_chars,
                    "reduction": reduction,
                    "archive_id": snapshot.archive_id,
                    "memory_result": task.get("result"),
                    "model_inputs": len(assembled),
                }
        finally:
            ACTIVE.reset(token)

    async def collision(self) -> dict[str, Any]:
        tenant = Tenant(
            self.prefix + "-collision", "alice", "mail", "same-id", "COLLISION-" + self.prefix
        )
        token = ACTIVE.set(tenant.identity("collision"))
        try:
            await self.sdk.create_session(
                tenant.session, identity=tenant.identity("x").sdk_identity
            )
            await self.sdk.add_message(
                tenant.session,
                role="assistant",
                content=tenant.marker + "-MAIL",
                peer_id="mail",
                identity=tenant.identity("x").sdk_identity,
            )
            from fastapi import HTTPException

            from examples.deepagents_ov_context.chat.server import SessionInfo, SessionStore

            store_path = self.output / "private/session-directory.sqlite3"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store = SessionStore(store_path)
            store.save(
                SessionInfo(
                    session_id=tenant.session,
                    account_id=tenant.account,
                    user_id=tenant.user,
                    agent_id=tenant.peer,
                    created_at=time.time(),
                )
            )
            store.close()
            # A fresh application instance must enforce the persisted owner,
            # independently of Middleware/checkpoint state and OV capabilities.
            store = SessionStore(store_path)
            other = RunIdentity(tenant.account, tenant.user, "doc", tenant.session, "collision")
            before = len(self.requests)
            rejected = False
            try:
                store.require_identity(other)
                await self.sdk.get_session_context(tenant.session, identity=other.sdk_identity)
            except HTTPException as exc:
                rejected = exc.status_code == 403
            finally:
                store.close()
            check(len(self.requests) == before, "Rejected identity reached the OV service")
            data = {
                "account": tenant.account,
                "user": tenant.user,
                "session": tenant.session,
                "writer_peer": "mail",
                "reader_peer": "doc",
                "different_peer_rejected": rejected,
                "rejection_layer": "application",
                "ov_requests_after_rejection": len(self.requests) - before,
            }
            save(self.output / "collision-summary.json", data)
            check(
                rejected,
                "Same account/user/session must reject a different peer",
            )
            return data
        finally:
            ACTIVE.reset(token)

    async def logs(self) -> dict[str, Any]:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        tenant = Tenant(self.prefix + "-logs", "alice", "audit", "logs", "LOGS-" + self.prefix)
        identity = tenant.identity("logging")
        token = ACTIVE.set(identity)
        before = len(self.metrics)
        try:
            async with AsyncSqliteSaver.from_conn_string(str(self.output / "logs.sqlite")) as saver:
                graph, mw = self.graph(tenant, saver, commit=True)
                engine = mw._ov_session_engine
                engine.context_reader = OVContextReader(
                    self.sdk,
                    MemoryReaderConfig(enabled=True, experience_enabled=False, recall_timeout=30),
                )
                config: RunnableConfig = {
                    "configurable": {"thread_id": tenant.thread},
                    "recursion_limit": 30,
                }
                await graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(id="log-user", content="只回复ready，不调用工具。")
                        ]
                    },
                    config=config,
                    context=RequestContext(identity),
                )
                state = (await graph.aget_state(config)).values
                context = OVContext.from_state(state["ov_context"])
                engine.config = replace(engine.config, commit_wait_timeout=300, poll_interval=1)
                success = await engine.commit(
                    context,
                    identity=identity.sdk_identity,
                    trigger="capacity",
                    pending_tokens=1,
                    keep_recent_count=0,
                )
                check(success, "Log probe did not submit a real archive")
                await engine.wait_summary(context, identity=identity.sdk_identity)
                failed = False
                bad = OVContext(
                    "missing-" + self.prefix,
                    owner=(tenant.account, tenant.user, tenant.peer, "missing-" + self.prefix),
                    messages=[HumanMessage(id="missing-user", content="Failure probe")],
                )
                try:
                    await engine.commit(
                        bad,
                        identity=identity.sdk_identity,
                        trigger="capacity",
                        pending_tokens=1,
                        keep_recent_count=0,
                    )
                except Exception as exc:
                    failed = True
                    save(
                        self.output / "private/commit-failure-probe.json",
                        {
                            "error_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                            "scope": "acceptance collector, not native logging",
                        },
                    )
                events = self.metrics[before:]
                found = {
                    "commit_success": any(
                        e["event"] == "context.commit"
                        and e.get("status") in ("ok", "archived", "accepted")
                        for e in events
                    ),
                    "commit_failure": any(
                        e["event"] == "context.commit" and e.get("status") in ("failed", "error")
                        for e in events
                    ),
                    "recall_count": any(
                        e["event"] == "memory.recall" and "count" in e for e in events
                    ),
                    "compression_ratio": any(
                        "compression_ratio" in e or "reduction_ratio" in e for e in events
                    ),
                    "error_stack": any("traceback" in e or "error_stack" in e for e in events),
                }
                data = {
                    "actual_commit_submitted": success,
                    "actual_commit_failed": failed,
                    "archive_id": context.archive_id,
                    "native_metrics_fields": found,
                    "events": sorted({e["event"] for e in events}),
                    "note": "Server log must be checked separately; collector diagnostics are not native operation logs.",
                }
                save(self.output / "logs-summary.json", data)
                await mw.aclose()
                check(
                    all(found.values()),
                    "Native metric gaps: " + ", ".join(k for k, v in found.items() if not v),
                )
                return data
        finally:
            ACTIVE.reset(token)

    async def index(self) -> dict[str, Any]:
        from benchmark.office.suites.common.memory_inventory import memory_leaves

        tenant = Tenant(
            self.prefix + "-history", "alice", "audit", "history", "BUDGET-" + self.prefix
        )
        identity = tenant.identity("index")
        token = ACTIVE.set(identity)
        try:
            leaves = await memory_leaves(self.sdk, identity.sdk_identity)
            baseline_path = self.output / "memory-baseline.json"
            check(
                baseline_path.exists(), "Memory baseline missing; run compression in a fresh output"
            )
            baseline = initial_memory_baseline(identity.user_id)
            baseline.update(json.loads(baseline_path.read_text()))
            rows = []
            for leaf in leaves:
                body = await self.sdk.read(leaf["uri"], identity=identity.sdk_identity)
                # Initialization templates are not commit outputs. Changed
                # templates are checked too, regardless of the chosen type.
                if baseline.get(leaf["uri"]) == sha(body):
                    continue
                found = await self.sdk.find(
                    "订单核对完成 订单O-42 预算137元",
                    target_uri=leaf["uri"],
                    limit=20,
                    score_threshold=0,
                    identity=identity.sdk_identity,
                )
                rows.append(
                    {
                        "uri": leaf["uri"],
                        "file_chars": len(body),
                        "utf8_bytes": len(body.encode()),
                        "indexed_leaf": any(
                            m.uri == leaf["uri"] and m.level == 2 for m in found.memories
                        ),
                        "retrieved": [{"uri": m.uri, "level": m.level} for m in found.memories],
                    }
                )
            data = {"files": rows, "missing_index": sum(not r["indexed_leaf"] for r in rows)}
            save(self.output / "index-summary.json", data)
            check(bool(rows), "No memories found after real extraction")
            check(
                all(r["indexed_leaf"] for r in rows),
                "Persisted memory files are missing from the searchable leaf index",
            )
            return data
        finally:
            ACTIVE.reset(token)

    async def worker(self) -> dict[str, Any]:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        tenant = Tenant(
            self.prefix + "-restart", "alice", "audit", "restart", "RESTART-" + self.prefix
        )
        async with AsyncSqliteSaver.from_conn_string(str(self.output / "restart.sqlite")) as saver:
            return await self.run(
                tenant,
                saver,
                "interrupted",
                interrupt=self.args.worker_phase == "pause",
                resume=self.args.worker_phase == "resume",
            )

    async def restart(self) -> dict[str, Any]:
        observations = []
        for phase in ("pause", "resume"):
            command = [
                sys.executable,
                "-m",
                "tests.st.ov_context.component_worker",
                "--output",
                str(self.output),
                "--model-config",
                str(self.args.model_config),
                "--api-key-file",
                str(self.args.api_key_file),
                "--timeout",
                str(self.args.timeout),
                "--ov-url",
                self.args.ov_url,
                "--prefix",
                self.prefix,
                "--worker-phase",
                phase,
                "--source",
                self.args.source,
            ]
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), self.args.timeout)
            except BaseException:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), 5)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                raise
            save(
                self.output / f"private/worker-{phase}-process.json",
                {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "returncode": proc.returncode,
                },
            )
            check(proc.returncode == 0, f"Restart {phase} worker failed")
            value = json.loads((self.output / f"worker-{phase}.json").read_text())
            observations.append(value)
        check(observations[0]["pid"] != observations[1]["pid"], "Worker process not recreated")
        check(
            observations[0]["next"] and not observations[1]["next"],
            "Checkpoint did not resume to completion",
        )
        check(
            set(observations[0]["raw_message_ids"]).issubset(
                set(observations[1]["raw_message_ids"])
            ),
            "Checkpoint raw messages lost on resume",
        )
        check(
            observations[0]["tool_result_ids"] == observations[1]["tool_result_ids"],
            "Completed tool results changed or repeated on resume",
        )
        check(
            len(observations[0]["executed_tools"]) == 2 and not observations[1]["executed_tools"],
            "Resume re-executed completed tools",
        )
        return {"pause": observations[0], "resume": observations[1]}


async def execute(args: argparse.Namespace) -> int:
    args.output = args.output.resolve()
    if (
        args.parallelism < 1
        or args.parallelism > 32
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        raise ValueError("Use parallelism 1..32 and a finite positive timeout")
    existing = args.output / (
        f"worker-{args.worker_phase}.json"
        if args.command == "worker"
        else f"conditions-{args.command}.json"
    )
    if existing.exists():
        raise FileExistsError("This output already contains evidence; choose a fresh directory")
    async with Suite(args) as suite:
        if args.command == "worker":
            result = await suite.worker()
            save(args.output / f"worker-{args.worker_phase}.json", result)
            return 0
        actions = {
            "writer": suite.writer,
            "matrix": suite.matrix,
            "compression": suite.compression,
            "restart": suite.restart,
            "collision": suite.collision,
            "logs": suite.logs,
            "index": suite.index,
        }
        await suite.case(args.command, actions[args.command])
        if args.command == "compression" and suite.results[-1]["status"] == "passed":
            await suite.case("index", suite.index)
        return int(any(x["status"] != "passed" for x in suite.results))
