# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local streaming Chat API over real DeepAgents, OV Reader/Writer and models.

Run one worker on loopback. Session records, observations and checkpoints live
in SQLite. Each Run builds an Agent using the saved session configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from pprint import pformat
from typing import Any, cast

import anyio
import httpx
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from examples.deepagents_ov_context.chat.agent import DemoAgent, RequestContext
from examples.deepagents_ov_context.chat.artifacts import (
    ARTIFACT_TYPES,
    FileArtifact,
    artifact_directory,
    list_artifacts,
    valid_input_name,
)
from examples.deepagents_ov_context.chat.browser_tool import AgentBrowser
from examples.deepagents_ov_context.chat.config import (
    ChatOVPolicy,
    ModelConfig,
    commit_config_values,
    load_commit_defaults,
    load_context_config,
    load_model_call_timeout,
    load_ov_policy,
    load_reader_config,
    load_service_api_key,
    load_todo_enabled,
)
from examples.deepagents_ov_context.chat.environments import ChatEnvironment, load_environments
from examples.deepagents_ov_context.chat.event_log import RunEventLog
from examples.deepagents_ov_context.chat.history import (
    EventKind,
    RunEvent,
    RunHistory,
    RunStatistics,
    RunSummary,
    RunTrace,
)
from examples.deepagents_ov_context.chat.model import (
    ArkChatOpenAI,
    ModelCallTimeoutError,
    TimedChatOpenAI,
    windowed_model,
)
from examples.deepagents_ov_context.chat.native_summary import (
    DisabledNativeSummaryMiddleware,
    native_summary,
)
from examples.deepagents_ov_context.chat.office_skills import (
    BUNDLED_SKILLS,
    OfficeSkillRun,
    RunSkillsMiddleware,
    validate_skill,
)
from examples.deepagents_ov_context.chat.ov_resilience import ResilientOVMiddleware
from examples.deepagents_ov_context.chat.planning import DisabledTodoListMiddleware
from examples.deepagents_ov_context.chat.profiles import (
    MAX_OUTPUT_TOKENS,
    ExecutionProfile,
    ObservedContextReader,
)
from examples.deepagents_ov_context.chat.prompts import (
    BROWSER_CONTRACT_VERSION,
    browser_contract,
    build_chat_system_prompt,
    current_browser_prompt,
)
from examples.deepagents_ov_context.chat.session import (
    ChatRequest,
    ContextView,
    MemoryExtractionStatus,
    NewSession,
    SessionConfigSnapshot,
    SessionDetail,
    SessionInfo,
    SessionStore,
)
from examples.deepagents_ov_context.chat.stream import (
    ActiveRun,
    InputTrace,
    StreamEvent,
    _active_run,
    error_stack,
)
from examples.deepagents_ov_context.chat.tool_call_recovery import (
    InvalidToolCallError,
    ToolCallRecoveryMiddleware,
)
from openviking.integrations.langchain.middleware import (
    OpenVikingCommitPolicy,
    OpenVikingContextMiddleware,
)
from openviking.integrations.langchain.ov_context import (
    BudgetSpec,
    ContextBudgetExceeded,
    Metrics,
    OVContextConfig,
)
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.errors import ContextReadError
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking.integrations.langchain.ov_context.retry import retry_call
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter, WriterConfig
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity, valid_identity_field
from openviking_cli.exceptions import AlreadyExistsError, NotFoundError, OpenVikingError

# Keep the existing import entry points while implementations live in focused modules.
__all__ = [
    "ActiveRun",
    "ChatRequest",
    "ChatRuntime",
    "ChatSettings",
    "ContextView",
    "EventKind",
    "InputTrace",
    "MemoryExtractionStatus",
    "ModelOutputLimitError",
    "NewSession",
    "RunEvent",
    "RunHistory",
    "RunStatistics",
    "RunSummary",
    "RunTrace",
    "SessionDetail",
    "SessionInfo",
    "SessionStore",
    "StreamEvent",
    "create_app",
    "debug_echo",
    "main",
]

logger = logging.getLogger(__name__)


async def ensure_extraction_disabled_session(
    sdk: AsyncHTTPClient, session_id: str, identity: OVIdentity
) -> None:
    """Reuse a prepared session only when both extraction routes are disabled."""
    try:
        await sdk.create_session(
            session_id,
            memory_policy={"self": {"enabled": False}, "peer": {"enabled": False}},
            identity=identity,
        )
    except AlreadyExistsError:
        existing = await sdk.get_session(session_id, identity=identity, auto_create=False)
        policy = existing.get("memory_policy", {})
        if any(policy.get(scope, {}).get("enabled") is not False for scope in ("self", "peer")):
            raise ValueError("Existing session must have extraction disabled") from None


class ModelOutputLimitError(RuntimeError):
    """The provider stopped generating before a complete answer or tool call."""


class ArtifactPublishError(RuntimeError):
    """Generated files failed publication after the graph saved its state."""


@dataclass(frozen=True)
class ChatSettings:
    model_config: Path
    data_dir: Path
    ov_url: str = "http://127.0.0.1:1933"
    account_id: str = "chat-account"
    max_tokens: int = 2048
    agent_browser: Path | None = None
    browser_executable: Path | None = None
    browser_proxy: str | None = None
    recursion_limit: int = 24
    thinking: bool = False
    enable_todos: bool = field(default_factory=load_todo_enabled)
    pptx_skill: Path | None = None
    skill_node_modules: Path | None = None
    office_skills: bool = False
    reader_config: MemoryReaderConfig = field(default_factory=load_reader_config)
    commit_config: CommitConfig = field(default_factory=load_commit_defaults)
    ov_policy: ChatOVPolicy = field(default_factory=load_ov_policy)
    context_config: OVContextConfig = field(default_factory=load_context_config)
    ov_api_key: str = field(default_factory=load_service_api_key, repr=False)
    environments_file: Path | None = None
    environments: tuple[ChatEnvironment, ...] = ()
    model_call_timeout_seconds: float = field(default_factory=load_model_call_timeout)

    def __post_init__(self) -> None:
        key_file = self.data_dir.parent / "ov" / "service.key"
        if not self.ov_api_key and key_file.is_file():
            object.__setattr__(self, "ov_api_key", key_file.read_text().strip())
        valid_identity_field(self.account_id)
        loaded = (*self.environments, *load_environments(self.environments_file))
        if any(item.id == "daily" for item in loaded) or len({e.id for e in loaded}) != len(loaded):
            raise ValueError("Environment IDs must be unique; daily is reserved")
        object.__setattr__(self, "environments", loaded)
        object.__setattr__(self, "environments_file", None)
        if not 1 <= self.max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens must be between 1 and {MAX_OUTPUT_TOKENS}")
        if not 1 <= self.recursion_limit <= 1000:
            raise ValueError("recursion_limit must be between 1 and 1000")
        if (
            not math.isfinite(self.model_call_timeout_seconds)
            or not 0 < self.model_call_timeout_seconds <= 3600
        ):
            raise ValueError("model_call_timeout_seconds must be finite and within (0, 3600]")
        if not 0 <= self.context_config.retry_attempts <= 10:
            raise ValueError("retry_attempts must be between 0 and 10")
        for path in (self.agent_browser, self.browser_executable):
            if path is not None and not path.is_file():
                raise ValueError(f"Browser executable not found: {path}")
        if self.pptx_skill is not None:
            validate_skill(self.pptx_skill)
            if shutil.which("node") is None:
                raise ValueError("pptx Skill requires Node; run install-pptx-skill")
        if self.skill_node_modules is not None and not self.skill_node_modules.is_dir():
            raise ValueError("Skill Node modules directory does not exist")
        for name, path in self.skill_sources.items():
            validate_skill(path, name)
        if self.office_skills:
            from examples.deepagents_ov_context.scripts.check import check_office_skills

            check_office_skills()

    @property
    def skill_sources(self) -> dict[str, Path]:
        sources = {"pptx": self.pptx_skill} if self.pptx_skill is not None else {}
        if self.office_skills:
            names = ["office-documents", "data-analysis"]
            if self.agent_browser is not None:
                names.append("agent-browser")
            sources.update({name: BUNDLED_SKILLS / name for name in names})
        return sources


@tool
def debug_echo(text: str) -> str:
    """回显传入文本，用于验证工具调用、工具结果写入 OV 和后续模型组装。"""
    return text


class ChatRuntime:
    def __init__(
        self, settings: ChatSettings, model: ModelConfig, *, checkpointer: BaseCheckpointSaver[Any]
    ) -> None:
        self.settings, self.model_config = settings, model
        self.store = SessionStore(settings.data_dir / "sessions.sqlite3")
        self.event_log = RunEventLog(settings.data_dir / "logs" / "runs.jsonl")
        try:
            self.store.history.bind_log(self.event_log, self.store.observation_identity)
        except BaseException:
            with suppress(Exception):
                self.store.close()
            self.event_log.close()
            raise
        interrupted = self.store.history.interrupt_running()
        for session in self.store.list():
            if session.last_run is not None and session.last_run.status == "running":
                session.last_run.status = "interrupted"
                interrupted.add(session.session_id)
            if session.session_id in interrupted:
                session.needs_restore = False
                self.store.save(session)
        self.saver = checkpointer
        self.active: dict[str, ActiveRun] = {}
        self.restoring: set[str] = set()
        self.benchmark_jobs: dict[tuple[str, str], asyncio.Task] = {}
        self.ov_http = httpx.AsyncClient(
            timeout=self.settings.ov_policy.wait_budget, trust_env=False
        )

        async def observe_ov(request: httpx.Request) -> None:
            run = _active_run.get()
            if run is not None:
                run.emit_metric("ov_request", {"method": request.method, "path": request.url.path})

        self.ov_http.event_hooks["request"].append(observe_ov)
        # The model adapter bounds response waits per Session. Keep transport
        # connect/write/pool timeouts while delegating read waits to that timer.
        model_transport_timeout = httpx.Timeout(90, read=None)
        self.model_http = httpx.AsyncClient(timeout=model_transport_timeout, trust_env=False)
        model_class = ArkChatOpenAI if "volces.com" in model.api_base else TimedChatOpenAI
        self.model = model_class(
            model=model.chat_model,
            api_key=SecretStr(model.api_key),
            base_url=model.api_base,
            http_async_client=self.model_http,
            max_completion_tokens=settings.max_tokens,
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=model_transport_timeout,
            max_retries=0,
            async_call_timeout_seconds=settings.model_call_timeout_seconds,
            model_kwargs={"parallel_tool_calls": False} if settings.agent_browser else {},
            extra_body=(
                {
                    "thinking": {"type": "enabled" if settings.thinking else "disabled"},
                }
                if "volces.com" in model.api_base
                else None
            ),
        )

    def environment(self, environment_id: str = "daily") -> ChatEnvironment:
        if environment_id == "daily":
            return ChatEnvironment(
                id="daily",
                label="日常调试",
                ov_url=self.settings.ov_url,
                account_id=self.settings.account_id,
            )
        for environment in self.settings.environments:
            if environment.id == environment_id:
                return environment
        raise HTTPException(409, "运行环境未配置，请检查服务配置")

    def session_environment(self, session: SessionInfo | None = None) -> ChatEnvironment:
        environment = self.environment(session.environment_id if session else "daily")
        if session is not None:
            binding = session.config_snapshot.environment if session.config_snapshot else None
            if session.account_id != environment.account_id or (
                binding is not None
                and (binding.id, binding.ov_url, binding.account_id)
                != (environment.id, environment.ov_url, environment.account_id)
            ):
                raise HTTPException(409, "会话绑定的运行环境已变化，请恢复原环境配置")
            if binding is None and session.environment_id != "daily":
                raise HTTPException(409, "实验会话缺少运行环境快照")
        return environment

    def environment_defaults(self, environment: ChatEnvironment) -> ChatSettings:
        if environment.env_file is None:
            return self.settings
        return replace(
            self.settings,
            reader_config=load_reader_config(environment.env_file),
            context_config=load_context_config(environment.env_file),
            commit_config=load_commit_defaults(environment.env_file),
            ov_policy=load_ov_policy(environment.env_file),
            enable_todos=load_todo_enabled(environment.env_file),
            model_call_timeout_seconds=load_model_call_timeout(environment.env_file),
        )

    def environment_key(self, environment: ChatEnvironment) -> str | None:
        return (
            self.settings.ov_api_key if environment.id == "daily" else environment.api_key()
        ) or None

    def sdk(self, session: SessionInfo | None = None) -> AsyncHTTPClient:
        environment = self.session_environment(session)
        return AsyncHTTPClient(
            url=environment.ov_url,
            http_client=self.ov_http,
            auth_mode="trusted",
            api_key=self.environment_key(environment),
            profile_enabled=False,
            timeout=self.settings.ov_policy.wait_budget,
        )

    async def ov_read(self, operation: Any) -> Any:
        """Idempotent OV read under the shared retry budget."""
        async with asyncio.timeout(self.settings.ov_policy.wait_budget):
            return await retry_call(operation, retries=self.settings.context_config.retry_attempts)

    async def context(self, session: SessionInfo, *, graph: DemoAgent | None = None) -> ContextView:
        self.store.require_identity(session.identity("context"))
        sdk: AsyncHTTPClient | None = None
        middleware: OpenVikingContextMiddleware | None = None
        values: dict[str, Any] = {}
        try:
            if graph is None:
                sdk = self.sdk(session)
                graph, middleware = self.session_graph(session, sdk, ActiveRun())
            # Read through the graph to restore DeltaChannel history. The caller
            # retains ownership of an existing Run's graph and its resources.
            values = (await graph.aget_state(session.graph_config)).values
            failure = values.get("ov_failure") or {}
            if not session.execution.native_context or failure:
                messages = values.get("messages", [])
                return ContextView(
                    source="checkpoint" if messages else "empty",
                    messages=[m.model_dump(mode="json") for m in messages],
                    message_reprs=[pformat(m, width=100, sort_dicts=False) for m in messages],
                    raw_message_count=len(messages),
                    cached_message_count=len(messages),
                    ov_degraded=bool(failure),
                    ov_error_type=failure.get("error_type"),
                    ov_sync_pending=bool(failure),
                )
            if values.get("ov_context"):
                context = values["ov_context"]
                messages = context["messages"]
                return ContextView(
                    source="checkpoint",
                    messages=[m.model_dump(mode="json") for m in messages],
                    message_reprs=[
                        pformat(message, width=100, sort_dicts=False) for message in messages
                    ],
                    raw_message_count=len(values.get("messages", [])),
                    cached_message_count=len(messages),
                    write_index=context["write_index"],
                    archive_id=context.get("archive_id"),
                    run_id=context.get("run_id"),
                    completed_runs=context.get("completed_runs", 0),
                    owner=list(context["owner"]),
                    context_status=context.get("context_status"),
                )
            if sdk is None:
                sdk = self.sdk(session)
            async with asyncio.timeout(self.settings.ov_policy.wait_budget):
                await sdk.initialize()
                reader = OVContextReader(sdk)
                await self.ov_read(
                    lambda: sdk.get_session(
                        session.session_id, identity=session.identity("inspect").sdk_identity
                    )
                )
                snapshot = await self.ov_read(
                    lambda: reader.exchange_context(
                        session.session_id, identity=session.identity("inspect").sdk_identity
                    )
                )
            return ContextView(
                source="ov",
                messages=[m.model_dump(mode="json") for m in snapshot.messages],
                message_reprs=[
                    pformat(message, width=100, sort_dicts=False) for message in snapshot.messages
                ],
                cached_message_count=len(snapshot.messages),
                archive_id=snapshot.archive_id,
                owner=list(session.identity("inspect").session_key),
            )
        except NotFoundError:
            return ContextView(source="empty", messages=[])
        except Exception as exc:
            if not self.settings.ov_policy.fail_open:
                raise
            messages = values.get("messages", [])
            return ContextView(
                source="checkpoint" if messages else "empty",
                messages=[m.model_dump(mode="json") for m in messages],
                raw_message_count=len(messages),
                cached_message_count=len(messages),
                ov_degraded=True,
                ov_error_type=type(exc).__name__,
                ov_sync_pending=True,
            )
        finally:
            if middleware is not None:
                await middleware.aclose()
            if sdk is not None:
                await sdk.close()

    async def memory_status(self, session: SessionInfo) -> MemoryExtractionStatus:
        self.store.require_identity(session.identity("memory_status"))
        if self.settings.ov_policy.fail_open and (await self.state(session)).get("ov_failure"):
            return MemoryExtractionStatus(
                status="unknown", task_id=session.memory_task_id, stage="ov_degraded"
            )
        if session.memory_commit_unknown:
            return MemoryExtractionStatus(status="unknown", task_id=session.memory_task_id)
        if session.memory_task_id is None:
            return MemoryExtractionStatus(status="idle")
        sdk = self.sdk(session)
        try:
            await sdk.initialize()
            task = await self.ov_read(
                lambda: sdk.get_task(
                    session.memory_task_id,
                    identity=session.identity("memory-status").sdk_identity,
                )
            )
            if task is None:
                return MemoryExtractionStatus(status="unknown", task_id=session.memory_task_id)
            if (
                task.get("resource_id") != session.session_id
                or task.get("task_type") != "session_commit"
            ):
                raise HTTPException(502, "记忆提取任务与当前会话不匹配")
            result = task.get("result") or {}
            return MemoryExtractionStatus(
                status=task["status"],
                task_id=session.memory_task_id,
                stage=task.get("stage"),
                memories_extracted=result.get("memories_extracted") or {},
                archive_uri=result.get("archive_uri"),
            )
        except HTTPException:
            raise
        except Exception:
            if not self.settings.ov_policy.fail_open:
                raise
            return MemoryExtractionStatus(
                status="unknown", task_id=session.memory_task_id, stage="ov_unavailable"
            )
        finally:
            await sdk.close()

    async def extract_memory(self, session: SessionInfo) -> MemoryExtractionStatus:
        self.store.require_identity(session.identity("extract_memory"))
        if session.needs_restore:
            raise HTTPException(409, "上次执行未完成，请先恢复会话")
        progress = await self.memory_status(session)
        if progress.status in {"pending", "running", "unknown"}:
            raise HTTPException(409, "已有提取任务或提交结果未知，请先检查任务状态")
        sdk = self.sdk(session)
        try:
            await sdk.initialize()
            identity = session.identity("memory-extract").sdk_identity
            try:
                meta = await self.ov_read(
                    lambda: sdk.get_session(session.session_id, identity=identity)
                )
            except NotFoundError:
                raise HTTPException(409, "请先发送消息，空会话没有可提取内容") from None
            if meta.get("message_count") == 0:
                return MemoryExtractionStatus(status="idle")
            # Mark before I/O: a lost receipt cannot safely be retried automatically.
            session.memory_commit_unknown = True
            self.store.save(session)
            try:
                receipt = await sdk.commit_session(
                    session.session_id, keep_recent_count=0, identity=identity
                )
                task_id = receipt.get("task_id")
                if not task_id and receipt.get("archived"):
                    raise HTTPException(502, "已归档但缺少提取任务标识，请检查 OV 状态")
            except BaseException:
                session.needs_restore = True
                self.store.save(session)
                raise
            session.memory_task_id = task_id
            session.memory_commit_unknown = False
            self.store.save(session)
            return MemoryExtractionStatus(
                status="pending" if task_id else "idle",
                task_id=task_id,
                archive_uri=receipt.get("archive_uri"),
            )
        finally:
            await sdk.close()

    def snapshot(self, session: SessionInfo) -> SessionConfigSnapshot:
        environment = self.environment(session.environment_id)
        defaults = self.environment_defaults(environment)
        profile = session.execution
        window = profile.context_window or defaults.context_config.context_window or 256_000
        reserve = profile.max_tokens or defaults.max_tokens
        context = replace(
            defaults.context_config,
            context_window=window,
            output_reserve=reserve,
            base_context_ratio=profile.base_context_ratio
            if profile.base_context_ratio is not None
            else defaults.context_config.base_context_ratio,
            capacity_ratio=profile.capacity_ratio
            if profile.capacity_ratio is not None
            else defaults.context_config.capacity_ratio,
        )
        if profile.native_context:
            BudgetSpec(
                window,
                reserve,
                context.safety_margin,
                context.max_input_tokens,
                context.base_context_ratio,
                context.capacity_ratio,
            )
        resolved = profile.model_copy(
            update={
                "context_window": window,
                "max_tokens": reserve,
                "model_call_timeout_seconds": profile.model_call_timeout_seconds
                or defaults.model_call_timeout_seconds,
                "recursion_limit": profile.recursion_limit or defaults.recursion_limit,
                "thinking": profile.thinking if profile.thinking is not None else defaults.thinking,
                "enable_todos": profile.enable_todos
                if profile.enable_todos is not None
                else defaults.enable_todos,
                "base_context_ratio": context.base_context_ratio,
                "capacity_ratio": context.capacity_ratio,
            }
        )
        tool_names = {"debug_echo"}
        if defaults.agent_browser:
            tool_names.add("browser")
        if defaults.office_skills and resolved.enable_todos:
            tool_names.add("write_todos")
        reader_config = profile.memory_config(defaults.reader_config)
        if profile.target_uris is None:
            reader_config = replace(
                reader_config,
                target_uris=defaults.ov_policy.recall_targets(
                    reader_config.target_uris,
                    session.agent_id,
                    context_quotas=reader_config.search_mode == "context"
                    and reader_config.recall_quotas is not None,
                ),
            )
        return SessionConfigSnapshot(
            environment=environment.binding(),
            session_key=session.identity("snapshot").session_key,
            execution={
                "profile": resolved.model_dump(mode="json"),
                "model": self.model_config.chat_model,
                "model_profile": windowed_model(self.model, window).profile,
                "system_prompt": build_chat_system_prompt(
                    tool_names=tool_names,
                    custom_prompt=profile.system_prompt,
                    skill_names=defaults.skill_sources.keys(),
                ),
                "browser": bool(defaults.agent_browser),
                "browser_contract_version": BROWSER_CONTRACT_VERSION,
                "skills": list(defaults.skill_sources),
                "office_skills": defaults.office_skills,
                "ov_policy": defaults.ov_policy.model_dump(mode="json"),
                "tokens_per_image": 1024,
            },
            context_config=context,
            reader_config=reader_config,
            writer_config=WriterConfig(),
            commit_config=profile.commit_settings(defaults.commit_config),
        )

    def graph(
        self,
        sdk: AsyncHTTPClient,
        run: ActiveRun,
        *,
        browser: AgentBrowser | None = None,
        skill_run: OfficeSkillRun | None = None,
        session: SessionInfo | None = None,
    ) -> tuple[DemoAgent, OpenVikingContextMiddleware | None]:
        snapshot = session.config_snapshot if session else None
        if session is not None and snapshot is None:
            raise HTTPException(409, "会话缺少有效配置快照，请新建会话")
        if snapshot is not None:
            assert session is not None
            if snapshot.session_key != session.identity("config").session_key:
                raise HTTPException(403, "配置快照身份与会话不匹配")
        profile = (
            ExecutionProfile.model_validate(snapshot.execution["profile"])
            if snapshot
            else ExecutionProfile()
        )
        # Old frozen snapshots predate this switch and retain planning enabled.
        enable_todos = (
            profile.enable_todos
            if profile.enable_todos is not None
            else True
            if snapshot
            else self.settings.enable_todos
        )
        context_window = profile.context_window or 256_000
        max_tokens = profile.max_tokens or self.settings.max_tokens

        def metric(event: str, **fields: Any) -> None:
            run.emit_metric(event, fields)

        policy = (
            ChatOVPolicy.model_validate(snapshot.execution["ov_policy"])
            if snapshot
            else self.settings.ov_policy
        )
        reader_config = (
            snapshot.reader_config
            if snapshot
            else profile.memory_config(self.settings.reader_config)
        )
        reader = ObservedContextReader(
            sdk,
            reader_config,
            scope_policy=policy if profile.target_uris is None else None,
        )
        reader.observer = metric
        optional_middleware: dict[str, Any] = (
            {
                "io_timeout": policy.io_timeout,
                "recall_retry_attempts": profile.memory_recall_retry_attempts,
            }
            if policy.fail_open
            else {}
        )
        middleware: OpenVikingContextMiddleware | None = (
            (ResilientOVMiddleware if policy.fail_open else OpenVikingContextMiddleware)(
                client=sdk,
                message_writer=OVMessageWriter(sdk, snapshot.writer_config if snapshot else None),
                context_reader=reader,
                memory_policy=None
                if profile.mode == "current" and profile.memory_extraction
                else {"self": {"enabled": False}, "peer": {"enabled": False}},
                commit_config=snapshot.commit_config
                if snapshot
                else profile.commit_settings(self.settings.commit_config),
                commit_policy=OpenVikingCommitPolicy(mode="never")
                if profile.mode != "current"
                else None,
                context_config=snapshot.context_config
                if snapshot
                else OVContextConfig(
                    context_window=context_window,
                    output_reserve=max_tokens,
                    safety_margin=512,
                    retry_attempts=self.settings.context_config.retry_attempts,
                ),
                metrics=Metrics(metric),
                **optional_middleware,
            )
            if profile.native_context
            else None
        )
        model = self.model
        if isinstance(model, ChatOpenAI) and snapshot:
            model = model.model_copy(
                update={
                    "model_name": snapshot.execution["model"],
                    "profile": snapshot.execution["model_profile"],
                }
            )
        if isinstance(model, ChatOpenAI) and (
            profile.mode != "current"
            or profile.max_tokens is not None
            or profile.thinking is not None
        ):
            extra = dict(model.extra_body or {})
            if "volces.com" in self.model_config.api_base:
                extra["thinking"] = {
                    "type": "enabled"
                    if (
                        profile.thinking if profile.thinking is not None else self.settings.thinking
                    )
                    else "disabled"
                }
            model = model.model_copy(update={"max_tokens": max_tokens, "extra_body": extra})
        office_skills = (
            snapshot.execution["office_skills"] if snapshot else self.settings.office_skills
        )
        model = windowed_model(model, context_window)
        if isinstance(model, TimedChatOpenAI):
            model = model.model_copy(
                update={
                    "async_call_timeout_seconds": profile.model_call_timeout_seconds
                    or self.settings.model_call_timeout_seconds
                }
            )
        tools: list[BaseTool] = [debug_echo]
        if skill_run is not None:
            tools.append(skill_run.file_tool())
        if browser is not None:
            contract = (
                browser_contract(snapshot.execution) if snapshot else BROWSER_CONTRACT_VERSION
            )
            browser.legacy_file_aliases = contract == 1
            tools.extend([browser.tool(), browser.file_tool()])
            run.emit_metric(
                "browser.compatibility",
                {
                    "stored_contract": contract,
                    "runtime_contract": BROWSER_CONTRACT_VERSION,
                    "legacy_file_aliases": browser.legacy_file_aliases,
                },
            )
        graph_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            *(
                [DisabledNativeSummaryMiddleware()]
                if profile.native_context
                else [native_summary(model, skill_run.backend if skill_run else None)]
            ),
            *(
                [RunSkillsMiddleware(backend=skill_run.backend, sources=["/skills/"])]
                if skill_run
                else []
            ),
            FilesystemMiddleware(
                backend=skill_run.backend if skill_run else None,
                tool_token_limit_before_evict=5000
                if profile.mode == "baseline" and profile.native_tool_compression
                else None,
                human_message_token_limit_before_evict=None,
            ),
            *(
                [DisabledTodoListMiddleware()]
                if not enable_todos
                else [TodoListMiddleware(system_prompt="")]
                if office_skills
                else []
            ),
            *([middleware] if middleware else []),
            ToolCallRecoveryMiddleware(
                observer=metric,
            ),
        ]
        graph = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=(
                current_browser_prompt(
                    snapshot.execution["system_prompt"], legacy=browser.legacy_file_aliases
                )
                if snapshot and browser is not None
                else snapshot.execution["system_prompt"]
                if snapshot
                else build_chat_system_prompt(
                    tool_names={item.name for item in tools}
                    | {
                        item.name
                        for component in graph_middleware
                        for item in getattr(component, "tools", ())
                    },
                    custom_prompt=profile.system_prompt,
                    skill_names=skill_run.skill_names if skill_run else (),
                )
            )
            + ("\n\n" + skill_run.instructions if skill_run else ""),
            skills=["/skills/"] if skill_run else None,
            backend=skill_run.backend if skill_run else None,
            middleware=graph_middleware,
            context_schema=RequestContext,
            checkpointer=self.saver,
        )
        return cast(DemoAgent, graph), middleware

    def session_graph(
        self,
        session: SessionInfo,
        sdk: AsyncHTTPClient,
        run: ActiveRun,
        *,
        browser: AgentBrowser | None = None,
        skill_run: OfficeSkillRun | None = None,
    ) -> tuple[DemoAgent, OpenVikingContextMiddleware | None]:
        return self.graph(sdk, run, session=session, browser=browser, skill_run=skill_run)

    async def state(self, session: SessionInfo) -> dict[str, Any]:
        sdk = self.sdk(session)
        middleware: OpenVikingContextMiddleware | None = None
        try:
            graph, middleware = self.session_graph(session, sdk, ActiveRun())
            return (await graph.aget_state(session.graph_config)).values
        finally:
            if middleware is not None:
                await middleware.aclose()
            await sdk.close()

    async def produce(self, session: SessionInfo, prompt: str, run: ActiveRun) -> None:
        self.store.require_identity(session.identity(run.run_id))
        run.bind_log(session, self.event_log)
        token = _active_run.set(run)
        sdk = self.sdk(session)
        middleware: OpenVikingContextMiddleware | None = None
        graph: DemoAgent | None = None
        identity = session.identity(run.run_id)
        snapshot = session.config_snapshot
        browser = (
            AgentBrowser(
                self.settings.agent_browser,
                identity.run_id,
                chrome=self.settings.browser_executable,
                proxy=self.settings.browser_proxy,
                download_dir=self.settings.data_dir
                / "downloads"
                / session.session_id
                / identity.run_id,
                screenshot_dir=(
                    self.settings.data_dir
                    / "artifacts"
                    / session.session_id
                    / identity.run_id
                    / "output"
                    if snapshot is not None and snapshot.execution["skills"]
                    else None
                ),
            )
            if self.settings.agent_browser is not None
            and snapshot is not None
            and snapshot.execution["browser"]
            else None
        )
        started = time.monotonic()
        statistics = RunStatistics(run_id=identity.run_id)
        benchmark_audit = None
        run.statistics = statistics
        session.last_run = statistics
        self.store.save(session)
        try:
            if snapshot is None:
                raise HTTPException(409, "会话缺少有效配置快照，请新建会话")
            from benchmark.office.suites.common.audit import create_audit

            benchmark_audit = create_audit(self.settings.data_dir, session, run)
            if benchmark_audit is not None:
                started = run.started
                benchmark_audit.capture(round(time.monotonic() - run.started, 3), "run.start")
            await run.emit(
                "start",
                session_id=session.session_id,
                run_id=identity.run_id,
                thread_id=session.thread_id,
                model=snapshot.execution["model"],
                identity=list(identity.session_key),
                statistics=statistics.model_dump(),
                execution=session.execution.model_dump(mode="json"),
                environment_id=session.environment_id,
                experiment_id=session.experiment_id,
                reader_config=asdict(snapshot.reader_config)
                if session.execution.native_context
                else None,
                commit_config=commit_config_values(snapshot.commit_config)
                if session.execution.mode == "current"
                else None,
            )
            skill_sources = {
                name: self.settings.skill_sources[name] for name in snapshot.execution["skills"]
            }
            skill_run = (
                OfficeSkillRun(
                    skill_sources,
                    self.settings.skill_node_modules,
                    self.settings.data_dir / "artifacts",
                    session.session_id,
                    identity.run_id,
                    shell_enabled=not session.execution.browser_only,
                    input_dir=self.settings.data_dir / "inputs" / session.session_id,
                )
                if skill_sources
                else None
            )
            graph, middleware = self.session_graph(
                session, sdk, run, browser=browser, skill_run=skill_run
            )
            if session.execution.uses_ov:
                await sdk.initialize()
                if session.execution.mode != "current" or not session.execution.memory_extraction:

                    async def prepare_session():
                        await ensure_extraction_disabled_session(
                            sdk, session.session_id, identity.sdk_identity
                        )

                    if isinstance(middleware, ResilientOVMiddleware):
                        values = (await graph.aget_state(session.graph_config)).values
                        await middleware.prepare_session(prepare_session, values)
                    else:
                        await prepare_session()
            input_trace = InputTrace(run, session, self.store)
            input_trace.benchmark_audit = benchmark_audit
            config: RunnableConfig = {
                **session.graph_config,
                "callbacks": [input_trace],
            }
            # LangGraph awaits a separate task to cancel and join its workers.
            # AnyIO level cancellation can cancel that cleanup task as well.
            async with asyncio.timeout(session.execution.timeout_seconds):
                async for item in graph.astream(
                    {"messages": [HumanMessage(id="u-" + uuid.uuid4().hex, content=prompt)]},
                    config=config,
                    context=RequestContext(identity),
                    stream_mode=["messages", "updates"],
                    subgraphs=False,
                ):
                    mode, payload = cast(tuple[str, Any], item)
                    if mode == "messages":
                        chunk, metadata = payload
                        if (
                            isinstance(chunk, AIMessage)
                            and metadata.get("langgraph_node") == "model"
                        ):
                            if chunk.text:
                                if statistics.first_token_seconds is None:
                                    statistics.first_token_seconds = round(
                                        time.monotonic() - started, 3
                                    )
                                    self.store.save(session)
                                    run.emit_metric(
                                        "run.first_text",
                                        {"first_token_seconds": statistics.first_token_seconds},
                                    )
                                await run.emit(
                                    "token",
                                    text=chunk.text,
                                    message_id=chunk.id,
                                    first_token_seconds=statistics.first_token_seconds,
                                )
                    elif mode == "updates":
                        for node, update in payload.items():
                            await run.emit("stage", name=node)
                            if isinstance(update, dict):
                                state_view = update.get("ov_context")
                                if isinstance(state_view, dict):
                                    run.emit_metric(
                                        "context.state",
                                        {
                                            **{
                                                key: state_view.get(key)
                                                for key in (
                                                    "write_index",
                                                    "context_status",
                                                    "archive_id",
                                                    "pending_commit",
                                                    "recall_done",
                                                    "recall_status",
                                                    "completed_runs",
                                                )
                                            },
                                            "cached_message_count": len(
                                                state_view.get("messages", [])
                                            ),
                                        },
                                    )
                                for message in update.get("messages", []):
                                    if isinstance(message, AIMessage) and (
                                        message.invalid_tool_calls
                                        or (
                                            message.response_metadata.get("finish_reason")
                                            == "tool_calls"
                                            and not message.tool_calls
                                        )
                                    ):
                                        raise InvalidToolCallError(
                                            "Model returned no executable tool call; task incomplete"
                                        )
                                    if (
                                        isinstance(message, AIMessage)
                                        and message.response_metadata.get("finish_reason")
                                        == "length"
                                    ):
                                        raise ModelOutputLimitError("Incomplete model output")
                                    if isinstance(message, (AIMessage, ToolMessage)):
                                        await run.emit(
                                            "message", message=message.model_dump(mode="json")
                                        )
                                    if (
                                        skill_run is not None
                                        and isinstance(message, ToolMessage)
                                        and message.name in {"write_file", "edit_file"}
                                        and message.status != "error"
                                    ):
                                        try:
                                            skill_run.publish()
                                            if skill_run.last_published:
                                                run.emit_metric(
                                                    "artifacts.published",
                                                    {
                                                        "artifacts": skill_run.last_published,
                                                        "elapsed_seconds": round(
                                                            time.monotonic() - run.started, 6
                                                        ),
                                                    },
                                                )
                                        except Exception as exc:
                                            run.emit_metric(
                                                "artifacts.publish_failed",
                                                {"error_type": type(exc).__name__},
                                            )
            view = await self.context(session, graph=graph)
            if (
                session.execution.native_context
                and not view.ov_degraded
                and (view.run_id is not None or view.write_index != view.raw_message_count)
            ):
                raise RuntimeError("Run 未完成 OV 写入或临时状态清理")
            if skill_run is not None:
                try:
                    skill_run.publish()
                except Exception as exc:
                    raise ArtifactPublishError("Generated files could not be published") from exc
            session.needs_restore = False
            statistics.status = "done"
            statistics.seconds = round(time.monotonic() - started, 3)
            self.store.save(session)
            await run.emit(
                "done",
                seconds=statistics.seconds,
                statistics=statistics.model_dump(),
                context=view.model_dump(),
                artifacts=[
                    item.model_dump()
                    for item in list_artifacts(
                        self.settings.data_dir / "artifacts", session.session_id
                    )
                    if item.run_id == identity.run_id
                ],
            )
        except (Exception, asyncio.CancelledError) as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            run.emit_metric(
                "run.error",
                {
                    "status": "cancelled" if cancelled else "error",
                    "error_type": type(exc).__name__,
                    "error_stack": error_stack(exc),
                },
            )
            artifact_error = isinstance(exc, ArtifactPublishError)
            # Preparation and pre-send write failures can certify that OV and
            # the saved cursor still agree. Artifact publication happens after the
            # graph and its persistence checks have completed, so both errors
            # allow the next turn to continue from the existing checkpoint.
            consistent = not cancelled and (
                artifact_error or bool(getattr(exc, "ov_state_consistent", False))
            )
            if not consistent:
                session.needs_restore = True
            statistics.status = "cancelled" if cancelled else "error"
            statistics.seconds = round(time.monotonic() - started, 3)
            self.store.save(session)
            if (
                not artifact_error
                and graph is not None
                and isinstance(middleware, OpenVikingContextMiddleware)
            ):
                try:
                    await middleware.acleanup_run(graph, session.graph_config)
                except Exception:
                    consistent = False
                    logger.warning("Run cleanup failed for %s", session.session_id)
                else:
                    if not cancelled and isinstance(exc, ContextBudgetExceeded):
                        # Admission failure did not execute the next model call.
                        # Cleanup checkpoints known commit progress and ends the
                        # old Run before the application admits a new Run.
                        consistent = True
            session.needs_restore = not consistent
            self.store.save(session)
            # No exception/config dump: model request errors may contain credentials.
            logger.warning("Chat Run %s: %s", identity.run_id, type(exc).__name__)
            await run.emit(
                "cancelled" if cancelled else "error",
                error_type=type(exc.__cause__).__name__
                if artifact_error and exc.__cause__ is not None
                else type(exc).__name__,
                message="已停止。继续前请从 OV 恢复或新建会话。"
                if cancelled
                else "等待模型响应超时：流式首包或后续片段长时间未到达，或非流式响应未完成。本轮任务未完成。"
                if isinstance(exc, ModelCallTimeoutError)
                else "模型输出达到上限，回答未完成。请调整输出预算后从 OV 恢复或新建会话。"
                if isinstance(exc, ModelOutputLimitError)
                else "模型工具参数无效，任务未完成。请从 OV 恢复或新建会话后重试。"
                if isinstance(exc, InvalidToolCallError)
                else "交付文件校验或发布失败，会话已保存。可继续对话重新生成文件。"
                if artifact_error
                else "上下文预算不足，本轮已结束。可在当前会话发起新请求，或新建会话调整预算。"
                if consistent and isinstance(exc, ContextBudgetExceeded)
                else "执行失败，已保留确认的写入进度。请检查服务后在当前会话发起新请求。"
                if consistent
                else "执行失败。请检查本地服务和模型配置，然后从 OV 恢复或新建会话。",
                run_id=identity.run_id,
                statistics=statistics.model_dump(),
                **(
                    {"stage": "artifact_publish", "can_continue": True}
                    if artifact_error
                    else {"stage": "budget", "can_continue": True}
                    if consistent and isinstance(exc, ContextBudgetExceeded)
                    else {"can_continue": True}
                    if consistent
                    else {}
                ),
            )
        finally:
            try:
                if browser is not None:
                    with suppress(Exception):
                        await browser.close()
                if middleware is not None:
                    await middleware.aclose()
            finally:
                await sdk.close()
                self.store.save(session)
                if benchmark_audit is not None:
                    try:
                        benchmark_audit.capture(
                            statistics.seconds, "run.end", status=statistics.status
                        )
                    except Exception as exc:
                        run.emit_metric(
                            "benchmark.capture_error", {"error_type": type(exc).__name__}
                        )
                _active_run.reset(token)
                if run.cancel_requested:
                    with suppress(asyncio.QueueFull):
                        run.queue.put_nowait(None)
                else:
                    await run.queue.put(None)

    async def close(self) -> None:
        tasks = []
        for task in self.benchmark_jobs.values():
            task.cancel()
            tasks.append(task)
        for run in self.active.values():
            run.cancel_requested = True
            if run.task is not None:
                tasks.append(run.task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.model_http.aclose()
        await self.ov_http.aclose()
        try:
            self.store.close()
        finally:
            self.event_log.close()


def create_app(settings: ChatSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = ModelConfig.load(settings.model_config)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(
            str(settings.data_dir / "checkpoints.sqlite3")
        ) as saver:
            await saver.setup()
            runtime = ChatRuntime(settings, model, checkpointer=saver)
            app.state.runtime = runtime
            try:
                yield
            finally:
                # Finish cancellation/cleanup writes before closing the saver.
                await runtime.close()

    app = FastAPI(title="OVContext Chat", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.exception_handler(OpenVikingError)
    @app.exception_handler(ContextReadError)
    async def ov_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {
                "detail": "无法读取 OV 会话，请检查本地服务和会话状态。",
                "error_type": type(exc).__name__,
            },
            status_code=503,
        )

    @app.middleware("http")
    async def local_only(request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Loopback demo has no login. Reject cross-origin browser mutations.
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin is not None and origin != expected:
            return JSONResponse({"detail": "仅允许同源访问"}, status_code=403)
        return await call_next(request)

    def runtime() -> ChatRuntime:
        return cast(ChatRuntime, app.state.runtime)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        rt = runtime()
        try:
            async with asyncio.timeout(settings.ov_policy.wait_budget):
                response = await rt.ov_http.get(
                    rt.settings.ov_url + "/health", timeout=settings.ov_policy.wait_budget
                )
                response.raise_for_status()
                ov = response.json()
        except Exception as exc:
            if not settings.ov_policy.fail_open:
                raise HTTPException(503, "OV 服务不可用") from None
            ov = {"healthy": False, "status": "unavailable", "error_type": type(exc).__name__}
        return {
            "ov": ov,
            "model": rt.model_config.chat_model,
            "checkpoint": "AsyncSqliteSaver",
            "browser": "agent-browser" if rt.settings.agent_browser else None,
            "thinking": rt.settings.thinking,
            "presentation": rt.settings.pptx_skill is not None,
            "skills": sorted(rt.settings.skill_sources),
        }

    @app.get("/api/environments")
    async def environments() -> list[dict[str, Any]]:
        rt = runtime()
        return [
            environment.binding().model_dump()
            for environment in (rt.environment(), *rt.settings.environments)
        ]

    @app.get("/api/execution-profiles")
    async def profiles(environment_id: str = "daily") -> dict[str, Any]:
        environment = runtime().environment(environment_id)
        defaults = runtime().environment_defaults(environment)
        return {
            "modes": ["current", "baseline", "context_only", "ordinary", "experience"],
            "automatic_benchmark": True,
            "environment_id": environment.id,
            "account_id": environment.account_id,
            "model": runtime().model_config.chat_model,
            "ov_url": environment.ov_url,
            "checkpoint": "AsyncSqliteSaver",
            "browser": bool(defaults.agent_browser),
            "reader_defaults": asdict(defaults.reader_config),
            "context_defaults": asdict(defaults.context_config),
            "agent_defaults": {"enable_todos": defaults.enable_todos},
            "commit_defaults": commit_config_values(defaults.commit_config),
            "ov_policy": defaults.ov_policy.model_dump(),
            "skills": {
                name: {
                    "skill_md_sha256": hashlib.sha256(
                        (source / "SKILL.md").read_bytes()
                    ).hexdigest(),
                    "tree_sha256": hashlib.sha256(
                        "\n".join(
                            f"{path.relative_to(source)}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
                            for path in sorted(source.rglob("*"))
                            if path.is_file()
                        ).encode()
                    ).hexdigest(),
                }
                for name, source in runtime().settings.skill_sources.items()
            },
        }

    @app.get("/api/sessions/{session_id}/artifacts", response_model=list[FileArtifact])
    async def artifacts(session_id: str) -> list[FileArtifact]:
        rt = runtime()
        rt.store.get(session_id)
        return list_artifacts(rt.settings.data_dir / "artifacts", session_id)

    @app.post("/api/sessions/{session_id}/inputs/{filename}")
    async def upload_input(session_id: str, filename: str, request: Request) -> dict[str, Any]:
        """Receive a bounded Office input; never accept a client-side host path."""
        from examples.deepagents_ov_context.chat.office_skills import validate_output

        rt = runtime()
        session = rt.store.get(session_id)
        if not session.config_snapshot or not session.config_snapshot.execution["skills"]:
            raise HTTPException(409, "本会话未启用文件 Skill")
        if not valid_input_name(filename):
            raise HTTPException(400, "输入文件名或类型不支持")
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(413, "输入文件不能超过10MiB")
        # Recheck after receiving the body, before any filesystem mutation.
        if session_id in rt.active or session_id in rt.restoring:
            raise HTTPException(409, "会话执行期间不能上传输入文件")
        directory = rt.settings.data_dir / "inputs" / session_id
        if directory.is_symlink() or directory.parent.is_symlink():
            raise HTTPException(400, "输入目录无效")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        try:
            with target.open("xb") as destination:
                destination.write(content)
        except FileExistsError:
            raise HTTPException(409, "输入文件已存在，请使用新的文件名或会话") from None
        try:
            validate_output(target)
        except Exception:
            target.unlink()
            raise HTTPException(400, "输入文件无法解析") from None
        return {
            "filename": filename,
            "path": f"/inputs/{filename}",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    @app.get("/api/sessions/{session_id}/artifacts/{run_id}/{filename}")
    async def artifact_file(session_id: str, run_id: str, filename: str) -> FileResponse:
        rt = runtime()
        rt.store.get(session_id)
        items = list_artifacts(rt.settings.data_dir / "artifacts", session_id)
        if not any(item.run_id == run_id and item.filename == filename for item in items):
            raise HTTPException(404, "文件不存在")
        path = artifact_directory(rt.settings.data_dir / "artifacts", session_id, run_id) / filename
        return FileResponse(
            path,
            filename=filename,
            media_type=ARTIFACT_TYPES[path.suffix][1],
        )

    @app.post("/api/sessions", response_model=SessionInfo, status_code=201)
    async def new_session(body: NewSession) -> SessionInfo:
        window = body.execution.context_window or 256_000
        reserve = body.execution.max_tokens or settings.max_tokens
        if window <= reserve + 512:
            raise HTTPException(422, "上下文窗口不足以保留模型输出和安全余量")
        environment = runtime().environment(body.environment_id)
        session = SessionInfo(
            session_id="chat-" + uuid.uuid4().hex,
            account_id=environment.account_id,
            environment_id=environment.id,
            experiment_id=body.experiment_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            created_at=time.time(),
            execution=body.execution,
        )
        try:
            session.config_snapshot = runtime().snapshot(session)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        runtime().store.save(session)
        return session

    @app.get("/api/sessions", response_model=list[SessionInfo])
    async def sessions() -> list[SessionInfo]:
        return runtime().store.list()

    @app.get("/api/sessions/{session_id}", response_model=SessionDetail)
    async def detail(session_id: str) -> SessionDetail:
        rt = runtime()
        session = rt.store.get(session_id)
        return SessionDetail(
            session=session,
            busy=session_id in rt.active or session_id in rt.restoring,
            context=await rt.context(session),
        )

    @app.get("/api/sessions/{session_id}/runs", response_model=list[RunSummary])
    async def runs(session_id: str) -> list[RunSummary]:
        rt = runtime()
        rt.store.get(session_id)
        return rt.store.history.list(session_id)

    @app.get("/api/sessions/{session_id}/runs/{run_id}", response_model=RunTrace)
    async def run_trace(session_id: str, run_id: str) -> RunTrace:
        rt = runtime()
        rt.store.get(session_id)
        return rt.store.history.get(session_id, run_id)

    @app.post("/api/sessions/{session_id}/runs/{run_id}/benchmark")
    async def benchmark_grade(session_id: str, run_id: str) -> dict[str, Any]:
        from benchmark.office.suites.common.automatic import evaluate_audit
        from benchmark.office.suites.common.document_judge import DocumentJudge

        rt = runtime()
        rt.store.get(session_id)
        trace = rt.store.history.get(session_id, run_id)
        if trace.statistics.status == "running" or session_id in rt.active:
            raise HTTPException(409, "Benchmark grading requires a terminal Run")
        folder = rt.settings.data_dir / "benchmark" / session_id / run_id
        if not (folder / "manifest.json").is_file():
            raise HTTPException(404, "This Run has no automatic benchmark snapshots")
        if (folder / "grade.json").is_file():
            return json.loads((folder / "grade.json").read_text())

        async def grade():
            manifest = json.loads((folder / "manifest.json").read_text())
            judge = (
                DocumentJudge(rt.settings.model_config)
                if manifest["case"] == "reqdoc_generation"
                else None
            )
            try:
                return await evaluate_audit(folder, trace.statistics.model_dump(), judge=judge)
            finally:
                if judge is not None:
                    await judge.close()

        key = (session_id, run_id)
        if key not in rt.benchmark_jobs:
            task = asyncio.create_task(grade())
            rt.benchmark_jobs[key] = task
            task.add_done_callback(lambda finished: rt.benchmark_jobs.pop(key, None))
        return await asyncio.shield(rt.benchmark_jobs[key])

    @app.get("/api/sessions/{session_id}/state")
    async def graph_state(session_id: str) -> dict[str, Any]:
        rt = runtime()
        return await rt.state(rt.store.get(session_id))

    def downloads(session_id: str, run_id: str) -> Path:
        rt = runtime()
        rt.store.get(session_id)
        rt.store.history.get(session_id, run_id)
        return rt.settings.data_dir / "downloads" / session_id / run_id

    @app.get("/api/sessions/{session_id}/runs/{run_id}/downloads")
    async def download_list(session_id: str, run_id: str) -> list[dict[str, Any]]:
        folder = downloads(session_id, run_id)
        return [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(folder.glob("*.pdf"))
            if path.is_file() and not path.is_symlink()
        ]

    @app.get("/api/sessions/{session_id}/runs/{run_id}/downloads/{filename}")
    async def download_file(session_id: str, run_id: str, filename: str) -> FileResponse:
        folder = downloads(session_id, run_id)
        path = folder / filename
        if (
            path.parent != folder
            or path.is_symlink()
            or not path.is_file()
            or path.suffix != ".pdf"
        ):
            raise HTTPException(404, "文件不存在")
        return FileResponse(path, filename=filename, media_type="application/pdf")

    @app.post("/api/sessions/{session_id}/restore", response_model=SessionDetail)
    async def restore(session_id: str) -> SessionDetail:
        rt = runtime()
        session = rt.store.get(session_id)
        if session.history_only:
            raise HTTPException(409, "该会话为实验历史归档，请新建实验会话")
        if session_id in rt.active or session_id in rt.restoring:
            raise HTTPException(409, "会话正在执行，请先停止")
        rt.restoring.add(session_id)
        if not session.execution.native_context:
            try:
                await rt.saver.adelete_thread(session.thread_id)
                session.needs_restore = False
                rt.store.save(session)
                return SessionDetail(session=session, busy=False, context=await rt.context(session))
            finally:
                rt.restoring.discard(session_id)
        # Check the remote snapshot BEFORE dropping the existing checkpoint.
        sdk = rt.sdk(session)
        try:
            await sdk.initialize()
            try:
                await rt.ov_read(
                    lambda: sdk.get_session(
                        session_id, identity=session.identity("restore").sdk_identity
                    )
                )
                await rt.ov_read(
                    lambda: OVContextReader(sdk).exchange_context(
                        session_id, identity=session.identity("restore").sdk_identity
                    )
                )
            except NotFoundError:
                pass
            await rt.saver.adelete_thread(session.thread_id)
            session.needs_restore = False
            rt.store.save(session)
            view = await rt.context(session)
            return SessionDetail(session=session, busy=False, context=view)
        finally:
            await sdk.close()
            rt.restoring.discard(session_id)

    @app.get("/api/sessions/{session_id}/memory", response_model=MemoryExtractionStatus)
    async def memory_status(session_id: str) -> MemoryExtractionStatus:
        rt = runtime()
        session = rt.store.get(session_id)
        if session.history_only or not session.execution.native_context:
            return MemoryExtractionStatus(status="idle")
        return await rt.memory_status(session)

    @app.post("/api/sessions/{session_id}/memory", response_model=MemoryExtractionStatus)
    async def extract_memory(session_id: str) -> MemoryExtractionStatus:
        rt = runtime()
        session = rt.store.get(session_id)
        if session.history_only:
            raise HTTPException(409, "该会话为实验历史归档，请新建实验会话")
        if not session.execution.native_context:
            raise HTTPException(409, "该配置不提供本版本的记忆提取")
        if session_id in rt.active or session_id in rt.restoring:
            raise HTTPException(409, "会话正在执行，请等待结束后提取记忆")
        rt.restoring.add(session_id)
        try:
            return await rt.extract_memory(session)
        finally:
            rt.restoring.discard(session_id)

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel(session_id: str) -> dict[str, bool]:
        rt = runtime()
        rt.store.get(session_id)
        run = rt.active.get(session_id)
        if run is None or run.task is None or run.task.done() or run.cancel_requested:
            return {"cancelled": False}
        run.cancel_requested = True
        run.task.cancel()
        return {"cancelled": True}

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        rt = runtime()
        session = rt.store.get(body.session_id)
        if session.history_only:
            raise HTTPException(409, "该会话为实验历史归档，请新建实验会话")
        rt.session_environment(session)
        if body.session_id in rt.active or body.session_id in rt.restoring:
            raise HTTPException(409, "同一会话正在执行；不同会话可并发")
        if session.needs_restore:
            raise HTTPException(409, "上次执行未完成，请先从 OV 恢复或新建会话")
        if session.title == "新会话":
            session.title = body.message[:36]
            rt.store.save(session)
        run = ActiveRun(history=rt.store.history, session_id=session.session_id)
        rt.store.history.begin(session.session_id, run.run_id, body.message)
        session.last_run = RunStatistics(run_id=run.run_id)
        rt.store.save(session)
        rt.active[body.session_id] = run

        async def events() -> AsyncIterator[str]:
            run.task = asyncio.create_task(rt.produce(session, body.message, run))
            ordinal = 0
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(run.queue.get(), timeout=10)
                    except TimeoutError:
                        if run.task.done():
                            run.task.result()
                            break
                        yield ": keepalive\n\n"
                        continue
                    if event is None:
                        break
                    ordinal += 1
                    yield event.encode(ordinal)
            finally:
                # Client disconnect cancels the model request and releases this session.
                with anyio.CancelScope(shield=True):
                    try:
                        if not run.task.done():
                            if not run.cancel_requested:
                                run.cancel_requested = True
                                run.task.cancel()
                            await run.task
                    finally:
                        rt.active.pop(body.session_id, None)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Reader/commit/context settings from .env; process variables take precedence",
    )
    parser.add_argument(
        "--environments-file", type=Path, help="Server-owned experiment OV destinations"
    )
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ov-url", default="http://127.0.0.1:1933")
    parser.add_argument("--account-id", default="chat-account")
    parser.add_argument("--port", type=int, default=1934)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--agent-browser", type=Path, help="Enable the agent-browser CLI tool")
    parser.add_argument("--browser-executable", type=Path, help="Optional Chromium executable")
    parser.add_argument(
        "--browser-proxy", help="Browser-only proxy; OV/model connections stay direct"
    )
    parser.add_argument("--recursion-limit", type=int, default=24)
    parser.add_argument(
        "--retry-attempts",
        type=int,
        help="Override OV_CONTEXT__RETRY_ATTEMPTS for transient OV failures; 0 disables retries",
    )
    parser.add_argument("--thinking", action="store_true", help="Enable Ark model thinking")
    parser.add_argument("--pptx-skill", type=Path, help="Installed Anthropic pptx Skill directory")
    parser.add_argument(
        "--office-skills", action="store_true", help="Enable bundled office and data Skills"
    )
    parser.add_argument("--skill-node-modules", type=Path, help="Node dependencies used by Skills")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"--max-tokens must be between 1 and {MAX_OUTPUT_TOKENS}")
    context_config = load_context_config(args.env_file)
    if args.retry_attempts is not None:
        context_config = replace(context_config, retry_attempts=args.retry_attempts)
    settings = ChatSettings(
        args.model_config,
        args.data_dir,
        args.ov_url,
        account_id=args.account_id,
        max_tokens=args.max_tokens,
        agent_browser=args.agent_browser,
        browser_executable=args.browser_executable,
        browser_proxy=args.browser_proxy,
        recursion_limit=args.recursion_limit,
        thinking=args.thinking,
        enable_todos=load_todo_enabled(args.env_file),
        model_call_timeout_seconds=load_model_call_timeout(args.env_file),
        pptx_skill=args.pptx_skill,
        office_skills=args.office_skills,
        skill_node_modules=args.skill_node_modules,
        reader_config=load_reader_config(args.env_file),
        commit_config=load_commit_defaults(args.env_file),
        ov_policy=load_ov_policy(args.env_file),
        context_config=context_config,
        ov_api_key=load_service_api_key(args.env_file),
        environments_file=args.environments_file,
    )
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
