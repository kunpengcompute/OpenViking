# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Chat session schemas and SQLite directory, separate from graph checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from examples.deepagents_ov_context.chat.environments import EnvironmentBinding
from examples.deepagents_ov_context.chat.history import RunHistory, RunStatistics
from examples.deepagents_ov_context.chat.profiles import ExecutionProfile
from openviking.integrations.langchain.ov_context import RunIdentity
from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking.integrations.langchain.ov_context.writer import WriterConfig
from openviking_cli.client.identity import valid_identity_field


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_id: str = Field(default="debug-user", min_length=1, max_length=100)
    agent_id: str = Field(default="chat-agent", min_length=1, max_length=100)
    environment_id: str = Field(default="daily", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    experiment_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,100}$")
    execution: ExecutionProfile = Field(default_factory=ExecutionProfile)

    @field_validator("user_id", "agent_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        valid_identity_field(value)
        return value


class SessionConfigSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    environment: EnvironmentBinding | None = None
    session_key: tuple[str, str, str, str]
    execution: dict[str, Any]
    context_config: OVContextConfig
    reader_config: MemoryReaderConfig
    writer_config: WriterConfig
    commit_config: CommitConfig


class SessionInfo(BaseModel):
    session_id: str
    account_id: str
    user_id: str
    agent_id: str
    title: str = "新会话"
    created_at: float
    history_only: bool = False
    needs_restore: bool = False
    last_run: RunStatistics | None = None
    memory_task_id: str | None = None
    memory_commit_unknown: bool = False
    environment_id: str = Field(default="daily", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    experiment_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,100}$")
    execution: ExecutionProfile = Field(default_factory=ExecutionProfile)
    config_snapshot: SessionConfigSnapshot | None = None

    def identity(self, run_id: str) -> RunIdentity:
        return RunIdentity(self.account_id, self.user_id, self.agent_id, self.session_id, run_id)

    @property
    def thread_id(self) -> str:
        # Partition even when another account/user/agent has the same session ID.
        encoded = json.dumps(self.identity("inspect").session_key).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def graph_config(self) -> RunnableConfig:
        limit = (
            self.config_snapshot.execution["profile"]["recursion_limit"]
            if self.config_snapshot
            else 24
        )
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": limit}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=32_000)


class ContextView(BaseModel):
    source: Literal["checkpoint", "ov", "empty"]
    messages: list[dict[str, Any]]
    message_reprs: list[str] = Field(default_factory=list)
    raw_message_count: int = 0
    cached_message_count: int = 0
    write_index: int | None = None
    archive_id: str | None = None
    run_id: str | None = None
    completed_runs: int = 0
    owner: list[str] = Field(default_factory=list)
    context_status: str | None = None
    ov_degraded: bool = False
    ov_error_type: str | None = None
    ov_sync_pending: bool = False


class SessionDetail(BaseModel):
    session: SessionInfo
    busy: bool
    context: ContextView


class MemoryExtractionStatus(BaseModel):
    status: Literal["idle", "pending", "running", "completed", "failed", "unknown"]
    task_id: str | None = None
    stage: str | None = None
    memories_extracted: dict[str, int] = Field(default_factory=dict)
    archive_uri: str | None = None


class SessionStore:
    """Small local session directory, independent of OV and graph checkpoints."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.history = RunHistory(self.db)
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT)")
        self.db.commit()

    def save(self, session: SessionInfo) -> None:
        if (
            session.config_snapshot is not None
            and session.config_snapshot.session_key != session.identity("config").session_key
        ):
            raise HTTPException(403, "配置快照身份与会话不匹配")
        # Serialize creation/update across connections. The existing session
        # directory is authoritative; no separate peer binding table is needed.
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT data FROM sessions WHERE id = ?", (session.session_id,)
            ).fetchone()
            if row is not None:
                saved = SessionInfo.model_validate_json(row[0])
                if saved.identity("save").session_key != session.identity("save").session_key:
                    raise HTTPException(403, "会话归属不匹配，请为不同 Agent 使用独立会话")
                if (saved.environment_id, saved.experiment_id, saved.history_only) != (
                    session.environment_id,
                    session.experiment_id,
                    session.history_only,
                ):
                    raise HTTPException(409, "会话运行环境和实验批次已固定，请新建会话")
                if saved.execution != session.execution:
                    raise HTTPException(409, "执行配置不能在会话中途修改，请新建会话")
                if saved.config_snapshot != session.config_snapshot:
                    raise HTTPException(409, "会话配置快照已固定，请使用新配置创建会话")
            data = session.model_dump_json()
            if row is None or row[0] != data:
                self.db.execute(
                    "INSERT OR REPLACE INTO sessions VALUES (?, ?)",
                    (session.session_id, data),
                )
            if session.last_run is not None:
                self.history.update(session.session_id, session.last_run)

    def require_identity(self, identity: RunIdentity) -> SessionInfo:
        """Validate application-owned identity before any graph or OV call."""
        session = self.get(identity.session_id)
        if session.identity("validate").session_key != identity.session_key:
            raise HTTPException(403, "会话归属不匹配，请为不同 Agent 使用独立会话")
        return session

    def get(self, session_id: str) -> SessionInfo:
        row = self.db.execute("SELECT data FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "会话不存在")
        return SessionInfo.model_validate_json(row[0])

    def list(self) -> list[SessionInfo]:
        rows = self.db.execute("SELECT data FROM sessions ORDER BY rowid DESC").fetchall()
        return [SessionInfo.model_validate_json(row[0]) for row in rows]

    def observation_identity(self, session_id: str) -> tuple[str, str, str] | None:
        row = self.db.execute(
            "SELECT json_extract(data, '$.account_id'), json_extract(data, '$.user_id'), "
            "json_extract(data, '$.agent_id') FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return tuple(row) if row is not None else None

    def close(self) -> None:
        try:
            self.history.flush()
        finally:
            self.db.close()
