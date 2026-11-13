# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Load Chat model settings and generate a private local OV service configuration."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from openviking.integrations.langchain.ov_context.commit import CommitConfig
from openviking.integrations.langchain.ov_context.config import OVContextConfig
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig
from openviking_cli.client.identity import valid_identity_field
from openviking_cli.retrieve.context import ContextSearchOptions

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / "config/chat.example.env"

DEFAULT_READER_CONFIG = MemoryReaderConfig(recall_timeout=30, search_mode="context")
DEFAULT_FIND_READER_CONFIG = replace(
    DEFAULT_READER_CONFIG,
    recall_limit=5,
    ranking="score",
    prefer_leaf=False,
    search_mode="find",
)


class ChatAgentEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_AGENT__", extra="ignore")
    enable_todos: bool


def load_todo_enabled(env_file: Path | None = None) -> bool:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Agent environment file not found: {env_file}")
    defaults = DEFAULT_ENV_FILE
    files = (defaults, env_file) if env_file is not None else defaults
    return ChatAgentEnvironment(_env_file=files).enable_todos


class ChatModelEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_MODEL__", extra="ignore")
    call_timeout_seconds: float = Field(gt=0, le=3600, allow_inf_nan=False)


def load_model_call_timeout(env_file: Path | None = None) -> float:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Model environment file not found: {env_file}")
    defaults = DEFAULT_ENV_FILE
    files = (defaults, env_file) if env_file is not None else defaults
    return ChatModelEnvironment(_env_file=files).call_timeout_seconds


class ChatOVPolicy(BaseSettings):
    """Application I/O protection and identity-scoped ordinary recall defaults."""

    model_config = SettingsConfigDict(env_prefix="OV_CHAT__", extra="ignore", frozen=True)
    io_timeout: float = Field(default=5, gt=0, le=5, allow_inf_nan=False)
    fail_open: bool = True
    recall_agent_memories: bool = True

    def __init__(self, *, env_file: Path | None = None, **values: Any) -> None:
        super().__init__(_env_file=env_file, **values)

    @property
    def wait_budget(self) -> float:
        return self.io_timeout * 0.98

    def recall_targets(
        self,
        targets: tuple[str, ...] | None,
        agent_id: str,
        *,
        context_quotas: bool = False,
    ) -> tuple[str, ...] | None:
        valid_identity_field(agent_id)
        if targets is None:
            if context_quotas:
                if self.recall_agent_memories:
                    return None  # Let the server select each quota category's roots.
                # Disable peer memories while retaining resource and skill scopes.
                return (
                    "viking://user/memories",
                    "viking://resources",
                    "viking://user/resources",
                    f"viking://user/peers/{agent_id}/resources",
                    "viking://user/skills",
                )
            targets = ("viking://user/memories",)
        if self.recall_agent_memories and "viking://user/memories" in targets:
            return tuple(dict.fromkeys((*targets, f"viking://user/peers/{agent_id}/memories")))
        return targets


def load_ov_policy(env_file: Path | None = None) -> ChatOVPolicy:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Chat environment file not found: {env_file}")
    return ChatOVPolicy(env_file=env_file)


class ReaderEnvironment(BaseSettings):
    """Load typed reader defaults; process variables override an optional .env file."""

    model_config = SettingsConfigDict(
        env_prefix="OV_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    reader: MemoryReaderConfig = Field(default=DEFAULT_READER_CONFIG)

    def __init__(self, *, env_file: Path | None = None) -> None:
        super().__init__(_env_file=env_file)

    @field_validator("reader", mode="before")
    @classmethod
    def parse_json_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        defaults = (
            DEFAULT_FIND_READER_CONFIG
            if value.get("search_mode") == "find"
            else DEFAULT_READER_CONFIG
        )
        result = asdict(defaults)
        result.update(value)
        for name in ("target_uris", "experience_target_uris", "recall_quotas"):
            if isinstance(result.get(name), str):
                result[name] = json.loads(result[name])
        detail = result.get("recall_detail")
        if isinstance(detail, str) and detail.lstrip().startswith("{"):
            result["recall_detail"] = json.loads(detail)
        # Validate quotas before settings coerces booleans or numeric strings to ints.
        ContextSearchOptions(
            mode="context", quotas=result.get("recall_quotas"), detail=result.get("recall_detail")
        )
        return result


def load_reader_config(env_file: Path | None = None) -> MemoryReaderConfig:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Reader environment file not found: {env_file}")
    return ReaderEnvironment(env_file=env_file).reader


class CommitEnvironment(BaseSettings):
    """All four commit settings must come from the environment or an explicit file."""

    model_config = SettingsConfigDict(env_prefix="OV_", env_nested_delimiter="__", extra="ignore")
    commit: CommitConfig

    def __init__(self, *, env_file: Path | None = None) -> None:
        super().__init__(_env_file=env_file)


def load_commit_defaults(env_file: Path | None = None) -> CommitConfig:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Commit environment file not found: {env_file}")
    return CommitEnvironment(env_file=env_file).commit


class ContextEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OV_", env_nested_delimiter="__", extra="ignore")
    context: OVContextConfig = Field(default_factory=OVContextConfig)

    def __init__(self, *, env_file: Path) -> None:
        super().__init__(_env_file=env_file)


def load_context_config(env_file: Path | None = None) -> OVContextConfig:
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"Context environment file not found: {env_file}")
    return ContextEnvironment(env_file=env_file or DEFAULT_ENV_FILE).context


class ServiceCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OV_SERVICE_", extra="ignore")
    api_key: str = Field(default="", repr=False)

    def __init__(self, *, env_file: Path | None = None) -> None:
        super().__init__(_env_file=env_file)


def load_service_api_key(env_file: Path | None = None, *, key_file: Path | None = None) -> str:
    key = ServiceCredentials(env_file=env_file).api_key
    if not key and key_file is not None and key_file.is_file():
        key = key_file.read_text().strip()
    return key


def commit_config_values(config: CommitConfig) -> dict[str, int | float]:
    """Expose actual settings as JSON numbers, including the validated Decimal ratio."""
    return {
        "token_budget": config.token_budget,
        "commit_token_threshold_ratio": float(config.commit_token_threshold_ratio),
        "completed_run_threshold": config.completed_run_threshold,
        "keep_recent_count": config.keep_recent_count,
    }


@dataclass(frozen=True)
class ModelConfig:
    api_key: str = field(repr=False)
    api_base: str
    chat_model: str
    embedding_model: str
    embedding_dimension: int
    chat_max_tokens: int = 256
    embedding_provider: str = "openai"
    embedding_input: str = "text"

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        raw = json.loads(path.read_text())
        for name in ("api_key", "api_base", "chat_model", "embedding_model"):
            if not isinstance(raw.get(name), str) or not raw[name].strip():
                raise ValueError(f"Fill {name} in {path}")
        return cls(**raw)


def write_server_config(config: ModelConfig, output: Path, port: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    key_file = output / "service.key"
    if not key_file.exists():
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as key_handle:
            key_handle.write(secrets.token_urlsafe(32))
    service_key = key_file.read_text().strip()
    if not service_key:
        raise ValueError("OV service credential is empty")
    settings = {
        "server": {
            "host": "127.0.0.1",
            "port": port,
            "auth_mode": "trusted",
            "root_api_key": service_key,
        },
        "storage": {
            "workspace": str((output / "data").resolve()),
            "agfs": {"backend": "local"},
            "vectordb": {"backend": "local"},
            # Active leases renew every half-period. After SIGKILL, leave time for
            # archive recovery inside the application's 300-second commit wait.
            "transaction": {"lock_expire": 30.0},
        },
        "embedding": {
            "dense": {
                "provider": config.embedding_provider,
                "model": config.embedding_model,
                "api_base": config.api_base,
                "api_key": config.api_key,
                "dimension": config.embedding_dimension,
                "input": config.embedding_input,
            }
        },
        "vlm": {
            "provider": "openai",
            "model": config.chat_model,
            "api_base": config.api_base,
            "api_key": config.api_key,
            "max_tokens": 512,
            "max_retries": 1,
            "timeout": 60,
            "extra_request_body": (
                {"thinking": {"type": "disabled"}} if "volces.com" in config.api_base else {}
            ),
        },
        "oauth": {"enabled": False},
    }
    path = output / "ov.conf"
    # Create with owner-only permissions before writing credentials.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(settings, handle, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1933)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    config = ModelConfig.load(args.model_config)
    print(write_server_config(config, args.output, args.port))


if __name__ == "__main__":
    main()
