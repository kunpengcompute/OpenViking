# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Server-owned OV destinations selected by page and experiment session IDs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openviking_cli.client.identity import valid_identity_field


class EnvironmentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    label: str = Field(min_length=1, max_length=100)
    ov_url: str
    account_id: str

    @field_validator("account_id")
    @classmethod
    def valid_account(cls, value: str) -> str:
        valid_identity_field(value)
        return value

    @field_validator("ov_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("OV URL must be an HTTP service origin")
        return value.rstrip("/")


class ChatEnvironment(EnvironmentBinding):
    # Deployment paths and credentials stay outside session snapshots and HTTP responses.
    api_key_file: Path | None = Field(default=None, exclude=True, repr=False)
    env_file: Path | None = Field(default=None, exclude=True, repr=False)

    def binding(self) -> EnvironmentBinding:
        return EnvironmentBinding.model_validate(self.model_dump())

    def api_key(self) -> str:
        return self.api_key_file.read_text().strip() if self.api_key_file else ""


def load_environments(path: Path | None) -> tuple[ChatEnvironment, ...]:
    if path is None:
        return ()
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("Chat environments must be a JSON array")
    result = tuple(ChatEnvironment.model_validate(item) for item in records)
    ids = [item.id for item in result]
    if "daily" in ids or len(set(ids)) != len(ids):
        raise ValueError("Environment IDs must be unique; daily is reserved")
    for item in result:
        for location in (item.api_key_file, item.env_file):
            if location is not None and not location.is_file():
                raise ValueError(f"Environment file not found: {location}")
        item.api_key()
    return result
