# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Shared wire contracts for deterministic, server-assembled search context."""

import math
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Detail = Literal["abstract", "overview", "full"]
MEMORY_CATEGORIES = frozenset({"events", "entities", "preferences", "experiences"})
QUOTA_CATEGORIES = MEMORY_CATEGORIES | {"resources", "skills"}
REPORTED_CATEGORIES = QUOTA_CATEGORIES | {"memories"}


class RecallPolicy(BaseModel):
    """Invocation-local reader limits; no model, connection or session settings."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    # None selects category-owned roots with quotas, the legacy memory root otherwise.
    # An explicit empty tuple disables all quota scopes (ordinary scopes without quotas).
    target_uris: Optional[Tuple[str, ...]] = None
    candidate_limit: int = Field(default=20, ge=0, strict=True)
    recall_limit: int = Field(default=5, ge=0, strict=True)
    score_threshold: float = Field(default=0.1, ge=0, le=1)
    prefer_abstract: bool = True
    ranking: Literal["openclaw", "score"] = "openclaw"
    prefer_leaf: bool = True
    max_injected_chars: int = Field(default=4000, ge=0, strict=True)
    experience_enabled: bool = False
    experience_target_uris: Tuple[str, ...] = ("viking://user/memories/experiences",)
    experience_candidate_limit: int = Field(default=12, ge=0, strict=True)
    experience_limit: int = Field(default=3, ge=0, strict=True)
    experience_score_threshold: float = Field(default=0.35, ge=0, le=1)
    experience_max_chars: int = Field(default=2000, ge=0, strict=True)

    @field_validator("target_uris", "experience_target_uris")
    @classmethod
    def validate_targets(cls, value: Optional[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
        if value is not None and any(not uri.startswith("viking://") for uri in value):
            raise ValueError("Recall targets must be viking:// URIs")
        return value


class ContextSearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["list", "context"] = "list"
    quotas: Optional[Dict[str, int]] = None
    detail: Optional[Union[Detail, Dict[str, Detail]]] = None
    max_tokens: int = Field(default=1600, gt=0, le=128000, strict=True)
    render: bool = True
    exclude_uris: List[str] = Field(default_factory=list, max_length=200)
    query_expansion: Literal["off"] = "off"
    recall_policy: Optional[RecallPolicy] = None

    @field_validator("quotas", mode="before")
    @classmethod
    def validate_quotas(cls, value: Any) -> Any:
        if value is not None:
            if not isinstance(value, dict) or set(value) - QUOTA_CATEGORIES:
                raise ValueError("Unknown context quota categories")
            if (
                any(type(n) is not int or n < 0 for n in value.values())
                or sum(value.values()) > 200
            ):
                raise ValueError("Quotas must be non-negative integers totaling at most 200")
        return value

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) - REPORTED_CATEGORIES:
            raise ValueError("Unknown context detail categories")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "ContextSearchOptions":
        if self.mode == "list":
            defaults = ContextSearchOptions.model_fields
            for name in ContextSearchOptions.model_fields:
                if name == "mode":
                    continue
                default = defaults[name].get_default(call_default_factory=True)
                if getattr(self, name) != default:
                    raise ValueError("Context options require mode='context'")
        return self

    def context_kwargs(self) -> Dict[str, Any]:
        """Only the context path adds options to legacy calls and wire payloads."""
        return (
            self.model_dump(mode="json", include=set(ContextSearchOptions.model_fields))
            if self.mode == "context"
            else {}
        )


class ContextEntry(TypedDict):
    uri: str
    category: str
    score: float
    detail: str
    text: str


class ContextSearchResult(TypedDict):
    entries: List[ContextEntry]
    rendered: str
    stats: Dict[str, Any]


def parse_context_result(value: Any) -> ContextSearchResult:
    """Validate the response at SDK and reader boundaries without altering it."""
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        raise ValueError("Invalid context search entries")
    if not isinstance(value.get("rendered"), str) or not isinstance(value.get("stats"), dict):
        raise ValueError("Invalid context search result")
    for item in value["entries"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(k), str) for k in ("uri", "category", "detail", "text")
        ):
            raise ValueError("Invalid context search entry")
        score = item.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Invalid context search score")
        if not item["uri"] or item["detail"] not in {"abstract", "overview", "full"}:
            raise ValueError("Invalid context search source or detail")
    return value
