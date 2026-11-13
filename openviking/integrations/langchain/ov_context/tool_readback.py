# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Session-bound bounded tool-result backtracking; runtime identity is injected."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Protocol

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import ValidationError as V1ValidationError

from .identity import RunIdentity
from .sdk import AsyncClientHandle, not_found
from .state import OVContext
from .tool_result_search import MAX_MATCHES, MAX_QUERIES, MAX_QUERY_CHARS, search_many


class ToolResultAccess(Protocol):
    """Tool-result SDK access supplied by the calling orchestrator."""

    client: AsyncClientHandle


SearchQuery = Annotated[
    str, Field(min_length=1, max_length=MAX_QUERY_CHARS, pattern=r"\S", strict=True)
]


# Validation schemas are process-wide; tool closures remain bound to each Run.
# ToolRuntime is injected from the coroutine signature by LangGraph.
class _ReadArguments(BaseModel):
    ref: str
    offset: Annotated[int, Field(ge=0, strict=True)] = 0
    limit: Annotated[int, Field(ge=1, le=20_000, strict=True)] = 20_000


class _SearchArguments(BaseModel):
    ref: str
    queries: Annotated[list[SearchQuery], Field(min_length=1, max_length=MAX_QUERIES, strict=True)]
    limit: Annotated[int, Field(ge=1, le=MAX_MATCHES, strict=True)] = MAX_MATCHES


class _ListArguments(BaseModel):
    limit: Annotated[int, Field(ge=1, le=50, strict=True)] = 50


def _validation_error(error: ValidationError | V1ValidationError) -> str:
    errors = (
        error.errors(include_url=False, include_context=False, include_input=False)
        if isinstance(error, ValidationError)
        else error.errors()
    )
    details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in errors)
    return f"Invalid tool arguments: {details}. Correct the arguments and retry."


def make_session_tools(
    engine: ToolResultAccess, identity_resolver: Callable[..., RunIdentity]
) -> list[BaseTool]:
    def session(runtime: ToolRuntime) -> RunIdentity:
        identity = identity_resolver(runtime.state, runtime)
        context = OVContext.from_state(runtime.state["ov_context"], copy_messages=False)
        if context.session_key != identity.session_key or context.run_id != identity.run_id:
            raise ValueError("Tool request belongs to a different session/identity/Run")
        return identity

    def result_id(ref: str, sid: str) -> str:
        if not ref.startswith("viking://"):
            raise ToolException(
                "Invalid ref: expected an OV tool-result reference, not a local path or URL. "
                "Use the file-reading tool for local files, or call "
                "openviking_tool_result_list to obtain this session's references."
            )
        prefix = f"/sessions/{sid}/tool-results/"
        if prefix not in ref:
            raise ValueError("Tool result reference belongs to a different session")
        tail = ref.split(prefix, 1)[1]
        if not tail or "/" in tail or "\\" in tail or tail in (".", ".."):
            raise ValueError("Invalid tool result reference")
        return tail

    @tool(args_schema=_ReadArguments)
    async def openviking_tool_result_read(
        ref: str,
        runtime: ToolRuntime,
        offset: Annotated[int, Field(ge=0, strict=True)] = 0,
        limit: Annotated[int, Field(ge=1, le=20_000, strict=True)] = 20_000,
    ) -> dict[str, Any]:
        """Read this session's tool result; offset >= 0 and limit is 1..20000 characters."""
        identity = session(runtime)
        sid = identity.session_id
        tool_result_id = result_id(ref, sid)
        try:
            return await engine.client.call(
                "read_tool_result",
                identity=identity.sdk_identity,
                session_id=sid,
                tool_result_id=tool_result_id,
                offset=offset,
                limit=limit,
            )
        except Exception as exc:
            if not not_found(exc):
                raise
            raise ToolException(
                f"Tool result not found: {tool_result_id}. "
                "Call openviking_tool_result_list to see this session's references."
            ) from exc

    @tool(args_schema=_SearchArguments)
    async def openviking_tool_result_search_many(
        ref: str,
        queries: Annotated[
            list[SearchQuery], Field(min_length=1, max_length=MAX_QUERIES, strict=True)
        ],
        runtime: ToolRuntime,
        limit: Annotated[int, Field(ge=1, le=MAX_MATCHES, strict=True)] = MAX_MATCHES,
    ) -> dict[str, Any]:
        """Search one original with a real JSON array of 1..4 keywords, including a single term.

        Terms match independently with OR semantics, with limit 1..20 hits per term.
        Use independently known alternatives or information points for the same ref.
        Returns per-term counts and deduplicated exact windows with character offsets,
        at most 6000 source characters; partial marks omitted windows or a hit limit.
        Do not encode the array as a string or add terms solely to fill the array.
        """
        identity = session(runtime)
        sid = identity.session_id
        tool_result_id = result_id(ref, sid)

        async def search_one(query: str, count: int) -> dict[str, Any]:
            return await engine.client.call(
                "search_tool_result",
                identity=identity.sdk_identity,
                session_id=sid,
                tool_result_id=tool_result_id,
                query=query,
                limit=count,
            )

        try:
            return await search_many(search_one, queries, limit)
        except Exception as exc:
            if not not_found(exc):
                raise
            raise ToolException(
                f"Tool result not found: {tool_result_id}. "
                "Call openviking_tool_result_list to see this session's references."
            ) from exc

    @tool(args_schema=_ListArguments)
    async def openviking_tool_result_list(
        runtime: ToolRuntime, limit: Annotated[int, Field(ge=1, le=50, strict=True)] = 50
    ) -> dict[str, Any]:
        """List up to 50 tool-result references in this session."""
        identity = session(runtime)
        return await engine.client.call(
            "list_tool_results",
            identity=identity.sdk_identity,
            session_id=identity.session_id,
            limit=limit,
        )

    tools = [
        openviking_tool_result_read,
        openviking_tool_result_search_many,
        openviking_tool_result_list,
    ]
    for session_tool in tools:
        # Argument validation, non-OV refs and missing tool results are recoverable;
        # identity and other SDK failures retain their existing handling.
        session_tool.handle_validation_error = _validation_error
        session_tool.handle_tool_error = True
    return tools
