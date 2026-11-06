# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Search Service for OpenViking.

Provides semantic search operations: search, find.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from openviking.core.uri_validation import validate_optional_viking_uris
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError
from openviking_cli.retrieve.context import ContextSearchOptions
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session import Session

logger = get_logger(__name__)


def _ensure_non_empty_query(query: str) -> None:
    if not query.strip():
        raise InvalidArgumentError("Search query must not be empty.")


class SearchService:
    """Semantic search service."""

    def __init__(self, viking_fs: Optional[VikingFS] = None):
        self._viking_fs = viking_fs

    def set_viking_fs(self, viking_fs: VikingFS) -> None:
        """Set VikingFS instance (for deferred initialization)."""
        self._viking_fs = viking_fs

    def _ensure_initialized(self) -> VikingFS:
        """Ensure VikingFS is initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        session: Optional["Session"] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
        mode: str = "list",
        **context_options: Any,
    ) -> Any:
        """Return ranked hits, or deterministic context without session planning.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            session: Session object for context
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult for list mode; ContextSearchResult for context mode.
        """
        _ensure_non_empty_query(query)
        target_uri = validate_optional_viking_uris(target_uri, field_name="target_uri")
        viking_fs = self._ensure_initialized()

        try:
            options = ContextSearchOptions(mode=mode, **context_options)
        except ValueError as exc:
            raise InvalidArgumentError(str(exc)) from exc
        if options.mode == "context":
            from openviking.retrieve.context_assembler.pipeline import assemble_context

            # Return before consulting the session or entering VikingFS.search:
            # recall never invokes a query planner or an intent model.
            return await assemble_context(
                service=self,
                viking_fs=viking_fs,
                query=query,
                ctx=ctx,
                options=options,
                target_uri=target_uri,
                limit=limit,
                score_threshold=score_threshold,
                filter=filter,
                level=level,
            )

        session_info = None
        if session:
            session_info = await session.get_context_for_search(query)

        result = await viking_fs.search(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            session_info=session_info,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
        )
        return result

    async def find(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
    ) -> Any:
        """Semantic search without session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        _ensure_non_empty_query(query)
        target_uri = validate_optional_viking_uris(target_uri, field_name="target_uri")
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
        )
        return result
