# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""UT-only OVContext substitute; production cache belongs to its module owner."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from langchain_core.messages import BaseMessage, HumanMessage

from openviking.integrations.ov_memoryRetrieve.context_messages import comparable_content
from openviking.integrations.ov_memoryRetrieve.types import ContextReadError, RecallStatus


class FakeOVContext:
    """Simulate document-level cache behavior for offline reader contract tests only."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[BaseMessage] = []
        # Applied done/failed boundary, independent of the overview's source archive.
        self.archive_id: str | None = None
        self.run_id: str | None = None
        self.recall_done = False
        self.recall_status: RecallStatus | None = None
        self.original_user: HumanMessage | None = None

    def begin_run(self, run_id: str, original_user: HumanMessage) -> None:
        """Same-Run reentry retains recall; a new Run cleans up any previous temporary view."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        if self.run_id == run_id:
            return
        self.end_run()
        self.run_id = run_id
        self.original_user = original_user

    def append(self, messages: Sequence[BaseMessage]) -> None:
        """Append already-confirmed native increments in framework order."""
        self.messages.extend(messages)

    def _user_index(self, messages: Sequence[BaseMessage]) -> int:
        user = self.original_user
        indices = [
            i
            for i, message in enumerate(messages)
            if isinstance(message, HumanMessage)
            and user is not None
            and (message is user or (user.id is not None and message.id == user.id))
        ]
        if len(indices) != 1:
            raise ContextReadError("Current Run user message must have exactly one cache match")
        return indices[0]

    def inject_recall(self, content: str, status: RecallStatus) -> None:
        """Record all outcomes and add successful recall only to a user message copy."""
        if self.recall_done:
            return
        if self.run_id is None or self.original_user is None:
            raise RuntimeError("begin_run() must precede recall injection")
        if status not in {"success", "empty", "timeout", "error"}:
            raise ValueError("Unknown recall status")
        if status == "success" and content:
            index = self._user_index(self.messages)
            user = self.original_user
            prefix = f"[参考记忆]\n{content}\n\n[用户原文]\n"
            body = (
                prefix + user.content
                if isinstance(user.content, str)
                else [
                    {"type": "text", "text": prefix},
                    *user.content,
                ]
            )
            # Pydantic's deep=True copies existing fields, but not values supplied in update.
            self.messages[index] = user.model_copy(update={"content": deepcopy(body)}, deep=True)
        self.recall_done = True
        self.recall_status = status

    def replace(self, messages: Sequence[BaseMessage], archive_id: str | None) -> None:
        """Replace view/boundary only; simulate preserving this Run's state."""
        replacement = list(messages)
        if self.run_id is not None and self.original_user is not None:
            index = self._user_index(replacement)
            if comparable_content(replacement[index].content) != comparable_content(
                self.original_user.content
            ):
                raise ContextReadError("Refreshed user message does not match the original input")
            old_index = self._user_index(self.messages)
            replacement[index] = self.messages[old_index]
        self.messages = replacement
        self.archive_id = archive_id

    def end_run(self) -> None:
        """Restore the original user view and clear Run-local recall state."""
        if self.original_user is not None:
            user = self.original_user
            for i, message in enumerate(self.messages):
                if isinstance(message, HumanMessage) and (
                    message is user or (user.id is not None and message.id == user.id)
                ):
                    self.messages[i] = user
        self.run_id = None
        self.original_user = None
        self.recall_done = False
        self.recall_status = None
