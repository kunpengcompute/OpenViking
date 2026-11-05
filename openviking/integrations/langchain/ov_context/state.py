# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Session data only. Wire state uses plain dictionaries and native messages."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields
from typing import Literal, cast

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import Annotated, NotRequired, Required, Self, TypedDict

from openviking.integrations._message_content import comparable_content

from .errors import ContextReadError
from .identity import OVIdentity, SessionKey

RecallStatus = Literal["success", "empty", "timeout", "error"]
ContextStatus = Literal["uninitialized", "ready", "reload_required"]


class PendingCommit(TypedDict):
    task_id: str | None
    archive_id: str | None
    target_archive_id: NotRequired[str]


class OVContextState(TypedDict, total=False):
    """Checkpoint representation of OVContext; native messages remain objects."""

    session_id: Required[str]
    messages: list[BaseMessage]
    archive_id: str | None
    write_index: int
    completed_runs: int
    commit_count: int | None
    run_id: str | None
    recall_done: bool
    original_user: HumanMessage | None
    recall_status: RecallStatus | None
    context_status: ContextStatus
    pending_commit: PendingCommit | None
    last_completed_run: str | None
    last_activity: float
    summary_check_countdown: int
    auto_commit_enabled: bool | None
    owner: tuple[str, ...]


@dataclass
class OVContext:
    session_id: str
    messages: list[BaseMessage] = field(default_factory=list)
    archive_id: str | None = None
    write_index: int = 0
    completed_runs: int = 0
    commit_count: int | None = None
    run_id: str | None = None
    recall_done: bool = False
    original_user: HumanMessage | None = None
    recall_status: RecallStatus | None = None
    context_status: ContextStatus = "uninitialized"
    pending_commit: PendingCommit | None = None
    last_completed_run: str | None = None
    last_activity: float = 0.0
    summary_check_countdown: int = 0
    auto_commit_enabled: bool | None = None
    owner: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.context_status not in ("uninitialized", "ready", "reload_required"):
            raise ValueError("Invalid context_status")

    @property
    def identity(self) -> OVIdentity:
        if len(self.owner) != 4 or self.owner[3] != self.session_id:
            raise ValueError("OVContext is missing a valid session owner")
        return OVIdentity(*self.owner[:3])

    @property
    def session_key(self) -> SessionKey:
        return self.identity.session_key(self.session_id)

    @classmethod
    def from_state(cls, value: OVContextState | Self, *, copy_messages: bool = True) -> Self:
        if isinstance(value, cls):
            result = copy(value)
        else:
            result = cls(**cast(OVContextState, value))
        # Only the read-only no-op hook borrows the outer list. All mutation
        # paths use the default copy to isolate uncommitted graph state.
        if copy_messages:
            result.messages = list(result.messages)
        result.owner = tuple(result.owner)
        return result

    def to_state(self) -> OVContextState:
        # Do not dataclasses.asdict(): it deep-copies and converts BaseMessage.
        return cast(OVContextState, {item.name: getattr(self, item.name) for item in fields(self)})

    def begin_run(self, run_id: str, original_user: HumanMessage) -> None:
        if self.run_id == run_id:
            return
        self.end_run()
        self.run_id = run_id
        self.original_user = original_user

    def append(self, messages: list[BaseMessage]) -> None:
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

    def replace(self, messages: Sequence[BaseMessage], archive_id: str | None) -> None:
        replacement = list(messages)
        if self.run_id is not None and self.original_user is not None:
            from .assembler import ARCHIVED_USER, is_summary

            user = self.original_user
            view = self.messages[self._user_index(self.messages)]
            matches = [
                i
                for i, m in enumerate(replacement)
                if isinstance(m, HumanMessage)
                and m.id == user.id
                and m.additional_kwargs.get("openviking_message_kind") in (None, "user_query")
                and (
                    not m.additional_kwargs.get("openviking_source_message_ids")
                    or (
                        m.additional_kwargs.get("openviking_message_kind") == "user_query"
                        and m.additional_kwargs["openviking_source_message_ids"] == [user.id]
                    )
                )
            ]
            if not matches:
                if any(m.id == user.id for m in replacement):
                    raise ContextReadError("Current user ID collides with a refreshed message")
                matches = [
                    i
                    for i, m in enumerate(replacement)
                    if isinstance(m, HumanMessage)
                    and not is_summary(m)
                    and not m.additional_kwargs.get("openviking_source_message_ids")
                    and m.additional_kwargs.get("openviking_message_kind") in (None, "user_query")
                    and comparable_content(m.content) == comparable_content(user.content)
                ]
            if len(matches) != 1:
                raise ContextReadError("Current Run user message must have exactly one cache match")
            index = matches[0]
            if comparable_content(replacement[index].content) not in (
                comparable_content(user.content),
                comparable_content(view.content),
            ):
                raise ContextReadError("Refreshed user message does not match the original input")
            archived = replacement[index].additional_kwargs.get(ARCHIVED_USER) is True
            if archived != (view.additional_kwargs.get(ARCHIVED_USER) is True):
                view = view.model_copy(deep=True)
                if archived:
                    view.additional_kwargs[ARCHIVED_USER] = True
                else:
                    view.additional_kwargs.pop(ARCHIVED_USER, None)
            replacement[index] = view
        self.messages, self.archive_id = replacement, archive_id

    def inject_recall(self, content: str, status: RecallStatus) -> None:
        if self.recall_done:
            return
        if self.run_id is None or self.original_user is None:
            raise RuntimeError("begin_run() must precede recall injection")
        if status not in ("success", "empty", "timeout", "error"):
            raise ValueError("Invalid recall status")
        if status == "success" and content:
            from .assembler import INJECTED

            user = self.original_user
            # A copied view cannot retain object identity as its only Run anchor.
            if not user.id:
                raise ContextReadError("Current Run user message needs a non-empty native ID")
            index = self._user_index(self.messages)
            prefix = f"[参考记忆]\n{content}\n\n[用户原文]\n"
            body = (
                prefix + user.content
                if isinstance(user.content, str)
                else [{"type": "text", "text": prefix}, *user.content]
            )
            replacement = user.model_copy(
                update={
                    "content": deepcopy(body),
                    "additional_kwargs": {**user.additional_kwargs, INJECTED: True},
                },
                deep=True,
            )
            self.messages[index] = replacement
        self.recall_done, self.recall_status = True, status

    def end_run(self) -> None:
        if self.original_user:
            from .assembler import ARCHIVED_USER, is_user_anchor

            restored = []
            for message in self.messages:
                if message.id == self.original_user.id:
                    original = self.original_user
                    if is_user_anchor(message):
                        original = original.model_copy(deep=True)
                        original.additional_kwargs[ARCHIVED_USER] = True
                    restored.append(original)
                else:
                    restored.append(message)
            self.messages = restored
        self.run_id = self.original_user = self.recall_status = None
        self.recall_done = False


class OVPluginState(AgentState):
    # A plain mapping avoids requiring unsafe custom-class deserialization.
    ov_context: NotRequired[Annotated[OVContextState, PrivateStateAttr]]
