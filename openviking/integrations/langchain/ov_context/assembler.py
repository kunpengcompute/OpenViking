# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model request assembly and archive refresh planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypedDict, cast

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from openviking.integrations._message_content import comparable_content

from .budget import ApproximateTokenCounter, BudgetSpec, HistoryBudget, TokenCounter
from .config import OVContextConfig
from .context_guide import BATCH_SEARCH_GUIDANCE as BATCH_SEARCH_GUIDANCE
from .context_guide import GUIDE as GUIDE
from .context_guide import RULES as RULES
from .context_guide import build_context_guide
from .contracts import ContextSnapshot
from .errors import ContextBudgetExceeded, ContextReadError
from .state import OVContext

INJECTED = "ov_context_injected"
ARCHIVED_USER = "ov_context_archived_user"


def is_summary(message: BaseMessage) -> bool:
    return message.id == "ov:summary" or (message.id or "").startswith("ov_summary:")


def is_user_anchor(message: BaseMessage) -> bool:
    return (
        isinstance(message, HumanMessage) and message.additional_kwargs.get(ARCHIVED_USER) is True
    )


def query_text(message: HumanMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        str(block.get("text", ""))
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def validate_tool_pairs(messages: Sequence[BaseMessage]) -> None:
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError("Orphan or duplicate tool result")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError("Incomplete tool-result group")
            if isinstance(message, AIMessage):
                if message.invalid_tool_calls:
                    raise ValueError("Model input contains invalid tool calls")
                for call in message.tool_calls:
                    call_id = call.get("id")
                    if not call_id or call_id in seen:
                        raise ValueError("Missing or duplicate tool-call ID")
                    pending.add(call_id)
                    seen.add(call_id)
    if pending:
        raise ValueError("Model input contains unfinished tool calls")


_GUIDE_METADATA = "ov_context_system_guide"


_SystemContent = str | list[str | dict[str, Any]]


class _GuideMarker(TypedDict):
    format: Literal["string", "blocks"]
    position: int
    text: str


def _system_with_rules(system: SystemMessage | None, rules: str) -> SystemMessage:
    """Replace only the span/block explicitly marked by our previous assembly."""
    content: _SystemContent = system.content if system else ""
    kwargs = dict(system.additional_kwargs) if system else {}
    previous = kwargs.get(_GUIDE_METADATA)
    if isinstance(previous, dict):
        position, text = previous.get("position"), previous.get("text")
        if type(position) is int and position >= 0 and isinstance(text, str) and text:
            if isinstance(content, str) and previous.get("format") == "string":
                if content[position : position + len(text)] == text:
                    content = content[:position] + content[position + len(text) :]
            elif isinstance(content, list) and previous.get("format") == "blocks":
                if position < len(content) and content[position] == {"type": "text", "text": text}:
                    content = [*content[:position], *content[position + 1 :]]
    marker: _GuideMarker
    if isinstance(content, str):
        suffix = ("\n\n" if content else "") + rules
        marker = {"format": "string", "position": len(content), "text": suffix}
        content = content + suffix
    else:
        marker = {"format": "blocks", "position": len(content), "text": rules}
        content = [*content, {"type": "text", "text": rules}]
    kwargs[_GUIDE_METADATA] = marker
    if system is None:
        return SystemMessage(content=content, additional_kwargs=kwargs)
    return system.model_copy(update={"content": content, "additional_kwargs": kwargs})


@dataclass(frozen=True)
class PreparedModelRequest:
    request: ModelRequest[Any]
    budget: BudgetSpec
    input_tokens: int
    fixed_tokens: int
    base_context_tokens: int
    history_target_tokens: int
    base_context_over_target: bool
    capacity_reached: bool


@dataclass(frozen=True)
class ContextRefreshPlan:
    source_archive_id: str | None
    run_id: str | None
    source_write_index: int
    user_message: HumanMessage | None
    recent_messages: tuple[BaseMessage, ...]
    keep_recent_count: int
    archive_message_count: int


class ContextAssembler:
    def __init__(self, config: OVContextConfig, token_counter: TokenCounter | None = None) -> None:
        self.config = config
        self.token_counter = token_counter or ApproximateTokenCounter()

    def budget(self, request: ModelRequest[Any]) -> BudgetSpec:
        profile = getattr(request.model, "profile", None) or {}
        window = self.config.context_window or profile.get("context_window")
        if not window:
            raise ValueError("Configure context_window or provide a model profile")
        return BudgetSpec(
            window,
            self.config.output_reserve,
            self.config.safety_margin,
            max_input_tokens=self.config.max_input_tokens or profile.get("max_input_tokens"),
            base_context_ratio=self.config.base_context_ratio,
            capacity_ratio=self.config.capacity_ratio,
        )

    def _system_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system = request.system_message.model_copy(deep=True) if request.system_message else None
        names = {
            tool.get("name", tool.get("function", {}).get("name", ""))
            if isinstance(tool, dict)
            else getattr(tool, "name", "")
            for tool in request.tools
        }
        return request.override(
            system_message=_system_with_rules(system, build_context_guide(names))
        )

    def _count(self, request: ModelRequest[Any]) -> int:
        value = self.token_counter(request)
        if type(value) is not int or value < 0:
            raise ValueError("TokenCounter must return a non-negative integer")
        return value

    def check_input_budget(
        self, request: ModelRequest[Any], *, budget: BudgetSpec | None = None
    ) -> int:
        """Check the exact request being sent, including a local fallback request."""
        budget = budget if budget is not None else self.budget(request)
        tokens = self._count(request)
        if tokens > budget.input_limit:
            raise ContextBudgetExceeded(f"OV context input {tokens} exceeds {budget.input_limit}")
        return tokens

    def history_budget(self, request: ModelRequest[Any]) -> HistoryBudget:
        fixed = self._system_request(request).override(messages=[])
        budget = self.budget(fixed)
        tokens = self._count(fixed)
        return HistoryBudget(budget, tokens, max(0, budget.base_context_limit - tokens))

    def assemble(self, request: ModelRequest[Any], context: OVContext) -> ModelRequest[Any]:
        return self.prepare(request, context).request

    @staticmethod
    def _validate_messages(messages: Sequence[BaseMessage]) -> None:
        if any(not isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in messages):
            raise ContextReadError("Snapshot must contain native context messages")
        ids = [m.id for m in messages]
        if any(not key for key in ids) or len(set(ids)) != len(ids):
            raise ContextReadError("Snapshot needs unique native message IDs")
        try:
            validate_tool_pairs(messages)
        except ValueError as exc:
            raise ContextReadError("Snapshot contains invalid tool-call pairs") from exc

    def plan_refresh(self, context: OVContext, *, keep_recent_count: int) -> ContextRefreshPlan:
        if type(keep_recent_count) is not int or not 0 <= keep_recent_count <= 10000:
            raise ValueError("keep_recent_count must be an integer from 0 to 10000")
        messages = [m for m in context.messages if not is_summary(m) and not is_user_anchor(m)]
        self._validate_messages(messages)
        # Writer combines adjacent results in one transport spec, but OV splits
        # that aggregate into one persisted record per native ToolMessage.
        ranges = [(index, index + 1) for index in range(len(messages))]
        cutoff = ranges[-keep_recent_count][0] if 0 < keep_recent_count < len(ranges) else 0
        if keep_recent_count == 0:
            cutoff = len(messages)
        user = None
        if context.original_user is not None and context.run_id is not None:
            user = cast(
                HumanMessage, context.messages[context._user_index(context.messages)]
            ).model_copy(deep=True)
            user_index = next((i for i, m in enumerate(messages) if m.id == user.id), 0)
            latest = next(
                (
                    i
                    for i in range(len(messages) - 1, user_index, -1)
                    if isinstance(messages[i], AIMessage)
                ),
                user_index,
            )
            cutoff = min(cutoff, latest)
        while cutoff:
            try:
                validate_tool_pairs(messages[:cutoff])
                break
            except ValueError:
                cutoff -= 1
        archived = sum(end <= cutoff for _, end in ranges)
        return ContextRefreshPlan(
            context.archive_id,
            context.run_id,
            context.write_index,
            user,
            tuple(m.model_copy(deep=True) for m in messages[cutoff:]),
            len(ranges) - archived,
            archived,
        )

    @staticmethod
    def _apply_delta(messages: list[BaseMessage], delta: Sequence[BaseMessage]) -> None:
        for message in delta:
            index = next(
                (
                    i
                    for i, existing in enumerate(messages)
                    if existing.id == message.id
                    or isinstance(message, ToolMessage)
                    and isinstance(existing, ToolMessage)
                    and existing.tool_call_id == message.tool_call_id
                ),
                None,
            )
            if index is None:
                messages.append(message.model_copy(deep=True))
            else:
                messages[index] = message.model_copy(deep=True)

    def refresh(
        self,
        context: OVContext,
        snapshot: ContextSnapshot,
        *,
        plan: ContextRefreshPlan | None = None,
        appended_messages: Sequence[BaseMessage] = (),
    ) -> OVContext:
        """Build a candidate; only its selected tail survives an archive change."""
        self._validate_messages(snapshot.messages)
        summaries = [m for m in snapshot.messages if is_summary(m)]
        if len(summaries) > 1:
            raise ContextReadError("Snapshot must contain at most one summary")
        if plan is not None:
            if (
                plan.source_archive_id != context.archive_id
                or plan.run_id != context.run_id
                or plan.source_write_index > context.write_index
            ):
                raise ContextReadError("Refresh plan no longer matches the session")
            if (
                snapshot.archive_id == plan.source_archive_id
                or not summaries
                or not summaries[0].content
            ):
                raise ContextReadError("Compression did not produce a usable summary")
            messages = [m.model_copy(deep=True) for m in plan.recent_messages]
            self._apply_delta(messages, appended_messages)
            user = plan.user_message
            if user is not None:
                user = user.model_copy(deep=True)
                match = next((i for i, m in enumerate(messages) if m.id == user.id), None)
                if match is None:
                    user.additional_kwargs[ARCHIVED_USER] = True
                    messages.insert(0, user)
                else:
                    user.additional_kwargs.pop(ARCHIVED_USER, None)
                    messages[match] = user
            messages = [summaries[0].model_copy(deep=True), *messages]
        else:
            messages = [m.model_copy(deep=True) for m in snapshot.messages]
            user = context.original_user
            if user is not None and context.run_id is not None:
                matches = [
                    i
                    for i, m in enumerate(messages)
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
                    if any(m.id == user.id for m in messages):
                        raise ContextReadError("Current user ID collides with a refreshed message")
                    matches = [
                        i
                        for i, m in enumerate(messages)
                        if isinstance(m, HumanMessage)
                        and not is_summary(m)
                        and not m.additional_kwargs.get("openviking_source_message_ids")
                        and m.additional_kwargs.get("openviking_message_kind")
                        in (None, "user_query")
                        and comparable_content(m.content) == comparable_content(user.content)
                    ]
                if len(matches) > 1:
                    raise ContextReadError("Current Run user message has ambiguous cache matches")
                retained_user = context.messages[context._user_index(context.messages)].model_copy(
                    deep=True
                )
                if matches:
                    index = matches[0]
                    if comparable_content(messages[index].content) != comparable_content(
                        user.content
                    ):
                        raise ContextReadError(
                            "Refreshed user message does not match the original input"
                        )
                    retained_user.additional_kwargs.pop(ARCHIVED_USER, None)
                    messages[index] = retained_user
                else:
                    retained_user.additional_kwargs[ARCHIVED_USER] = True
                    messages.insert(len(summaries), retained_user)
        self._validate_messages(messages)
        candidate = OVContext.from_state(context)
        candidate.messages = [m.model_copy(deep=True) for m in context.messages]
        candidate.replace(messages, snapshot.archive_id)
        candidate.context_status = "ready"
        return candidate

    def prepare(
        self,
        request: ModelRequest[Any],
        context: OVContext,
        *,
        history_budget: HistoryBudget | None = None,
    ) -> PreparedModelRequest:
        """Build an isolated request from retained messages and the task anchor."""
        messages = context.messages  # Never inspect request.messages / state['messages'] here.
        if any(not isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in messages):
            raise TypeError("OV context must contain native LangChain messages")
        ids = [m.id for m in messages]
        if any(not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("OV context needs unique native message IDs")
        validate_tool_pairs(messages)
        updated = self._system_request(request).override(
            messages=cast(
                list[AnyMessage], [message.model_copy(deep=True) for message in messages]
            ),
        )
        history = history_budget if history_budget is not None else self.history_budget(request)
        budget = history.budget
        tokens = self.check_input_budget(updated, budget=budget)
        base_messages = updated.messages
        if context.original_user is not None:
            user_index = context._user_index(messages)
            base_messages = updated.messages[:user_index] + updated.messages[user_index + 1 :]
        base_tokens = (
            self._count(updated.override(messages=base_messages))
            if base_messages
            else history.fixed_tokens
        )
        return PreparedModelRequest(
            updated,
            budget,
            tokens,
            history.fixed_tokens,
            base_tokens,
            history.history_target_tokens,
            base_tokens > budget.base_context_limit,
            tokens >= budget.capacity_limit,
        )
