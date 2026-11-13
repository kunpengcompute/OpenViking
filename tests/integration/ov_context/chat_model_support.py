"""Deterministic model for offline Chat route and persistence tests."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class NoToolModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> NoToolModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="done",
                        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    )
                )
            ]
        )
