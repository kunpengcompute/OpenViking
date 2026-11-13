"""Chat-level control of the DeepAgents planning capability."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware


class DisabledTodoListMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace the built-in planning slot with no tools, prompt, or hooks."""

    @property
    def name(self) -> str:
        return "TodoListMiddleware"
