"""Select SDK-owned summarization or OV-owned context per graph."""

from typing import Any

from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware.types import AgentMiddleware


def native_summary(model: Any, backend: Any = None) -> AgentMiddleware[Any, Any, Any]:
    """Use DeepAgents' default counting, thresholds and summary trimming."""
    return create_summarization_middleware(
        model,
        backend if backend is not None else StateBackend(),
    )


class DisabledNativeSummaryMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace the default summary slot with no hooks when OV owns context."""

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"
