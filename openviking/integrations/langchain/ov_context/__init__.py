# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session modules; Reader imports remain independent of framework orchestration."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import OVContextConfig
from .errors import (
    AmbiguousWriteError,
    ContextBudgetExceeded,
    ContextHistoryBudgetError,
    ContextReadError,
    IdentityMismatchError,
    IncompleteWriteError,
)
from .identity import OVIdentity, RunIdentity, SessionKey
from .metrics import LatencySamples, LatencySummary, Metrics, MetricsSink
from .reader import (
    AsyncMemoryClient,
    ContextReaderBase,
    ContextReadResult,
    ContextSnapshot,
    MemoryReaderConfig,
    OVContextReader,
    RecallResult,
    RecallStatus,
)
from .sdk import AsyncClientHandle, CommitReceipt, OVSessionClient, SessionInfo, TaskInfo

if TYPE_CHECKING:
    from .assembler import ContextAssembler, ContextRefreshPlan, PreparedModelRequest
    from .budget import ApproximateTokenCounter, BudgetSpec, HistoryBudget, TokenCounter
    from .commit import (
        CommitConfig,
        CommitResult,
        OpenVikingCommitPolicy,
        OVCommitPolicy,
        load_commit_config,
    )
    from .state import (
        ContextStatus,
        OVContext,
        OVContextState,
        OVPluginState,
        PendingCommit,
    )
    from .writer import MessageWriterBase, OVMessageWriter, WriterConfig, WriteResult

# State, Writer and assembly import framework orchestration; Reader stays independent.
_FRAMEWORK_EXPORTS = {
    **dict.fromkeys(
        (
            "CommitConfig",
            "CommitResult",
            "OpenVikingCommitPolicy",
            "OVCommitPolicy",
            "load_commit_config",
        ),
        "commit",
    ),
    **dict.fromkeys(
        ("ContextAssembler", "ContextRefreshPlan", "PreparedModelRequest"), "assembler"
    ),
    **dict.fromkeys(
        ("ApproximateTokenCounter", "BudgetSpec", "HistoryBudget", "TokenCounter"), "budget"
    ),
    **dict.fromkeys(
        ("ContextStatus", "OVContext", "OVContextState", "OVPluginState", "PendingCommit"), "state"
    ),
    **dict.fromkeys(
        ("MessageWriterBase", "OVMessageWriter", "WriterConfig", "WriteResult"), "writer"
    ),
}


def __getattr__(name: str) -> Any:
    module = _FRAMEWORK_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AmbiguousWriteError",
    "ApproximateTokenCounter",
    "AsyncClientHandle",
    "AsyncMemoryClient",
    "BudgetSpec",
    "CommitReceipt",
    "CommitConfig",
    "CommitResult",
    "ContextAssembler",
    "ContextBudgetExceeded",
    "ContextHistoryBudgetError",
    "ContextReadError",
    "ContextReaderBase",
    "ContextReadResult",
    "ContextRefreshPlan",
    "ContextSnapshot",
    "ContextStatus",
    "HistoryBudget",
    "IdentityMismatchError",
    "IncompleteWriteError",
    "LatencySamples",
    "LatencySummary",
    "MessageWriterBase",
    "MemoryReaderConfig",
    "Metrics",
    "MetricsSink",
    "OVContext",
    "OVContextConfig",
    "OVContextReader",
    "OVContextState",
    "OpenVikingCommitPolicy",
    "OVCommitPolicy",
    "OVIdentity",
    "OVMessageWriter",
    "OVPluginState",
    "OVSessionClient",
    "PendingCommit",
    "PreparedModelRequest",
    "RecallResult",
    "RecallStatus",
    "RunIdentity",
    "SessionInfo",
    "SessionKey",
    "TaskInfo",
    "TokenCounter",
    "WriteResult",
    "WriterConfig",
    "load_commit_config",
]
