# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""OV context operation errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openviking.integrations._context_contracts import (
    ContextHistoryBudgetError as ContextHistoryBudgetError,
)
from openviking.integrations._context_contracts import ContextReadError as ContextReadError

if TYPE_CHECKING:
    from .writer import WriteResult


class IdentityMismatchError(ValueError):
    """Authenticated invocation and session ownership must never diverge."""


class ContextBudgetExceeded(ValueError):
    pass


class AmbiguousWriteError(RuntimeError):
    """The server may have stored the batch. Automatic retry is unsafe."""


class IncompleteWriteError(AmbiguousWriteError):
    """Do not automatically retry; confirmed_result describes earlier batches.

    The failed batch may also have been stored. The existing middleware leaves
    its cursor unchanged on exceptions. A new Run continues from the saved
    cursor; an earlier batch may be written again.
    """

    def __init__(self, confirmed_result: WriteResult) -> None:
        self.confirmed_result = confirmed_result
        super().__init__("Batch write incomplete or outcome unknown; do not automatically retry")
