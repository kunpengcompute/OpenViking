# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Bounded retries for transient OV failures.

Every retry in the integration is gated by ``OVContextConfig.retry_attempts``.
Zero means one attempt, without sleeps or swallowed errors.

Classification is deliberately narrow:

* reads are idempotent and may retry on transport errors and retryable OV
  status codes;
* writes are retried only when the batch provably never reached OV;
* commits retry only known pre-send connection failures; uncertain outcomes
  propagate without task discovery or automatic resubmission.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from openviking_cli.exceptions import (
    AbortedError,
    ConflictError,
    DeadlineExceededError,
    InternalError,
    ResourceExhaustedError,
    UnavailableError,
)

from .errors import ContextReadError
from .sdk import auth_error

T = TypeVar("T")

_PRE_SEND = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

_TRANSIENT = (
    httpx.TransportError,
    asyncio.TimeoutError,
    ConflictError,
    AbortedError,
    UnavailableError,
    InternalError,
    ResourceExhaustedError,
    DeadlineExceededError,
)

_TRANSIENT_CODES = frozenset(
    {
        "ABORTED",
        "CONFLICT",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
    }
)


def retry_delay(attempt: int, *, base: float = 0.1, maximum: float = 0.3) -> float:
    """Exponential backoff with a little jitter; attempt starts at zero."""
    delay = min(maximum, base * (2**attempt))
    return delay + random.uniform(0, 0.05)


def retryable_read(exc: BaseException) -> bool:
    """True for failures where repeating an idempotent read is safe."""
    if auth_error(exc):
        return False
    if isinstance(exc, ContextReadError):
        cause = exc.__cause__
        return cause is not None and retryable_read(cause)
    if isinstance(exc, _TRANSIENT):
        return True
    return getattr(exc, "code", None) in _TRANSIENT_CODES


def retryable_commit(exc: BaseException) -> bool:
    """Retry only connection failures that prove no commit request was sent."""
    return not auth_error(exc) and isinstance(exc, _PRE_SEND)


def retryable_write(exc: BaseException) -> bool:
    """True only when no batch reached OV.

    ``IncompleteWriteError`` carries the confirmed prefix. A zero
    ``written_count`` alone is not enough: the failed batch may have been
    stored before the connection broke. The cause must therefore be a
    pre-send error.
    """
    if auth_error(exc):
        return False
    result = getattr(exc, "confirmed_result", None)
    if result is None or getattr(result, "written_count", 0) != 0:
        return False
    return isinstance(exc.__cause__, _PRE_SEND)


async def retry_call(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    retry_on: Callable[[BaseException], bool] = retryable_read,
) -> T:
    """Run ``operation`` once plus at most ``retries`` extra attempts.

    ``retries <= 0`` performs exactly one call and re-raises immediately.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if attempt >= retries or not retry_on(exc):
                raise
            await asyncio.sleep(retry_delay(attempt))
            attempt += 1
