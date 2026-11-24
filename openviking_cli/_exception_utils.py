# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared exception classification for SDK clients and integrations."""

from openviking_cli.exceptions import InvalidArgumentError


def invalid_argument_error(exc: BaseException) -> bool:
    """Recognize invalid requests, including an explicitly wrapped SDK failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(getattr(current, "response", None), "status_code", None)
        if (
            isinstance(current, InvalidArgumentError)
            or getattr(current, "code", None) in ("INVALID_ARGUMENT", "INVALID_URI")
            or status in (400, 422)
        ):
            return True
        current = current.__cause__
    return False


def auth_error(exc: BaseException) -> bool:
    """Recognize SDK error codes, transport status codes and HTTP response errors."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403) or getattr(exc, "code", None) in (
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "UNAUTHORIZED",
        "FORBIDDEN",
    )
