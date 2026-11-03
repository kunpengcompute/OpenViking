# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared exception classification for SDK clients and integrations."""


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
