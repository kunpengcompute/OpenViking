# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Invalid requests must survive transport and partial-write error wrapping."""

import httpx
import pytest

from openviking.integrations.langchain.ov_context.errors import IncompleteWriteError
from openviking.integrations.langchain.ov_context.writer import WriteResult
from openviking_cli._exception_utils import invalid_argument_error
from openviking_cli.exceptions import InvalidArgumentError, InvalidURIError, OpenVikingError


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        InvalidArgumentError("invalid identity"),
        InvalidURIError("invalid", "invalid path"),
        OpenVikingError("invalid request", code="INVALID_ARGUMENT"),
        httpx.HTTPStatusError(
            "invalid request",
            request=httpx.Request("GET", "http://ov.test"),
            response=httpx.Response(400),
        ),
        httpx.HTTPStatusError(
            "invalid request",
            request=httpx.Request("GET", "http://ov.test"),
            response=httpx.Response(422),
        ),
    ],
)
def test_invalid_argument_is_recognized_through_write_error(error, wrapped):
    if wrapped:
        outer = IncompleteWriteError(WriteResult([], 0, 0))
        outer.__cause__ = error
        error = outer
    assert invalid_argument_error(error)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("offline"),
        httpx.ReadTimeout("slow response"),
        OpenVikingError("unavailable", code="UNAVAILABLE"),
        ValueError("invalid JSON response"),
        RuntimeError("unexpected response"),
    ],
)
def test_optional_service_failures_are_not_classified_as_invalid_requests(error):
    assert not invalid_argument_error(error)


def test_cause_cycle_does_not_hang_classification():
    error = RuntimeError("cyclic cause")
    error.__cause__ = error
    assert not invalid_argument_error(error)
