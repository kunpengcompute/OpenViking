# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Metric collection uses portable Docker exec and diagnoses failed sampling."""

import json
import struct
from types import SimpleNamespace

import httpx
import pytest

from tests.st.ov_context.live.extended_support import DockerOV


def frame(stream, payload):
    return struct.pack(">BxxxI", stream, len(payload)) + payload


@pytest.fixture
def docker_response():
    return {
        "output": frame(1, b'{"fds":12,"rss_kib":3456}'),
        "running": False,
        "exit_code": 0,
        "calls": [],
    }


@pytest.fixture
async def control(monkeypatch, docker_response):
    name = "ov-middleware-integration-metrics-test"
    monkeypatch.setenv("OV_LIVE_CONTAINER", name)
    value = DockerOV(SimpleNamespace(url="http://127.0.0.1:29552"))
    await value.http.aclose()

    def respond(request):
        docker_response["calls"].append((request.method, request.url.path))
        if request.url.path == f"/containers/{name}/stats":
            return httpx.Response(
                200,
                json={
                    "memory_stats": {"usage": 1024, "limit": 4096, "stats": {"inactive_file": 256}},
                    "pids_stats": {"current": 3},
                    "cpu_stats": {"cpu_usage": {"total_usage": 1234}},
                },
            )
        if request.url.path == f"/containers/{name}/exec":
            command = json.loads(request.content)["Cmd"]
            assert command[:2] == ["python", "-c"]
            return httpx.Response(200, json={"Id": "sample"})
        if request.url.path == "/exec/sample/start":
            return httpx.Response(200, content=docker_response["output"])
        if request.url.path == "/exec/sample/json":
            return httpx.Response(
                200,
                json={
                    "Running": docker_response["running"],
                    "ExitCode": docker_response["exit_code"],
                },
            )
        raise AssertionError(f"Unexpected Docker API request: {request.url.path}")

    value.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://docker")
    try:
        yield value
    finally:
        await value.close()


async def test_metrics_uses_path_python_and_checks_exec_result(control, docker_response):
    result = await control.metrics()
    assert result["fds"] == 12 and result["rss_kib"] == 3456
    assert result["working_set"] == 768 and result["memory_limit"] == 4096
    assert result["pids"] == 3 and result["cpu_total_ns"] == 1234
    assert docker_response["calls"][-1] == ("GET", "/exec/sample/json")


async def test_metrics_keeps_stderr_out_of_fragmented_json(control, docker_response):
    docker_response["output"] = (
        frame(1, b'{"fds":12,') + frame(2, b"diagnostic warning\n") + frame(1, b'"rss_kib":3456}')
    )
    result = await control.metrics()
    assert result["fds"] == 12 and result["rss_kib"] == 3456


@pytest.mark.parametrize("framed", [True, False])
async def test_metrics_reports_command_failure_before_json(control, docker_response, framed):
    error = b"exec: python: executable file not found"
    docker_response.update(exit_code=127, output=frame(2, error) if framed else error)
    with pytest.raises(RuntimeError, match="exit=127.*executable file not found"):
        await control.metrics()


async def test_metrics_rejects_unfinished_exec(control, docker_response):
    docker_response.update(running=True, exit_code=None)
    with pytest.raises(RuntimeError, match="has not finished"):
        await control.metrics()


@pytest.mark.parametrize(
    "output,message",
    [
        (b"", "no process metrics"),
        (frame(1, b"not JSON"), "invalid JSON"),
        (frame(1, b"\xff"), "invalid JSON"),
        (frame(1, b"[]"), "invalid process metrics"),
        (frame(1, b'{"fds":12}'), "invalid process metrics"),
        (frame(1, b'{"fds":-1,"rss_kib":123}'), "invalid process metrics"),
        (frame(1, b'{"fds":true,"rss_kib":123}'), "invalid process metrics"),
        (b"\x01\0", "Truncated Docker exec"),
        (struct.pack(">BxxxI", 1, 10) + b"{}", "Truncated Docker exec"),
        (frame(3, b"{}"), "Invalid Docker exec"),
    ],
)
async def test_metrics_rejects_unusable_output(control, docker_response, output, message):
    docker_response["output"] = output
    with pytest.raises(RuntimeError, match=message):
        await control.metrics()
