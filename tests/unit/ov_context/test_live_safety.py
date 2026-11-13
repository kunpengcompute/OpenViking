"""Destructive live controls must reject shared or mismatched OV deployments."""

from types import SimpleNamespace

import httpx
import pytest

from tests.st.ov_context.live.extended_support import DockerOV


@pytest.mark.parametrize("name", ["", "ov-office-20261026", "chat-office-20261026"])
def test_live_control_rejects_unscoped_container_names(monkeypatch, name):
    monkeypatch.setenv("OV_LIVE_CONTAINER", name)
    with pytest.raises(ValueError, match="isolated"):
        DockerOV(SimpleNamespace(url="http://127.0.0.1:19552"))


@pytest.mark.parametrize("label,port", [("false", "19552"), ("true", "19448")])
async def test_live_kill_requires_both_test_label_and_matching_port(monkeypatch, label, port):
    monkeypatch.setenv("OV_LIVE_CONTAINER", "ov-middleware-integration-safety-test")
    control = DockerOV(SimpleNamespace(url="http://127.0.0.1:19552"))
    await control.http.aclose()
    requests = []

    def inspect(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(
            200,
            json={"Config": {"Labels": {"ov.live-test": label}, "Cmd": ["python", "--port", port]}},
        )

    control.http = httpx.AsyncClient(
        transport=httpx.MockTransport(inspect), base_url="http://docker"
    )
    try:
        with pytest.raises(ValueError):
            await control.kill()
        assert requests == [("GET", "/containers/ov-middleware-integration-safety-test/json")]
    finally:
        await control.close()
