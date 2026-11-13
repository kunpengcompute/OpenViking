"""Verify deployment checks reject incomplete state and preserve private configuration."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from examples.deepagents_ov_context.scripts.check import (
    CheckError,
    check_downloads,
    check_services,
    validate_session,
)


def completed_session() -> dict[str, Any]:
    return {
        "session": {
            "session_id": "test",
            "account_id": "a",
            "user_id": "u",
            "agent_id": "p",
            "created_at": 0,
        },
        "busy": False,
        "context": {
            "source": "checkpoint",
            "messages": [],
            "raw_message_count": 2,
            "cached_message_count": 2,
            "write_index": 2,
            "completed_runs": 1,
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("write_index", 1),
        ("run_id", "active"),
        ("completed_runs", 0),
        ("cached_message_count", 0),
        ("source", "ov"),
    ],
)
def test_session_check_rejects_incomplete_or_restored_state(field: str, value: object) -> None:
    detail = completed_session()
    detail["context"][field] = value
    with pytest.raises(CheckError):
        validate_session(detail)


def test_session_check_excludes_message_bodies() -> None:
    detail = completed_session()
    detail["context"]["messages"] = [{"content": "private trace"}]
    before = copy.deepcopy(detail)
    result = validate_session(detail)
    assert "private trace" not in json.dumps(result)
    assert result["write_index"] == 2
    assert detail == before
    detail["busy"] = True
    with pytest.raises(CheckError):
        validate_session(detail)


@pytest.mark.parametrize("browser,thinking", [(None, True), ("agent-browser", False)])
def test_browser_service_check_requires_both_capabilities(
    monkeypatch: pytest.MonkeyPatch, browser: str | None, thinking: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = {"status": "ready" if request.url.path == "/ready" else "ok"}
        if request.url.path == "/api/health":
            response = {"ov": {"healthy": True}, "browser": browser, "thinking": thinking}
        return httpx.Response(200, json=response)

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(CheckError):
        check_services("http://ov", "http://chat", require_browser=True)
    assert check_services("http://ov", "http://chat")["chat"]["ov"]["healthy"]


def test_download_check_rejects_html_and_path_escape(tmp_path: Path) -> None:
    directory = tmp_path / "downloads" / "run-a"
    directory.mkdir(parents=True)
    file = directory / "notice.pdf"
    file.write_text("<html>gateway error</html>")
    with pytest.raises(CheckError):
        check_downloads(tmp_path, "run-a")
    with pytest.raises(CheckError):
        check_downloads(tmp_path, "../run-a")
    file.unlink()
    file.symlink_to(tmp_path / "other.pdf")
    with pytest.raises(CheckError):
        check_downloads(tmp_path, "run-a")


def test_init_config_does_not_overwrite_existing_credentials(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[3] / "examples/deepagents_ov_context/scripts/deploy.sh"
    )
    environment = dict(os.environ, OV_CHAT_HOME=str(tmp_path / "private"))
    subprocess.run(
        ["bash", str(script), "init-config"], env=environment, check=True, capture_output=True
    )
    config = tmp_path / "private" / "models.local.json"
    assert config.stat().st_mode & 0o777 == 0o600
    config.write_text('{"api_key":"keep-existing-value"}')
    subprocess.run(
        ["bash", str(script), "init-config"], env=environment, check=True, capture_output=True
    )
    assert config.read_text() == '{"api_key":"keep-existing-value"}'


def test_presentation_check_downloads_and_counts_actual_pages(monkeypatch):
    import io

    from pptx import Presentation

    from examples.deepagents_ov_context.scripts.check import check_artifacts

    buffer = io.BytesIO()
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[0])
    deck.save(buffer)
    data = buffer.getvalue()
    route = "/api/sessions/s/artifacts/r/intro.pptx"
    manifest = {
        "filename": "intro.pptx",
        "run_id": "r",
        "url": route,
        "slides": 1,
        "bytes": len(data),
    }

    def respond(request):
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"presentation": True, "skills": ["pptx"]})
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(200, json=[manifest])
        return httpx.Response(200, content=data)

    factory = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: factory(transport=httpx.MockTransport(respond), **kwargs)
    )
    assert check_artifacts("http://localhost", "s", slides=1)["artifacts"][0]["slides"] == 1
    with pytest.raises(CheckError, match="slide count"):
        check_artifacts("http://localhost", "s", slides=10)
    manifest["url"] = "https://outside.invalid/file"
    with pytest.raises(CheckError, match="route"):
        check_artifacts("http://localhost", "s")
