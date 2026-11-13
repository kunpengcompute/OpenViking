"""Fixtures for offline UTs; invoke pytest with this directory as --confcutdir."""

import socket
from unittest.mock import AsyncMock

import pytest

from openviking.integrations.langchain.ov_context.reader import AsyncMemoryClient
from openviking_cli.retrieve.types import FindResult


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def unexpected_connection(*args, **kwargs):
        raise AssertionError("Memory reader UT attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    monkeypatch.setattr(socket, "create_connection", unexpected_connection)


@pytest.fixture
def client():
    mocked = AsyncMock(spec=AsyncMemoryClient)
    mocked.find.return_value = FindResult(memories=[], resources=[], skills=[])
    mocked.get_session_context.return_value = {
        "latest_finished_archive_id": None,
        "latest_archive_overview": "",
        "messages": [],
        "stats": {"totalArchives": 0},
    }
    return mocked
