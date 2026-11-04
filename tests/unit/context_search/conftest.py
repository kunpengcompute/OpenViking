"""Keep the context-search batch independent of external services and root fixtures."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def unexpected_connection(*args, **kwargs):
        raise AssertionError("Context-search tests attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    monkeypatch.setattr(socket, "create_connection", unexpected_connection)
