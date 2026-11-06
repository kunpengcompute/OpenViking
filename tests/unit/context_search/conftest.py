# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Keep the context-search batch independent of external services and root fixtures."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def unexpected_connection(*args, **kwargs):
        raise AssertionError("Context-search tests attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    monkeypatch.setattr(socket, "create_connection", unexpected_connection)
