"""Memory inventory scope and traversal contracts for office cases and ST."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from benchmark.office.suites.common.memory_inventory import memory_leaves, memory_scopes
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError

IDENTITY = OVIdentity("acct", "alice", "mail")
ROOT = "viking://user/alice/memories"


class Directory:
    def __init__(self, entries: dict[str, list[dict[str, Any]]]) -> None:
        self.entries = entries
        self.calls: list[str] = []

    async def ls(self, uri: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["identity"] == IDENTITY
        assert kwargs["recursive"] is False
        self.calls.append(uri)
        if uri not in self.entries:
            raise NotFoundError(uri)
        return self.entries[uri]


def entry(uri: str, directory: bool = False) -> dict[str, Any]:
    return {"uri": uri, "isDir": directory}


async def test_inventory_walks_beyond_default_depth_and_selects_only_real_leaves() -> None:
    directories = {
        ROOT: [
            entry(ROOT + "/events", True),
            entry(ROOT + "/experiences", True),
            entry(ROOT + "/.overview.md"),
        ]
    }
    current = ROOT + "/events"
    for name in ["2026", "10", "16"]:
        child = current + "/" + name
        directories[current] = [entry(child, True)]
        current = child
    directories[current] = [entry(current + "/公告.md")]
    directories[ROOT + "/experiences"] = [entry(ROOT + "/experiences/方法.md")]
    sdk = Directory(directories)
    inventory = await memory_leaves(sdk, IDENTITY)
    ordinary, experience = memory_scopes(inventory)
    assert ordinary == (current + "/公告.md",)
    assert experience == (ROOT + "/experiences/方法.md",)
    assert len(sdk.calls) == 7  # includes the absent optional peer root


async def test_inventory_does_not_swallow_permission_errors() -> None:
    sdk = SimpleNamespace(ls=AsyncMock(side_effect=PermissionDeniedError("denied")))
    with pytest.raises(PermissionDeniedError):
        await memory_leaves(sdk, IDENTITY)


@pytest.mark.parametrize("outside", [False, True])
async def test_inventory_fails_closed_for_truncation_or_foreign_entry(outside: bool) -> None:
    sdk = Directory(
        {ROOT: [entry("viking://user/bob/memories/x.md" if outside else ROOT + "/x.md")]}
    )
    with pytest.raises((ValueError, RuntimeError)):
        await memory_leaves(sdk, IDENTITY, node_limit=100 if outside else 1)
