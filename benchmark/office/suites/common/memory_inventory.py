"""Enumerate deep memory leaves through the public SDK, without a storage mount."""

from __future__ import annotations

from collections import deque
from pathlib import PurePosixPath
from typing import Any, Protocol

from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import NotFoundError


class DirectoryClient(Protocol):
    async def ls(
        self,
        uri: str,
        *,
        recursive: bool = False,
        node_limit: int = 1000,
        identity: OVIdentity | None = None,
    ) -> list[Any]: ...


async def memory_leaves(
    client: DirectoryClient,
    identity: OVIdentity,
    *,
    node_limit: int = 1000,
    max_nodes: int = 20000,
) -> list[dict[str, Any]]:
    """Walk one directory at a time: ls(recursive=True) has a default depth limit.

    Saturated listings fail closed, because an incomplete pool invalidates comparison.
    Missing peer memories are optional; authentication/network errors are not swallowed.
    """
    roots = (
        f"viking://user/{identity.user_id}/memories",
        f"viking://user/{identity.user_id}/peers/{identity.actor_peer_id}/memories",
    )
    queue = deque((root, root, True) for root in roots)
    seen: set[str] = set()
    leaves: dict[str, dict[str, Any]] = {}
    nodes = 0
    while queue:
        uri, root, optional = queue.popleft()
        if uri in seen:
            continue
        seen.add(uri)
        try:
            entries = await client.ls(
                uri, recursive=False, node_limit=node_limit, identity=identity
            )
        except NotFoundError:
            if optional:
                continue
            raise
        if len(entries) >= node_limit:
            raise RuntimeError(f"Directory may be truncated at node_limit={node_limit}: {uri}")
        nodes += len(entries)
        if nodes > max_nodes:
            raise RuntimeError("Memory inventory exceeded max_nodes; pool is incomplete")
        for entry in entries:
            child = str(entry.get("uri", ""))
            if not child.startswith(root + "/"):
                raise ValueError("Directory returned an entry outside its memory root")
            name = PurePosixPath(child).name
            if name.startswith("."):
                continue
            if entry.get("isDir"):
                queue.append((child, root, False))
            elif child.endswith(".md"):
                leaves[child] = {
                    **entry,
                    "uri": child,
                    "rel_path": child[len(root) + 1 :],
                    "root": root,
                }
    return [leaves[uri] for uri in sorted(leaves)]


def memory_scopes(inventory: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordinary = []
    experiences = []
    for entry in inventory:
        rel = entry["rel_path"]
        if rel.startswith("experiences/"):
            experiences.append(entry["uri"])
        elif rel.startswith(("entities/", "events/", "preferences/", "profile/")):
            ordinary.append(entry["uri"])
    if not ordinary or not experiences:
        raise RuntimeError(
            "Preparation requires both ordinary memories and genuine experience files"
        )
    return tuple(sorted(set(ordinary))), tuple(sorted(set(experiences)))
