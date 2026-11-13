# SPDX-License-Identifier: AGPL-3.0

import pytest

from openviking.core.directories import PRESET_DIRECTORIES, DirectoryInitializer
from openviking.core.namespace import canonical_user_root
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


class _FakeVikingDB:
    def __init__(self):
        self.embedding_messages = []

    async def get_context_by_uri(self, **_kwargs):
        return []

    async def enqueue_embedding_msg(self, message):
        self.embedding_messages.append(message)


class _FakeVikingFS:
    def __init__(self):
        self.contexts = {}

    async def abstract(self, uri, ctx):
        if uri not in self.contexts:
            raise FileNotFoundError(uri)
        return self.contexts[uri]["abstract"]

    async def write_context(self, uri, abstract, overview, is_leaf, ctx):
        self.contexts[uri] = {
            "abstract": abstract,
            "overview": overview,
            "is_leaf": is_leaf,
        }


@pytest.mark.asyncio
async def test_initialize_user_directories_creates_root_and_first_level_only():
    vikingdb = _FakeVikingDB()
    viking_fs = _FakeVikingFS()
    initializer = DirectoryInitializer(vikingdb, viking_fs=viking_fs)
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ADMIN)

    count = await initializer.initialize_user_directories(ctx)

    user_root = canonical_user_root(ctx)
    expected_uris = {
        user_root,
        *(f"{user_root}/{child.path}" for child in PRESET_DIRECTORIES["user"].children),
    }
    assert count == len(expected_uris)
    assert set(viking_fs.contexts) == expected_uris
    assert f"{user_root}/memories/preferences" not in viking_fs.contexts

    second_count = await initializer.initialize_user_directories(ctx)

    assert second_count == 0
    assert set(viking_fs.contexts) == expected_uris


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [Role.USER, Role.ADMIN, Role.ROOT])
async def test_actor_session_bootstrap_keeps_user_scope_and_request_view(role: Role) -> None:
    from openviking.storage.viking_fs import VikingFS
    from openviking_cli.exceptions import PermissionDeniedError

    guard = VikingFS.__new__(VikingFS)
    writes: list[RequestContext] = []

    class GuardedFS(_FakeVikingFS):
        async def write_context(self, uri, abstract, overview, is_leaf, ctx):
            guard._ensure_mutable_access(uri, ctx)
            writes.append(ctx)
            await super().write_context(uri, abstract, overview, is_leaf, ctx)

    fs, db = GuardedFS(), _FakeVikingDB()
    ctx = RequestContext(
        user=UserIdentifier("acme", "alice"),
        role=role,
        actor_peer_id="mail-agent",
        from_oauth=True,
    )
    initializer = DirectoryInitializer(db, viking_fs=fs)
    count = await initializer.initialize_user_directories(ctx)
    assert count == 1 + len(PRESET_DIRECTORIES["user"].children)
    assert all(uri.startswith("viking://user/alice") for uri in fs.contexts)
    assert all(c.user == ctx.user and c.role == role and c.from_oauth for c in writes)
    assert all(c.actor_peer_id is None for c in writes)
    assert ctx.actor_peer_id == "mail-agent"
    with pytest.raises(PermissionDeniedError):
        guard._ensure_mutable_access("viking://user/alice/peers/other/resources", ctx)
    assert await initializer.initialize_user_directories(ctx) == 0
