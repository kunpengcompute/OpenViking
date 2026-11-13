"""Default ordinary recall follows the request's user and current peer."""

import asyncio

import pytest

from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking_cli.client.identity import OVIdentity


@pytest.mark.parametrize("search_mode", ["find", "context"])
async def test_default_scope_includes_only_user_and_request_peer(client, search_mode):
    identity = OVIdentity("account-a", "user-a", "doc-agent")
    reader = OVContextReader(client, MemoryReaderConfig(search_mode=search_mode))
    await reader.recall("编写需求文档", identity=identity)
    expected = ["viking://user/memories", "viking://user/peers/doc-agent/memories"]
    client.find.assert_awaited_once()
    actual = client.find.await_args.kwargs["target_uri"]
    assert all(call.kwargs["identity"] == identity for call in client.find.await_args_list)
    assert actual == expected
    assert reader.config.target_uris is None


@pytest.mark.parametrize("search_mode", ["find", "context"])
async def test_explicit_user_only_scope_is_authoritative(client, search_mode):
    config = MemoryReaderConfig(search_mode=search_mode, target_uris=("viking://user/memories",))
    await OVContextReader(client, config).recall(
        "编写需求文档", identity=OVIdentity("account-a", "user-a", "doc-agent")
    )
    assert client.find.await_args.kwargs["target_uri"] == "viking://user/memories"


async def test_shared_reader_does_not_retain_another_users_peer(client):
    reader = OVContextReader(client)
    identities = [
        OVIdentity("account-a", "user-a", "mail"),
        OVIdentity("account-b", "user-b", "doc"),
    ]
    await asyncio.gather(*(reader.recall("继续任务", identity=value) for value in identities))
    for identity in identities:
        calls = [
            call for call in client.find.await_args_list if call.kwargs["identity"] == identity
        ]
        assert len(calls) == 1
        roots = set(calls[0].kwargs["target_uri"])
        assert roots == {
            "viking://user/memories",
            f"viking://user/peers/{identity.actor_peer_id}/memories",
        }
    assert reader.config.target_uris is None


async def test_quotas_keep_category_owned_default_roots(client):
    config = MemoryReaderConfig(recall_quotas={"events": 2, "skills": 1})
    await OVContextReader(client, config).recall(
        "继续任务", identity=OVIdentity("account-a", "user-a", "doc-agent")
    )
    assert client.find.await_count == 2
    roots = {root for call in client.find.await_args_list for root in call.kwargs["target_uri"]}
    assert "viking://user/user-a/memories/events" in roots
    assert "viking://user/user-a/peers/doc-agent/memories/events" in roots
    assert config.target_uris is None
