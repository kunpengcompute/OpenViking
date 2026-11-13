"""Read-only benchmark sessions may already exist before Chat starts a Run."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.deepagents_ov_context.chat.server import ensure_extraction_disabled_session
from openviking_cli.client.identity import OVIdentity
from openviking_cli.exceptions import AlreadyExistsError


async def test_existing_disabled_policy_is_preserved():
    identity = OVIdentity("a", "u", "p")
    policy = {"self": {"enabled": False}, "peer": {"enabled": False}, "memory_types": ["events"]}
    sdk = SimpleNamespace(
        create_session=AsyncMock(side_effect=AlreadyExistsError("exists")),
        get_session=AsyncMock(return_value={"session_id": "s", "memory_policy": policy}),
    )
    await ensure_extraction_disabled_session(sdk, "s", identity)
    sdk.get_session.assert_awaited_once_with("s", identity=identity, auto_create=False)
    assert policy["memory_types"] == ["events"]


async def test_missing_session_is_created_with_extraction_disabled():
    identity = OVIdentity("a", "u", "p")
    sdk = SimpleNamespace(create_session=AsyncMock(), get_session=AsyncMock())
    await ensure_extraction_disabled_session(sdk, "s", identity)
    sdk.create_session.assert_awaited_once_with(
        "s",
        identity=identity,
        memory_policy={"self": {"enabled": False}, "peer": {"enabled": False}},
    )
    sdk.get_session.assert_not_awaited()


@pytest.mark.parametrize("policy", [{}, {"self": {"enabled": True}, "peer": {"enabled": False}}])
async def test_existing_enabled_or_unknown_policy_is_not_silently_reused(policy):
    sdk = SimpleNamespace(
        create_session=AsyncMock(side_effect=AlreadyExistsError("exists")),
        get_session=AsyncMock(return_value={"session_id": "s", "memory_policy": policy}),
    )
    with pytest.raises(ValueError, match="extraction disabled"):
        await ensure_extraction_disabled_session(sdk, "s", OVIdentity("a", "u", "p"))


async def test_unrelated_create_error_propagates():
    sdk = SimpleNamespace(create_session=AsyncMock(side_effect=TimeoutError("unreachable")))
    with pytest.raises(TimeoutError):
        await ensure_extraction_disabled_session(sdk, "s", OVIdentity("a", "u", "p"))
