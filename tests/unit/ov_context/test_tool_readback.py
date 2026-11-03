"""Readback tools resolve invocation identity and reject cross-session references."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from openviking.integrations.langchain.ov_context import OVContext, RunIdentity
from openviking.integrations.langchain.ov_context.tool_readback import make_session_tools


async def test_readback_passes_runtime_identity_and_checks_cache_owner_before_io():
    identity = RunIdentity("a", "u", "p", "s", "r")
    context = OVContext(
        "s", owner=identity.session_key, messages=[HumanMessage(id="u", content="task")]
    )
    context.begin_run("r", context.messages[0])
    runtime = SimpleNamespace(state={"ov_context": context.to_state()}, context=identity)
    access = SimpleNamespace(
        client=SimpleNamespace(call=AsyncMock(return_value={"content": "raw"}))
    )
    read, _, listing = make_session_tools(access, lambda state, rt: rt.context)
    ref = "viking://user/u/sessions/s/tool-results/result"
    assert await read.coroutine(ref, runtime) == {"content": "raw"}
    access.client.call.assert_awaited_once_with(
        "read_tool_result",
        identity=identity.sdk_identity,
        session_id="s",
        tool_result_id="result",
        offset=0,
        limit=20000,
    )
    await listing.coroutine(runtime)
    before = access.client.call.await_count
    runtime.context = replace(identity, user_id="other")
    with pytest.raises(ValueError, match="different"):
        await read.coroutine(ref, runtime)
    assert access.client.call.await_count == before
    runtime.context = identity
    with pytest.raises(ValueError, match="different session"):
        await read.coroutine(ref.replace("/sessions/s/", "/sessions/other/"), runtime)
    assert access.client.call.await_count == before


def test_readback_schema_enforces_character_and_query_limits():
    access = SimpleNamespace(client=SimpleNamespace(call=AsyncMock()))
    read, search, _ = make_session_tools(access, lambda state, runtime: runtime.context)
    with pytest.raises(ValidationError):
        read.args_schema.model_validate({"ref": "ref", "offset": -1})
    with pytest.raises(ValidationError):
        search.args_schema.model_validate({"ref": "ref", "queries": ["x"], "limit": 30})
