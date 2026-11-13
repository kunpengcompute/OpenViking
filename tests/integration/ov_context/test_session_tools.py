"""Validate model-facing arguments and recovery through real LangGraph tool execution."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic.v1 import ValidationError as V1ValidationError

from openviking.integrations.langchain.ov_context import OVPluginState, RunIdentity
from openviking.integrations.langchain.ov_context.session_engine import SessionEngine
from openviking.integrations.langchain.ov_context.tool_readback import (
    _validation_error,
    make_session_tools,
)
from openviking_cli.exceptions import NotFoundError

REF = "viking://user/u/sessions/s/tool-results/result-1"
IDENTITY = RunIdentity("a", "u", "p", "s", "r")


def test_validation_handler_accepts_langchain_legacy_validation_errors() -> None:
    class LegacyArguments(V1BaseModel):
        limit: int

    with pytest.raises(V1ValidationError) as caught:
        LegacyArguments(limit="not-a-number")
    message = _validation_error(caught.value)
    assert "limit" in message and "retry" in message


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    async def call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, kwargs))
        if self.error:
            raise self.error
        if method == "search_tool_result":
            return {"tool_result_id": kwargs["tool_result_id"], "matches": []}
        return {"method": method, "limit": kwargs.get("limit")}


@pytest.fixture
def tools() -> tuple[RecordingClient, dict[str, BaseTool]]:
    client = RecordingClient()
    engine = cast(SessionEngine, SimpleNamespace(client=client))
    values = make_session_tools(engine, lambda state, runtime: IDENTITY)
    return client, {value.name.removeprefix("openviking_tool_result_"): value for value in values}


def state() -> OVPluginState:
    return {
        "messages": [HumanMessage(content="Find the air quality link")],
        "ov_context": {"session_id": "s", "owner": IDENTITY.session_key, "run_id": "r"},
    }


async def invoke(
    tool: BaseTool, args: dict[str, Any], initial: OVPluginState | None = None
) -> ToolMessage:
    graph = StateGraph(OVPluginState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    initial = state() if initial is None else initial
    initial["messages"] = [
        AIMessage(content="", tool_calls=[{"name": tool.name, "args": args, "id": "call-1"}])
    ]
    result = await graph.compile().ainvoke(initial)
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    return message


def test_model_schema_exposes_bounds_and_hides_runtime(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
) -> None:
    _, values = tools
    assert set(values) == {"read", "search_many", "list"}
    for name, maximum in (("read", 20_000), ("search_many", 20), ("list", 50)):
        properties = values[name].tool_call_schema.model_json_schema()["properties"]
        assert "runtime" not in properties
        assert properties["limit"]["minimum"] == 1
        assert properties["limit"]["maximum"] == maximum
        assert properties["limit"]["default"] == maximum
    properties = values["read"].tool_call_schema.model_json_schema()["properties"]
    assert properties["offset"]["minimum"] == 0
    queries = values["search_many"].tool_call_schema.model_json_schema()["properties"]["queries"]
    assert queries["type"] == "array"
    assert queries["minItems"] == 1 and queries["maxItems"] == 4
    assert queries["items"]["maxLength"] == 200


def test_argument_schemas_are_reused_with_unchanged_model_contract() -> None:
    engine = cast(SessionEngine, SimpleNamespace(client=RecordingClient()))
    first = make_session_tools(engine, lambda state, runtime: IDENTITY)
    second = make_session_tools(engine, lambda state, runtime: IDENTITY)
    for left, right in zip(first, second, strict=True):
        assert left is not right
        assert left.coroutine is not right.coroutine
        assert left.args_schema is right.args_schema
        inferred = StructuredTool.from_function(coroutine=left.coroutine)
        assert convert_to_openai_tool(left) == convert_to_openai_tool(inferred)


async def test_shared_schemas_keep_run_clients_and_identity_isolated() -> None:
    clients = [RecordingClient(), RecordingClient()]
    identities = [IDENTITY, RunIdentity("b", "v", "q", "t", "next-run")]
    requests = []
    for client, identity in zip(clients, identities, strict=True):
        engine = cast(SessionEngine, SimpleNamespace(client=client))
        values = make_session_tools(engine, lambda state, runtime, i=identity: i)
        initial = state()
        initial["ov_context"] = {
            "session_id": identity.session_id,
            "owner": identity.session_key,
            "run_id": identity.run_id,
        }
        requests.append(invoke(values[-1], {"limit": 1}, initial))
    messages = await asyncio.gather(*requests)
    for client, identity, message in zip(clients, identities, messages, strict=True):
        assert message.status == "success"
        assert client.calls == [
            (
                "list_tool_results",
                {"identity": identity.sdk_identity, "session_id": identity.session_id, "limit": 1},
            )
        ]


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        ("search_many", {"ref": REF, "queries": ["空气质量"], "limit": 30}, "limit"),
        ("search_many", {"ref": REF, "queries": ["空气质量"], "limit": 0}, "limit"),
        ("search_many", {"ref": REF, "queries": ["空气质量"], "limit": True}, "limit"),
        ("search_many", {"ref": REF, "queries": ["空气质量"], "limit": 1.5}, "limit"),
        ("search_many", {"ref": REF, "queries": ["空气质量"], "limit": "20"}, "limit"),
        ("search_many", {"ref": REF, "queries": []}, "queries"),
        ("search_many", {"ref": REF, "queries": ["a"] * 5}, "queries"),
        ("search_many", {"ref": REF, "queries": "空气质量"}, "queries"),
        ("search_many", {"ref": REF, "queries": '["空气质量"]'}, "queries"),
        ("search_many", {"ref": REF, "queries": [""]}, "queries"),
        ("search_many", {"ref": REF, "queries": [" \n "]}, "queries"),
        ("search_many", {"ref": REF, "queries": ["x" * 201]}, "queries"),
        ("search_many", {"ref": REF, "queries": [123]}, "queries"),
        ("read", {"ref": REF, "offset": -1}, "offset"),
        ("read", {"ref": REF, "limit": 0}, "limit"),
        ("read", {"ref": REF, "limit": 20_001}, "limit"),
        ("list", {"limit": 0}, "limit"),
        ("list", {"limit": 51}, "limit"),
    ],
)
async def test_invalid_arguments_return_tool_error_without_sdk_call(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
    name: str,
    args: dict[str, Any],
    field: str,
) -> None:
    client, values = tools
    message = await invoke(values[name], args)
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
    assert field in message.content and "retry" in message.content
    assert client.calls == []


@pytest.mark.parametrize("wrong_owner", [False, True])
async def test_identity_and_reference_guards_remain_enforced(
    tools: tuple[RecordingClient, dict[str, BaseTool]], wrong_owner: bool
) -> None:
    client, values = tools
    initial = state()
    ref = REF
    if wrong_owner:
        initial["ov_context"]["owner"] = ("a", "another-user", "p", "s")
    else:
        ref = REF.replace("/sessions/s/", "/sessions/another-session/")
    with pytest.raises(ValueError, match="different session"):
        await invoke(values["search_many"], {"ref": ref, "queries": ["空气质量", "AQI"]}, initial)
    assert client.calls == []


async def test_sdk_errors_are_not_misreported_as_argument_errors(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
) -> None:
    client, values = tools
    client.error = RuntimeError("OV unavailable")
    with pytest.raises(RuntimeError, match="OV unavailable"):
        await invoke(values["search_many"], {"ref": REF, "queries": ["空气质量"]})
    assert len(client.calls) == 1


@pytest.mark.parametrize("name", ["read", "search_many"])
async def test_missing_tool_result_is_recoverable(
    tools: tuple[RecordingClient, dict[str, BaseTool]], name: str
) -> None:
    client, values = tools
    client.error = NotFoundError("result-1", "tool result")
    args: dict[str, Any] = {"ref": REF}
    if name == "search_many":
        args["queries"] = ["空气质量"]
    message = await invoke(values[name], args)
    assert message.status == "error"
    assert "not found" in message.content
    assert "openviking_tool_result_list" in message.content
    assert client.calls


async def test_previous_run_cannot_start_batch_search(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
) -> None:
    client, values = tools
    initial = state()
    initial["ov_context"]["run_id"] = "previous-run"
    with pytest.raises(ValueError, match="different session/identity/Run"):
        await invoke(values["search_many"], {"ref": REF, "queries": ["AQI", "空气"]}, initial)
    assert client.calls == []


@pytest.mark.parametrize(
    ("name", "limit"),
    [
        ("read", 1),
        ("read", 20_000),
        ("search_many", 1),
        ("search_many", 20),
        ("list", 1),
        ("list", 50),
    ],
)
async def test_valid_boundaries_preserve_sdk_identity_and_arguments(
    tools: tuple[RecordingClient, dict[str, BaseTool]], name: str, limit: int
) -> None:
    client, values = tools
    args: dict[str, Any] = {"limit": limit}
    if name != "list":
        args["ref"] = REF
    if name == "search_many":
        args["queries"] = ["空气质量"]
    message = await invoke(values[name], args)
    assert message.status == "success"
    method, parameters = client.calls[0]
    assert method == f"{name.removesuffix('_many')}_tool_result" + ("s" if name == "list" else "")
    assert parameters["identity"] == IDENTITY.sdk_identity
    assert parameters["session_id"] == "s" and parameters["limit"] == limit
    if name != "list":
        assert parameters["tool_result_id"] == "result-1"
    if name == "search_many":
        assert parameters["query"] == "空气质量"


async def test_graph_can_correct_limit_30_and_finish(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
) -> None:
    client, values = tools
    search = values["search_many"]

    def model(current: OVPluginState) -> dict[str, list[AIMessage]]:
        last = current["messages"][-1]
        if isinstance(last, HumanMessage):
            limit = 30
        elif isinstance(last, ToolMessage) and last.status == "error":
            assert "limit" in last.content and "20" in last.content
            assert client.calls == []
            limit = 20
        else:
            return {"messages": [AIMessage(content="Found the link")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": search.name,
                            "args": {"ref": REF, "queries": ["空气质量"], "limit": limit},
                            "id": f"call-{limit}",
                        }
                    ],
                )
            ]
        }

    graph = StateGraph(OVPluginState)
    graph.add_node("model", model)
    graph.add_node("tools", ToolNode([search]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", lambda current: "tools" if current["messages"][-1].tool_calls else END
    )
    graph.add_edge("tools", "model")
    result = await graph.compile().ainvoke(state())
    assert result["messages"][-1].content == "Found the link"
    assert [m.status for m in result["messages"] if isinstance(m, ToolMessage)] == [
        "error",
        "success",
    ]
    assert len(client.calls) == 1 and client.calls[0][1]["limit"] == 20


async def test_many_tool_reuses_session_identity_and_deduplicates_terms(
    tools: tuple[RecordingClient, dict[str, BaseTool]],
) -> None:
    client, values = tools
    message = await invoke(values["search_many"], {"ref": REF, "queries": [" AQI ", "AQI", "空气"]})
    assert message.status == "success"
    assert [kwargs["query"] for _, kwargs in client.calls] == ["AQI", "空气"]
    assert all(kwargs["identity"] == IDENTITY.sdk_identity for _, kwargs in client.calls)
    assert all(kwargs["session_id"] == "s" for _, kwargs in client.calls)
    assert all(kwargs["tool_result_id"] == "result-1" for _, kwargs in client.calls)
    payload = json.loads(message.content)
    assert [item["query"] for item in payload["query_stats"]] == ["AQI", "空气"]
    assert payload["windows"] == [] and payload["partial"] is False


@pytest.mark.parametrize("name", ["read", "search_many"])
@pytest.mark.parametrize("ref", ["tmp/market_metric.json", "https://example.invalid/result"])
async def test_local_or_web_reference_returns_correctable_error(tools, name, ref):
    client, values = tools
    args = {"ref": ref}
    if name == "search_many":
        args["queries"] = ["excluded_intents", "market_events_excluded"]
    message = await invoke(values[name], args)
    assert message.status == "error"
    assert "ref" in message.content and "openviking_tool_result_list" in message.content
    assert client.calls == []
    corrected = await invoke(values[name], {**args, "ref": REF})
    assert corrected.status == "success"
    assert len(client.calls) == (2 if name == "search_many" else 1)
