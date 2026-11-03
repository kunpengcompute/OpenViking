import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage

from openviking.integrations.langchain.ov_context.tool_result_search import search_many
from openviking.integrations.langchain.ov_context.tool_views import apply_tool_views
from openviking.integrations.langchain.ov_context.writer import OVMessageWriter
from openviking.message import TextPart, ToolPart
from openviking.session.tool_views import add_messages_with_tool_views
from openviking_cli.exceptions import FailedPreconditionError
from tests.ov_context_support.message_examples import exchange
from tests.ov_context_support.writer import make_session


def tool_part(output, call_id="call", name="lookup", **kwargs):
    return ToolPart(
        tool_id=call_id, tool_name=name, tool_output=output, tool_status="completed", **kwargs
    )


async def test_receipt_uses_input_positions_after_native_group_split(tmp_path):
    session = make_session(tmp_path)
    small, large = tool_part("small", "a"), tool_part("data " * 1200, "b")
    result = add_messages_with_tool_views(
        session,
        [
            {"role": "user", "parts": [TextPart(text="hello")]},
            {"role": "user", "parts": [small, large]},
        ],
    )
    assert (result["added"], result["message_count"]) == (3, 3)
    assert len(result["tool_views"]) == 1
    view = result["tool_views"][0]
    assert (view["message_index"], view["part_index"], view["tool_id"]) == (1, 1, "b")
    assert view["tool_output"] == large.tool_output
    original = await session.read_tool_result(view["tool_output_ref"].rsplit("/", 1)[-1], limit=-1)
    assert original["content"] == "data " * 1200


@pytest.mark.parametrize("enabled", [True, False])
def test_no_compression_returns_empty_views(tmp_path, enabled):
    session = make_session(tmp_path, enabled=enabled)
    result = add_messages_with_tool_views(
        session,
        [
            {"role": "assistant", "parts": [ToolPart(tool_id="a", tool_status="pending")]},
            {"role": "user", "parts": [tool_part("small")]},
        ],
    )
    assert result["tool_views"] == []


@pytest.mark.parametrize("count,compressed", [(2000, False), (2001, True)])
def test_threshold_configuration_boundary(tmp_path, count, compressed):
    session = make_session(tmp_path)
    result = add_messages_with_tool_views(
        session,
        [
            {"role": "user", "parts": [tool_part("x" * count)]},
        ],
    )
    assert bool(result["tool_views"]) is compressed


def test_group_budget_compresses_even_below_single_threshold(tmp_path):
    session = make_session(tmp_path, threshold_chars=20000, assistant_turn_inline_budget_chars=6000)
    output = json.dumps({"rows": [{"id": i, "value": "sample-value"} for i in range(150)]})
    result = add_messages_with_tool_views(
        session,
        [
            {
                "role": "user",
                "parts": [tool_part(output, "a"), tool_part(output, "b")],
            },
        ],
    )
    assert result["tool_views"]
    assert any(
        p.tool_output_externalized_reason == "turn_budget"
        for m in session.messages
        for p in m.parts
    )


@pytest.mark.parametrize("failure_mode", ["preserve_raw", "preview_only", "reject"])
def test_externalization_failure_never_publishes_invalid_reference(tmp_path, failure_mode):
    session = make_session(tmp_path, failure_mode=failure_mode)
    session._viking_fs.fail_tool_store = True
    raw = "x" * 5000
    part = tool_part(raw)
    if failure_mode == "reject":
        with pytest.raises(FailedPreconditionError):
            add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
        assert session.messages == []
    else:
        result = add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
        assert result["tool_views"] == [] and part.tool_output_ref == ""
        if failure_mode == "preserve_raw":
            assert part.tool_output == raw


@pytest.mark.parametrize(
    "content,mime",
    [
        (
            json.dumps({"items": [{"id": i, "value": "value"} for i in range(500)]}),
            "application/json",
        ),
        ("id,name\n" + "1,Alice\n" * 1000, "text/csv"),
        ("第一部分：订单。\n" * 1000, "text/plain"),
    ],
    ids=["json", "csv", "text"],
)
async def test_native_rule_summaries_retain_exact_original(tmp_path, content, mime):
    session = make_session(tmp_path)
    part = tool_part(content, tool_output_mime_type=mime)
    result = add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
    assert len(result["tool_views"]) == 1
    assert len(part.tool_output) < len(content)
    result_id = part.tool_output_ref.rsplit("/", 1)[-1]
    stored = await session.read_tool_result(result_id, limit=-1)
    assert stored["content"] == content


async def test_read_search_and_list_use_externalized_original(tmp_path):
    session = make_session(tmp_path)
    raw = "start\n" + "订单状态：已发货\n" * 600 + "查证标记-42\nend"
    part = tool_part(raw)
    add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
    result_id = part.tool_output_ref.rsplit("/", 1)[-1]
    page = await session.read_tool_result(result_id, offset=7, limit=31)
    assert page["content"] == raw[7:38]
    matches = await session.search_tool_result(result_id, query="查证标记-42", limit=2)
    assert "查证标记-42" in json.dumps(matches, ensure_ascii=False)
    listed = await session.list_tool_results(limit=5)
    assert len(listed["tool_results"]) == 1
    assert listed["tool_results"][0]["tool_name"] == "lookup"
    # Session JSONL and tool-result original are actual temporary files.
    assert session._viking_fs.path(session.uri + "/messages.jsonl").is_file()


async def test_batch_windows_match_native_session_search_contract(tmp_path):
    session = make_session(tmp_path)
    raw = "🙂北京空气质量\n" * 600 + "查证标记-42\nend"
    part = tool_part(raw)
    add_messages_with_tool_views(session, [{"role": "user", "parts": [part]}])
    result_id = part.tool_output_ref.rsplit("/", 1)[-1]

    async def native_search(query, limit):
        return await session.search_tool_result(result_id, query=query, limit=limit)

    result = await search_many(native_search, ["北京", "查证标记-42"])
    assert result["returned_source_chars"] <= 6000
    assert any("查证标记-42" in window["snippet"] for window in result["windows"])
    for window in result["windows"]:
        assert window["snippet"] == raw[window["start_offset"] : window["end_offset"]]


async def test_source_read_reuses_original_reference_and_keeps_arguments(tmp_path):
    session = make_session(tmp_path)
    first = tool_part("original " * 1000)
    add_messages_with_tool_views(session, [{"role": "user", "parts": [first]}])
    ref = first.tool_output_ref
    read = tool_part(
        "evidence " * 500,
        "read-call",
        "openviking_tool_result_read",
        tool_input={
            "ref": ref,
            "offset": 10,
            "limit": 500,
        },
    )
    receipt = add_messages_with_tool_views(session, [{"role": "user", "parts": [read]}])
    assert receipt["tool_views"][0]["tool_output_ref"] == ref
    assert read.tool_output_source_offset == 10 and read.tool_output_source_limit == 500
    assert len((await session.list_tool_results())["tool_results"]) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"tool_id": "wrong"},
        {"message_index": 99},
        {"part_index": -1},
        {"message_index": True},
        {"tool_output_ref": ""},
        {"tool_output_ref": "viking://user/u/sessions/other/tool-results/r"},
        {"tool_output": None},
    ],
)
async def test_invalid_optional_view_does_not_damage_confirmed_write(change):
    raw = exchange(["original"])
    before = deepcopy(raw)
    view = {
        "message_index": 2,
        "part_index": 0,
        "tool_id": "call_0",
        "tool_output": "short",
        "tool_output_ref": "viking://user/u/sessions/s/tool-results/r",
        **change,
    }
    client = AsyncMock()
    client.batch_add_messages.return_value = {
        "session_id": "s",
        "message_count": 3,
        "added": 3,
        "tool_views": [view],
    }
    result = await OVMessageWriter(client).write("s", raw, 0)
    assert result.write_index == 3
    assert result.model_messages[-1] is raw[-1] and raw == before


def test_reversed_receipt_order_matches_by_position_and_call_id():
    messages = [
        ToolMessage(content="A", tool_call_id="a"),
        ToolMessage(content="B", tool_call_id="b"),
    ]
    views = [
        {
            "message_index": 0,
            "part_index": i,
            "tool_id": tool_id,
            "tool_output": output,
            "tool_output_ref": "viking://user/u/sessions/s/tool-results/r",
        }
        for i, tool_id, output in [(1, "b", "short B"), (0, "a", "short A")]
    ]
    apply_tool_views(
        messages,
        {"tool_views": views},
        {(0, 0): 0, (0, 1): 1},
        session_id="s",
        protected_indices=set(),
    )
    assert [m.content for m in messages] == ["short A", "short B"]
