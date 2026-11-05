"""Validate only outgoing OV fields, while preserving local native messages."""

from copy import deepcopy

import pytest
from langchain_core.messages import HumanMessage

from openviking.integrations.langchain.ov_context.writer import OVMessageWriter, WriterConfig
from tests.ov_context_support.message_examples import exchange
from tests.ov_context_support.writer import SessionBatchClient, make_session


@pytest.mark.parametrize("batch_size", [2, 100])
async def test_multiple_batches_of_parallel_tools_match_local_positions(
    tmp_path, batch_size: int
) -> None:
    session = make_session(tmp_path)
    client = SessionBatchClient(session)
    raw = [HumanMessage(content=str(i)) for i in range(100)]
    for group in range(3):
        messages = exchange(["small", f"large-{group} " * 1000])
        for message in messages:
            message.id = f"{group}-{message.id}"
        for call in messages[1].tool_calls:
            call["id"] = f"{group}-{call['id']}"
        for message in messages[2:]:
            message.tool_call_id = f"{group}-{message.tool_call_id}"
        raw.extend(messages)
    before = deepcopy(raw)
    result = await OVMessageWriter(client, WriterConfig(batch_size=batch_size)).write(
        session.session_id, raw, 0
    )
    batch_count = len(client.requests)
    assert sum(len(batch) for batch in client.requests) == 109
    assert max(len(batch) for batch in client.requests) <= batch_size
    if batch_size == 100:
        assert [len(batch) for batch in client.requests] == [100, 9]
    assert (result.write_index, result.written_count) == (112, 112)
    for index, (returned, original) in enumerate(zip(result.model_messages, raw, strict=True)):
        if index in (103, 107, 111):
            assert returned is not original and returned.content != original.content
            assert returned.model_dump(exclude={"content"}) == original.model_dump(
                exclude={"content"}
            )
        else:
            assert returned is original
    assert raw == before
    repeated = await OVMessageWriter(client).write(session.session_id, raw, result.write_index)
    assert repeated.model_messages == [] and len(client.requests) == batch_count
