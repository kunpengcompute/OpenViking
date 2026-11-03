"""Read persisted archive markers and tails through native Session snapshots."""

import json

import pytest

from openviking.message import Message, TextPart
from tests.ov_context_support.writer import make_session


async def archive(session, index, state=None, summary=None):
    uri = f"{session.uri}/history/archive_{index:03d}"
    message = Message(id=f"archived-{index}", role="user", parts=[TextPart(f"task {index}")])
    await session._viking_fs.write_file(f"{uri}/messages.jsonl", message.to_jsonl())
    if state is not None:
        await session._viking_fs.write_file(f"{uri}/{state}", "{}")
    if summary is not None:
        await session._viking_fs.write_file(f"{uri}/.overview.md", summary)
    return uri


async def test_pending_and_failed_archives_keep_raw_tail_until_summary_covers_them(tmp_path):
    session = make_session(tmp_path)
    await archive(session, 1, ".done", "summary one")
    await archive(session, 2, ".failed.json")
    await archive(session, 3)
    session._meta.commit_count = 3
    snapshot = await session.get_session_context()
    assert snapshot["latest_finished_archive_id"] == "archive_002"
    assert snapshot["latest_archive_overview"] == "summary one"
    assert [m["id"] for m in snapshot["messages"]] == ["archived-2", "archived-3"]
    assert await session.get_latest_finished_archive_id() == "archive_002"
    uri = await archive(session, 3, ".done", "summary three")
    snapshot = await session.get_session_context()
    assert snapshot["latest_finished_archive_id"] == "archive_003"
    assert snapshot["latest_archive_overview"] == "summary three"
    assert snapshot["messages"] == []
    await session._viking_fs.write_file(f"{uri}/.failed.json", "{}")
    assert await session._get_archive_state(uri) == "done"


async def test_boundary_and_overview_use_one_captured_archive_state(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    await archive(session, 1, ".done", "first")
    uri = await archive(session, 2)
    original = session._get_archive_states

    async def finishing_after_capture():
        states = await original()
        await session._viking_fs.write_file(f"{uri}/.overview.md", "second")
        await session._viking_fs.write_file(f"{uri}/.done", "{}")
        return states

    monkeypatch.setattr(session, "_get_archive_states", finishing_after_capture)
    snapshot = await session.get_session_context()
    assert snapshot["latest_finished_archive_id"] == "archive_001"
    assert snapshot["latest_archive_overview"] == "first"
    assert [m["id"] for m in snapshot["messages"]] == ["archived-2"]
    monkeypatch.setattr(session, "_get_archive_states", original)
    assert (await session.get_session_context())["latest_finished_archive_id"] == "archive_002"


async def test_corrupt_persisted_tail_is_reported_instead_of_silently_dropped(tmp_path):
    session = make_session(tmp_path)
    uri = await archive(session, 1)
    await session._viking_fs.write_file(f"{uri}/messages.jsonl", "not json")
    with pytest.raises(json.JSONDecodeError):
        await session.get_session_context()
