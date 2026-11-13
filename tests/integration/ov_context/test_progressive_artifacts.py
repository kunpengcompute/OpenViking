"""Previously delivered files remain downloadable if a later task item fails."""

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.deepagents_ov_context.chat.artifacts import list_artifacts

from .chat_model_support import NoToolModel
from .test_chat_api import parse_events
from .test_office_skills import office_run


def test_progressive_outputs_are_idempotent_and_keep_previous_files(tmp_path: Path):
    run = office_run(tmp_path)
    (run.folder / "output/mail_01.txt").write_text("first body")
    run.publish()
    assert [x["filename"] for x in run.last_published] == ["mail_01.txt"]
    assert (run.folder / "mail_01.txt").read_text() == "first body"
    (run.folder / "output/mail_02.txt").write_text("second body")
    assert len(run.publish()) == 2
    assert [x["filename"] for x in run.last_published] == ["mail_02.txt"]
    assert len(run.publish()) == 2
    assert run.last_published == []
    assert len(list_artifacts(tmp_path, "session-a")) == 2


def test_valid_revision_publishes_a_new_snapshot(tmp_path: Path):
    run = office_run(tmp_path)
    path = run.folder / "output/mail_01.txt"
    path.write_text("draft body")
    run.publish()
    first_hash = run.last_published[0]["sha256"]
    path.write_text("complete body")
    run.publish()
    assert run.last_published[0]["sha256"] != first_hash
    assert (run.folder / "mail_01.txt").read_text() == "complete body"


def test_invalid_later_file_does_not_revoke_a_delivered_snapshot(tmp_path: Path):
    run = office_run(tmp_path)
    (run.folder / "output/mail_01.txt").write_text("first body")
    run.publish()
    (run.folder / "output/mail_01.txt").write_text("changed draft")
    (run.folder / "output/broken.json").write_text("{")
    with pytest.raises(ValueError):
        run.publish()
    assert (run.folder / "mail_01.txt").read_text() == "first body"
    assert len(list_artifacts(tmp_path, "session-a")) == 1


class TwoFilesThenFail(NoToolModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage) and m.name == "write_file"]
        if len(results) >= 2:
            raise RuntimeError("planned failure after two delivered outputs")
        n = len(results) + 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": f"write-{n}",
                                "name": "write_file",
                                "args": {
                                    "file_path": f"/output/mail_0{n}.txt",
                                    "content": f"body {n}",
                                },
                            }
                        ],
                    )
                )
            ]
        )


async def test_chat_publishes_progress_before_later_run_failure(chat, monkeypatch):
    client, runtime = chat
    monkeypatch.setattr(
        "examples.deepagents_ov_context.scripts.check.check_office_skills", lambda: {}
    )
    runtime.settings = replace(runtime.settings, office_skills=True)
    runtime.model = TwoFilesThenFail()
    created = await client.post(
        "/api/sessions",
        json={
            "execution": {
                "mode": "baseline",
                "browser_only": True,
                "recursion_limit": 100,
            }
        },
    )
    sid = created.json()["session_id"]
    response = await client.post(
        "/api/chat", json={"session_id": sid, "message": "Export two files"}
    )
    events = parse_events(response.text)
    assert events[-1][0] == "error"
    publications = [
        data
        for kind, data in events
        if kind == "stage" and data.get("name") == "artifacts.published"
    ]
    assert [a["filename"] for p in publications for a in p["artifacts"]] == [
        "mail_01.txt",
        "mail_02.txt",
    ]
    artifacts = (await client.get(f"/api/sessions/{sid}/artifacts")).json()
    assert len(artifacts) == 2
    for item in artifacts:
        downloaded = await client.get(item["url"])
        assert downloaded.status_code == 200
        assert downloaded.text in {"body 1", "body 2"}
