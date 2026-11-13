# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Automatic acceptance uses immutable task revisions and the Run clock."""

import json

from benchmark.office.suites.common.automatic import RunAudit, evaluate_audit

from .test_ppt_six_run import SPEC, write_deck


async def test_ppt_completion_is_recomputed_after_a_late_repair(tmp_path):
    workspace = tmp_path / "workspace"
    audit = RunAudit(workspace, tmp_path / "audit", "ppt_update", "Q")
    tasks = SPEC["phases"]["Q"]["tasks"]
    deck = workspace / "output/updated.pptx"
    for index in range(6):
        write_deck(deck, tasks[: 2 * (index + 1)])
        audit.capture((index + 1) * 10, "tool.end", tool_call_id=f"edit-{index}")
    # A later tool destroys task 1; the next tool repairs it after its deadline.
    write_deck(deck, tasks[2:])
    audit.capture(80, "tool.end", tool_call_id="broken")
    write_deck(deck, tasks)
    audit.capture(130, "tool.end", tool_call_id="repair")
    audit.capture(150, "run.end")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 150, "status": "done"})
    rows = result["latency"]["subtasks"]
    assert [r["completed_seconds"] for r in rows] == [130, 20, 30, 40, 50, 60]
    assert result["latency"]["success_count"] == 5
    assert rows[-1]["prefix_cumulative_seconds"] == 130
    assert result["snapshots"] == 9


async def test_snapshot_survives_later_file_changes_and_missing_evidence_is_unknown(tmp_path):
    workspace = tmp_path / "workspace"
    audit = RunAudit(workspace, tmp_path / "audit", "ppt_update", "Q")
    deck = workspace / "output/updated.pptx"
    write_deck(deck, SPEC["phases"]["Q"]["tasks"])
    audit.capture(20, "run.end")
    deck.write_bytes(b"corrupted archive")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 30})
    assert result["latency"]["success_count"] == 6
    audit.capture(30, "run.end")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 30})
    assert result["latency"]["success_count"] == 0
    assert result["latency"]["unverified_subtasks"] == 6


async def test_mail_requires_the_exact_body_and_tracks_later_corruption(tmp_path):
    from benchmark.office.datasets.mail.data import goals

    workspace = tmp_path / "workspace"
    audit = RunAudit(workspace, tmp_path / "audit", "mail", "Q")
    output = workspace / "output"
    output.mkdir(parents=True)

    def publish(name, text):
        raw = text.encode()
        (output / name).write_bytes(raw)
        (workspace / name).write_bytes(raw)
        (workspace / (name + ".artifact.json")).write_text(
            json.dumps(
                {
                    "filename": name,
                    "run_id": "r-test",
                    "url": "/artifact/" + name,
                    "bytes": len(raw),
                    "kind": "text",
                }
            )
        )

    for index, goal in enumerate(goals("Q"), 1):
        publish(goal["filename"], goal["message"]["body"])
        audit.capture(index * 10, "tool.end")
    publish(goals("Q")[0]["filename"], "Wrong mail body")
    audit.capture(100, "run.end")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 100})
    assert result["latency"]["success_count"] == 5
    assert result["latency"]["subtasks"][0]["final_passed"] is False


async def test_calendar_needs_matching_reservation_and_detail_read(tmp_path):
    from .test_schedule_resource_deadline import successful_state

    ref, after, first = successful_state(tmp_path / "fixture")
    folder = tmp_path / "audit"
    RunAudit(tmp_path / "workspace", folder, "schedule", "Q", reference={"calendar": ref})
    (folder / "snapshots.jsonl").write_text(
        json.dumps(
            {
                "elapsed_seconds": 30,
                "event": "run.end",
                "files": {},
                "business": after,
            }
        )
        + "\n"
    )
    result = await evaluate_audit(folder, {"seconds": 100})
    assert result["latency"]["success_count"] == 0
    after["audit"] += [{"action": "detail_read", "event_id": first + i} for i in range(5)]
    with (folder / "snapshots.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"elapsed_seconds": 40, "event": "run.end", "files": {}, "business": after})
            + "\n"
        )
    result = await evaluate_audit(folder, {"seconds": 100})
    assert result["latency"]["success_count"] == 5


async def test_timesheet_needs_correct_increment_and_valid_screenshot(tmp_path):
    import io

    from PIL import Image

    from benchmark.office.datasets.timesheet.data import goals

    folder = tmp_path / "audit"
    audit = RunAudit(
        tmp_path / "workspace", folder, "timesheet", "Q", reference={"before": {"entries": []}}
    )
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, format="PNG")
    sha = audit.blob(image.getvalue())
    entries = [{"id": index, **goal} for index, goal in enumerate(goals("Q"), 1)]
    files = {f"output/timesheet_verification_{i:02}.png": sha for i in range(1, 6)}
    (folder / "snapshots.jsonl").write_text(
        json.dumps(
            {
                "elapsed_seconds": 100,
                "event": "run.end",
                "files": files,
                "business": {"entries": entries},
            }
        )
        + "\n"
    )
    result = await evaluate_audit(folder, {"seconds": 200})
    assert result["latency"]["success_count"] == 5


async def test_document_judges_saved_chapters_once_and_retains_scoring_evidence(tmp_path):
    from benchmark.office.suites.reqdoc_generation.scoring import TITLES

    workspace = tmp_path / "workspace"
    audit = RunAudit(workspace, tmp_path / "audit", "reqdoc_generation", "Q")
    output = workspace / "output"
    output.mkdir(parents=True)
    document = output / "requirements.md"
    calls = []

    class Judge:
        cache_key = "test-judge-v1"

        async def __call__(self, task, chapter, criteria, sources, evidence):
            calls.append(task)
            return {
                "checks": [
                    {
                        "id": f"{task}.{i}",
                        "status": "passed",
                        "evidence": "Compared with original source",
                        "quotes": [chapter[:15]],
                        "source_quotes": [sources[:15]],
                    }
                    for i in range(1, len(criteria) + 1)
                ],
                "model": "test-judge",
                "usage": {"total_tokens": 100},
            }

    document.write_text(
        "\n".join(
            f"## {i} {title}\n章节{i}有独立且完整的原始资料依据。\n"
            for i, title in enumerate(TITLES[:4], 1)
        )
    )
    audit.capture(100, "tool.end")
    audit.capture(200, "run.end")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 200}, judge=Judge())
    assert result["latency"]["success_count"] == 4
    assert calls == ["RD01", "RD02", "RD03", "RD04"]
    assert result["judge_usage"]["total_tokens"] == 400
    # Replaying the same immutable evidence does not call the judge again.
    again = await evaluate_audit(tmp_path / "audit", {"seconds": 200}, judge=Judge())
    assert len(calls) == 4 and again["latency"] == result["latency"]


async def test_document_unsupported_positive_judgment_remains_unverified(tmp_path):
    from benchmark.office.suites.reqdoc_generation.scoring import TITLES

    workspace = tmp_path / "workspace"
    audit = RunAudit(workspace, tmp_path / "audit", "reqdoc_generation", "Q")
    (workspace / "output").mkdir(parents=True)
    (workspace / "output/requirements.md").write_text(
        f"## 1 {TITLES[0]}\n这是一段需要原始依据的正文。"
    )
    audit.capture(10, "run.end")

    class BadJudge:
        cache_key = "bad-judge"

        async def __call__(self, task, chapter, criteria, sources, evidence):
            return {
                "checks": [
                    {
                        "id": f"{task}.{i}",
                        "status": "passed",
                        "evidence": "claim",
                        "quotes": ["Invented quote"],
                        "source_quotes": ["Invented source"],
                    }
                    for i in range(1, len(criteria) + 1)
                ]
            }

    result = await evaluate_audit(tmp_path / "audit", {"seconds": 10}, judge=BadJudge())
    assert result["latency"]["success_count"] == 0
    assert result["latency"]["unverified_subtasks"] >= 1


async def test_standard_chat_records_and_grades_a_benchmark_without_alternate_server(chat):
    from .chat_model_support import NoToolModel
    from .test_chat_api import parse_events

    client, runtime = chat
    runtime.model = NoToolModel()
    created = await client.post(
        "/api/sessions",
        json={
            "experiment_id": "office-auto--ppt_update--Q",
            "execution": {"mode": "baseline"},
        },
    )
    sid = created.json()["session_id"]
    response = await client.post("/api/chat", json={"session_id": sid, "message": "只回复 done"})
    events = parse_events(response.text)
    assert events[-1][0] == "done"
    rid = next(data["run_id"] for event, data in events if event == "start")
    result = await client.post(f"/api/sessions/{sid}/runs/{rid}/benchmark")
    assert result.status_code == 200
    assert result.json()["automatic"] and result.json()["latency"]["success_count"] == 0
    assert result.json()["snapshots"] >= 2
    again = await client.post(f"/api/sessions/{sid}/runs/{rid}/benchmark")
    assert again.json() == result.json()
    # An ordinary Chat Run has no benchmark capture or invented completion times.
    plain = await client.post("/api/sessions", json={"execution": {"mode": "baseline"}})
    plain_id = plain.json()["session_id"]
    resp = await client.post("/api/chat", json={"session_id": plain_id, "message": "done"})
    plain_rid = next(d["run_id"] for e, d in parse_events(resp.text) if e == "start")
    assert (
        await client.post(f"/api/sessions/{plain_id}/runs/{plain_rid}/benchmark")
    ).status_code == 404


async def test_document_judge_speaks_the_real_openai_http_protocol(tmp_path, monkeypatch):
    import httpx
    import openai

    from benchmark.office.suites.common.document_judge import DocumentJudge

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "api_key": "test-key",
                "api_base": "https://ark.volces.com/api/v3",
                "chat_model": "test-model",
                "embedding_model": "embedding",
                "embedding_dimension": 8,
            }
        )
    )
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "judge-call",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"checks":[]}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    original = openai.AsyncOpenAI

    def sdk(**kwargs):
        # Replace only the external HTTP transport; retain SDK request/response handling.
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return original(**kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", sdk)
    judge = DocumentJudge(path)
    try:
        result = await judge("RD01", "正文", ["验收条件"], "原始资料", {})
    finally:
        await judge.close()
    assert result["usage"]["total_tokens"] == 120
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["thinking"] == {"type": "disabled"}


async def test_incomplete_capture_cannot_be_scored_as_a_success(tmp_path):
    audit = RunAudit(tmp_path / "workspace", tmp_path / "audit", "ppt_update", "Q")
    write_deck(tmp_path / "workspace/output/updated.pptx", SPEC["phases"]["Q"]["tasks"])
    audit.capture(20, "tool.end")
    result = await evaluate_audit(tmp_path / "audit", {"seconds": 30})
    assert result["latency"]["success_rate"] is None
    assert result["latency"]["unverified_subtasks"] == 6
    assert result["errors"][-1]["type"] == "MissingTerminalSnapshot"
