"""Frozen helper H/Q, automatic preparation and a independent 120s task deadlines."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from benchmark.office.suites.common.chat import chat_events
from benchmark.office.suites.ppt_update.gates import FirstInputGate
from benchmark.office.suites.ppt_update.preparation import freeze_automatic_memory
from benchmark.office.suites.ppt_update.scoring import latency_result, stage_view
from benchmark.paths import PROJECT_ROOT

from .test_ppt_six_run import SOURCE, SPEC, write_deck


def test_frozen_assets_have_six_tasks_and_twelve_edits():
    assert SPEC["latency_budget_seconds"] == 120
    for phase in ["H", "Q"]:
        p = SPEC["phases"][phase]
        assert len(p["tasks"]) == 12
        assert len({t["task_id"] for t in p["tasks"]}) == 6
        assert (
            hashlib.sha256((PROJECT_ROOT / p["prompt"]).read_bytes()).hexdigest()
            == p["prompt_sha256"]
        )
    profile = json.loads((PROJECT_ROOT / SPEC["execution_profile"]).read_text())
    assert profile["max_tokens"] == 32768
    assert profile["timeout_seconds"] > SPEC["latency_budget_seconds"]
    assert "120秒" not in (PROJECT_ROOT / SPEC["phases"]["Q"]["prompt"]).read_text()


def test_ppt_deadline_uses_each_task_instead_of_run_completion():
    subtasks = [
        {
            "id": f"PPT-{i:02}",
            "final_passed": True,
            "completed_seconds": t,
            "evidence": f"verified tool call {i}",
        }
        for i, t in enumerate([30, 60, 90, 120, 120.001, 100], 1)
    ]
    value = latency_result(SPEC, "Q", {"status": "done", "seconds": 402.3}, subtasks)
    assert value["run_seconds"] == 402.3
    assert value["success_count"] == 5 and value["total"] == 6
    assert value["subtasks"][-1]["within_budget_passed"]
    assert latency_result(SPEC, "H", {"seconds": 500}, subtasks)["success_count"] is None


def test_one_edit_cannot_credit_a_two_edit_task(tmp_path):
    path = tmp_path / "half.pptx"
    tasks = SPEC["phases"]["Q"]["tasks"]
    write_deck(path, tasks[:1])
    assert stage_view(SOURCE, path, tasks, 14)["applied"] == []


def gate(tmp_path):
    return FirstInputGate(tmp_path, "task", "rules", "Q", "ordinary")


def test_first_input_accepts_recalled_method_without_function_source(tmp_path):
    check = gate(tmp_path)
    memory = "Copy the shared helper, unpack once, use node indexes and verify each edit."
    check(
        {"event": "stage", "data": {"name": "memory_recall_result", "count": 1, "content": memory}}
    )
    check({"event": "model_input", "data": {"messages": ["rules", "task\n" + memory]}})
    assert all(json.loads((tmp_path / "first-input-gate.json").read_text()).values())


def test_recall_event_without_actual_injection_is_rejected(tmp_path):
    check = gate(tmp_path)
    check(
        {
            "event": "stage",
            "data": {"name": "memory_recall_result", "count": 1, "content": "method"},
        }
    )
    with pytest.raises(RuntimeError, match="gate"):
        check({"event": "model_input", "data": {"messages": ["rules", "task"]}})


async def test_h_observes_automatic_commit_without_submitting_a_manual_one(tmp_path):
    body = "Copy the provided helper and use dump to locate text before apply_edits."
    sdk = SimpleNamespace(
        get_task=AsyncMock(return_value={"status": "completed"}),
        read=AsyncMock(return_value=body),
        commit_session=AsyncMock(),
    )
    trace = {
        "events": [
            {
                "event": "stage",
                "data": {"name": "context.commit", "status": "accepted", "task_id": "t"},
            }
        ]
    }
    await freeze_automatic_memory(
        sdk,
        None,
        tmp_path,
        trace,
        {},
        AsyncMock(return_value={"uri": hashlib.sha256(body.encode()).hexdigest()}),
    )
    assert json.loads((tmp_path / "frozen-memory.json").read_text())["automatic_only"]
    sdk.commit_session.assert_not_called()


async def test_missing_automatic_receipt_stops_h_without_manual_fallback(tmp_path):
    sdk = SimpleNamespace(commit_session=AsyncMock())
    with pytest.raises(RuntimeError, match="automatic commit"):
        await freeze_automatic_memory(sdk, None, tmp_path, {"events": []}, {}, AsyncMock())
    sdk.commit_session.assert_not_called()


async def test_sse_keeps_unicode_line_separators_inside_json():
    content = "第一段\u0085第二段\u2028第三段\u2029末段"
    body = (
        "event: done\r\ndata: " + json.dumps({"answer": content}, ensure_ascii=False) + "\r\n\r\n"
    )
    response = httpx.Response(200, request=httpx.Request("GET", "http://localhost"), text=body)
    assert (await chat_events(response))[-1]["data"]["answer"] == content


def test_delivery_filename_can_change_without_losing_task_or_budget_credit(tmp_path):
    from benchmark.office.suites.ppt_update.scoring import score

    from .test_ppt_six_run import result_files

    deck = tmp_path / "deck.pptx"
    write_deck(deck, SPEC["phases"]["Q"]["tasks"])
    result_files(tmp_path, deck)
    (tmp_path / "artifacts/updated.pptx").rename(tmp_path / "artifacts/final.pptx")
    (tmp_path / "answer.md").write_text("/artifact/final.pptx")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "statistics": {"status": "done", "seconds": 120},
                "artifacts": [
                    {"filename": "final.pptx", "status": 200, "url": "/artifact/final.pptx"}
                ],
            }
        )
    )
    value = score(SPEC, "Q", tmp_path, tmp_path / "audit", SOURCE)
    assert value["task_pass"]
    assert value["latency"]["unknown_completion_times"] == 6
    assert value["latency"]["success_count"] == 0
    assert value["final_correct"] == 6 and value["correct_subitems"] == 12


def test_ppt_reviewed_subtask_times_are_independent_of_final_delivery(tmp_path):
    from benchmark.office.suites.ppt_update.scoring import score

    from .test_ppt_six_run import result_files

    deck = tmp_path / "deck.pptx"
    write_deck(deck, SPEC["phases"]["Q"]["tasks"])
    result_files(tmp_path, deck)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    result["statistics"]["seconds"] = 200
    result_path.write_text(json.dumps(result))
    (tmp_path / "answer.md").write_text("Missing download link")
    (tmp_path / "completion-review.json").write_text(
        json.dumps(
            {
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "subtasks": [
                    {"id": f"PPT-{i:02}", "completed_seconds": seconds, "evidence": f"call-{i}"}
                    for i, seconds in enumerate([30, 60, 90, 120, 140, 100], 1)
                ],
            }
        )
    )
    value = score(SPEC, "Q", tmp_path, tmp_path / "audit", SOURCE)
    assert not value["task_pass"] and not value["delivery_pass"]
    assert value["latency"]["success_count"] == 5


async def test_comparison_prepares_h_once_and_alternates_fresh_q_runs(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("benchmark.office.suites.ppt_update.compare")
    calls = []

    async def fake_run(args):
        calls.append((args.phase, args.mode, args.expected_memory))
        args.output.mkdir(parents=True)
        (args.output / "grade.json").write_text(
            json.dumps(
                {"task_pass": True, "latency": {"success_count": 6 if args.phase == "Q" else None}}
            )
        )
        (args.output / "condition.json").write_text(json.dumps({"metadata": {"model": "fixed"}}))
        (args.output / "result.json").write_text(
            json.dumps({"statistics": {"status": "done", "seconds": 100}})
        )
        if args.phase == "H":
            (args.output / "frozen-memory.json").write_text(
                json.dumps({"fingerprint": {"uri": "hash"}})
            )
        elif args.mode == "ordinary":
            assert args.expected_memory.is_file()

    monkeypatch.setattr(module, "run", fake_run)
    args = SimpleNamespace(
        spec=PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json",
        pairs=3,
        output=tmp_path / "result",
    )
    await module.compare(args)
    assert [(p, m) for p, m, _ in calls] == [
        ("H", "current"),
        ("Q", "baseline"),
        ("Q", "ordinary"),
        ("Q", "ordinary"),
        ("Q", "baseline"),
        ("Q", "baseline"),
        ("Q", "ordinary"),
    ]
    assert len(list(args.output.glob("pair-*/Q-*/grade.json"))) == 6
    assert json.loads((args.output / "comparison.json").read_text())["status"] == "completed"


async def test_wrong_reader_profile_is_rejected_before_creating_a_session(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("benchmark.office.suites.ppt_update.run")
    requests = []
    actual_client = httpx.AsyncClient

    def respond(request):
        requests.append(request.method)
        return httpx.Response(
            200,
            json={
                "account_id": "account",
                "ov_url": "http://ov",
                "skills": {"pptx": {}},
                "automatic_benchmark": True,
                "reader_defaults": SPEC["reader_contract"] | {"ranking": "score"},
            },
        )

    def client(**kwargs):
        return actual_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    args = SimpleNamespace(
        spec=PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json",
        phase="Q",
        mode="baseline",
        expected_memory=None,
        output=tmp_path / "run",
        account="account",
        user="alice",
        peer="ppt",
        environment_id="daily",
        chat_url="http://chat",
        ov_url="http://ov",
    )
    with pytest.raises(ValueError, match="Reader defaults"):
        await module.run(args)
    assert requests == ["GET"]
