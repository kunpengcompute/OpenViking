"""Freeze automatic H memories independently of shared helper source retention."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmark.office.suites.ppt_update.preparation import freeze_automatic_memory
from benchmark.paths import PROJECT_ROOT


def prepared_sdk(body):
    return SimpleNamespace(
        get_task=AsyncMock(return_value={"status": "completed"}),
        read=AsyncMock(return_value=body),
        commit_session=AsyncMock(),
    ), {"viking://user/memories/events/helper.md": hashlib.sha256(body.encode()).hexdigest()}


def automatic_trace():
    return {
        "events": [
            {
                "event": "stage",
                "data": {"name": "context.commit", "status": "accepted", "task_id": "t"},
            }
        ]
    }


async def test_method_summary_without_code_is_frozen_and_injected(tmp_path):
    from benchmark.office.suites.ppt_update.gates import FirstInputGate

    memory = "Unpack once, copy the shared helper, dump node indexes, edit and verify."
    sdk, pool = prepared_sdk(memory)
    await freeze_automatic_memory(
        sdk, None, tmp_path, automatic_trace(), {}, AsyncMock(return_value=pool)
    )
    frozen = json.loads((tmp_path / "frozen-memory.json").read_text())
    acceptance = json.loads((tmp_path / "memory-acceptance.json").read_text())
    assert frozen["fingerprint"] == pool
    assert frozen["artifact_acceptance"]["passed"] is True
    assert acceptance["body_hashes_match"] and acceptance["pool_unchanged"]
    sdk.commit_session.assert_not_called()
    gate = FirstInputGate(tmp_path, "Q task", "rules", "Q", "ordinary")
    gate(
        {"event": "stage", "data": {"name": "memory_recall_result", "count": 1, "content": memory}}
    )
    gate({"event": "model_input", "data": {"messages": ["rules", "Q task\n" + memory]}})
    assert all(json.loads((tmp_path / "first-input-gate.json").read_text()).values())


@pytest.mark.parametrize("drift", ["body", "pool", "empty"])
async def test_changed_or_empty_memory_is_not_frozen(tmp_path, drift):
    body = "Inspect node indexes, apply both edits, then verify the result."
    sdk, pool = prepared_sdk(body)
    if drift == "body":
        pool = dict.fromkeys(pool, "0" * 64)
    if drift == "empty":
        pool = {}
    snapshots = [pool, {} if drift == "pool" else pool]
    with pytest.raises(RuntimeError, match="memory artifact"):
        await freeze_automatic_memory(
            sdk, None, tmp_path, automatic_trace(), {}, AsyncMock(side_effect=snapshots)
        )
    assert not (tmp_path / "frozen-memory.json").exists()
    acceptance = json.loads((tmp_path / "memory-acceptance.json").read_text())
    assert acceptance["passed"] is False
    if drift == "body":
        assert acceptance["body_hashes_match"] is False
    if drift == "pool":
        assert acceptance["pool_unchanged"] is False


async def test_already_frozen_artifact_is_not_overwritten(tmp_path):
    frozen = tmp_path / "frozen-memory.json"
    frozen.write_text('{"existing":true}')
    with pytest.raises(FileExistsError, match="already frozen"):
        await freeze_automatic_memory(None, None, tmp_path, automatic_trace(), {}, None)
    assert frozen.read_text() == '{"existing":true}'


@pytest.mark.parametrize(
    "acceptance", [None, {"passed": False}, {"passed": True, "contract_sha256": "other"}]
)
async def test_q_rejects_unaccepted_h_before_creating_a_session(tmp_path, acceptance):
    from benchmark.office.suites.ppt_update.run import run

    saved = tmp_path / "saved.json"
    record = {"fingerprint": {"uri": "hash"}}
    if acceptance is not None:
        record["artifact_acceptance"] = acceptance
    saved.write_text(json.dumps(record))
    args = SimpleNamespace(
        spec=PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json",
        phase="Q",
        mode="ordinary",
        expected_memory=saved,
        output=tmp_path / "Q",
    )
    with pytest.raises(ValueError, match="artifact acceptance"):
        await run(args)
    assert not args.output.exists()


def test_both_phases_reference_same_helper_but_only_h_explains_usage():
    from benchmark.office.suites.ppt_update.gates import CaseContract

    spec = json.loads((PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json").read_text())
    item = spec["support_files"][0]
    path = PROJECT_ROOT / item["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    for phase in ["H", "Q"]:
        text = (PROJECT_ROOT / spec["phases"][phase]["prompt"]).read_text()
        assert "inputs/" + item["filename"] in text
        assert "def apply_edits" not in text
    q = (PROJECT_ROOT / spec["phases"]["Q"]["prompt"]).read_text()
    assert "run_index" not in q and "tmp/unpacked" not in q
    assert CaseContract.from_case(spec).sha256


def test_shared_helper_applies_edits_checks_old_text_and_keeps_backup(tmp_path):
    import runpy

    path = PROJECT_ROOT / "benchmark/office/datasets/ppt_update/edit_slide.txt"
    local = tmp_path / "edit_slide.py"
    local.write_bytes(path.read_bytes())
    slide = tmp_path / "unpacked/ppt/slides/slide3.xml"
    slide.parent.mkdir(parents=True)
    original = "<a:t>周帐单</a:t><a:t>原获取方式</a:t>"
    slide.write_text(original)
    helper = runpy.run_path(str(local))
    helper["apply_edits"]("slide3", [(0, "周", "日"), (1, "原", "新")])
    assert slide.read_text() == "<a:t>日帐单</a:t><a:t>新获取方式</a:t>"
    assert (tmp_path / "backup/slide3.xml").read_text() == original
    with pytest.raises(SystemExit, match="does not contain"):
        helper["apply_edits"]("slide3", [(0, "周", "月")])


def test_case_rejects_helper_names_not_accepted_by_chat_before_run():
    from benchmark.office.suites.ppt_update.gates import CaseContract

    spec = json.loads((PROJECT_ROOT / "benchmark/office/cases/ppt_update/spec.json").read_text())
    spec["support_files"][0]["filename"] = "edit_slide.py.txt"
    with pytest.raises(ValueError, match="filename"):
        CaseContract.from_case(spec)
