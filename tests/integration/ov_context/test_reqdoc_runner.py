# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Current Chat runner gates retain real recall and failed-stream evidence."""

import json

import pytest

from benchmark.office.suites.reqdoc_generation.run import category


def test_strict_ordinary_category():
    assert category("viking://user/u/memories/profile.md") == "profile"
    assert category("viking://user/u/memories/preferences/x.md") == "preferences"
    assert category("viking://user/u/memories/experiences/x.md") == "experiences"
    assert category("viking://user/u/memories/identity.md") == "template_or_unknown"


def test_run_without_model_input_fails_comparison_gate(tmp_path):
    from benchmark.office.suites.reqdoc_generation.run import FirstInputGate

    gate = FirstInputGate("Q", "ordinary", {}, tmp_path)
    with pytest.raises(ValueError, match="before first model input"):
        gate.finish()
    assert json.loads((tmp_path / "first-input-gate.json").read_text())["passed"] is False


def test_first_input_gate_requires_real_body_and_recall(tmp_path):
    from benchmark.office.suites.reqdoc_generation.run import FirstInputGate

    body = "这是与本次任务相关的真实冻结记忆正文，包含足够长的原始内容用于检查实际注入。"
    pool = {"viking://user/u/memories/entities/e.md": {"category": "entities", "content": body}}
    gate = FirstInputGate("Q", "ordinary", pool, tmp_path)
    gate(
        {
            "event": "stage",
            "data": {"name": "memory_recall_result", "status": "success", "count": 1},
        }
    )
    with pytest.raises(ValueError, match="first model input|First model input"):
        gate(
            {
                "event": "model_input",
                "data": {"messages": [{"type": "human", "content": "output/requirements.md RD06"}]},
            }
        )
    gate = FirstInputGate("Q", "ordinary", pool, tmp_path)
    gate(
        {
            "event": "stage",
            "data": {"name": "memory_recall_result", "status": "success", "count": 1},
        }
    )
    gate(
        {
            "event": "model_input",
            "data": {
                "messages": [
                    {
                        "type": "human",
                        "content": "[参考记忆]\n"
                        + body
                        + "\n\n[用户原文]\noutput/requirements.md RD06",
                    }
                ]
            },
        }
    )
    assert json.loads((tmp_path / "first-input-gate.json").read_text())["passed"] is True


@pytest.mark.asyncio
async def test_sse_observer_failure_preserves_journal(tmp_path):
    import httpx

    from benchmark.office.suites.common.chat import chat_events

    response = httpx.Response(
        200,
        text='event: model_input\ndata: {"messages": []}\n\nevent: done\ndata: {}\n\n',
        request=httpx.Request("POST", "http://test/api/chat"),
    )

    def reject(event):
        raise ValueError("reject invalid injection")

    with pytest.raises(ValueError, match="invalid injection"):
        await chat_events(response, tmp_path / "events.json", on_event=reject)
    saved = json.loads((tmp_path / "events.json").read_text())
    assert [e["event"] for e in saved] == ["model_input"]
    assert (tmp_path / "events.jsonl").is_file()


def test_runner_defaults_to_two_retries_and_records_attempt_history(tmp_path):
    from benchmark.office.suites.reqdoc_generation.run import FirstInputGate, build_parser

    args = build_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--chat-url",
            "http://chat",
            "--ov-url",
            "http://ov",
            "--ov-key",
            "key-path",
            "--account",
            "a",
            "--user",
            "u",
        ]
    )
    assert args.recall_retry_attempts == 2
    body = "这是实际冻结普通记忆正文，保留足够完整内容用于核对首个输入中的真实召回。"
    gate = FirstInputGate(
        "Q", "ordinary", {"uri": {"category": "entities", "content": body}}, tmp_path
    )
    for name, fields in [
        ("memory_recall_attempt", {"attempt": 1, "status": "timeout"}),
        ("memory_recall_retry", {"attempt": 1, "next_attempt": 2}),
        ("memory_recall_attempt", {"attempt": 2, "status": "success"}),
        ("memory_recall_result", {"status": "success", "count": 1}),
    ]:
        gate({"event": "stage", "data": {"name": name, **fields}})
    gate(
        {
            "event": "model_input",
            "data": {
                "messages": [
                    {
                        "type": "human",
                        "content": f"[参考记忆]\n{body}\n[用户原文]\noutput/requirements.md RD06",
                    }
                ]
            },
        }
    )
    saved = json.loads((tmp_path / "first-input-gate.json").read_text())
    assert saved["passed"]
    assert [x["status"] for x in saved["recall_attempts"]] == ["timeout", "success"]
    assert saved["recall_retries"][0]["next_attempt"] == 2
