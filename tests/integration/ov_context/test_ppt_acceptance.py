# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Actual recall must be present before the first model input."""

from benchmark.office.cases.ppt_update.acceptance import memory_in_first_input


def test_recall_success_requires_content_in_first_real_input():
    recall = {
        "event": "stage",
        "data": {"name": "memory_recall_result", "count": 1, "content": "concrete edit method"},
    }
    first = {"event": "model_input", "data": {"messages": [{"content": "user task"}]}}
    later = {"event": "model_input", "data": {"messages": [{"content": "concrete edit method"}]}}
    assert memory_in_first_input({"events": [recall]}) is None
    assert memory_in_first_input({"events": [recall, first, later]}) is False
    first["data"]["messages"][0]["content"] = "concrete edit method\nuser task"
    assert memory_in_first_input({"events": [recall, first]}) is True


def test_late_recall_cannot_certify_first_input():
    first = {"event": "model_input", "data": {"messages": [{"content": "same text"}]}}
    recall = {
        "event": "stage",
        "data": {"name": "memory_recall_result", "count": 1, "content": "same text"},
    }
    assert memory_in_first_input({"events": [first, recall]}) is False
