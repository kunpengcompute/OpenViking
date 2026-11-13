# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Frozen document rubric: six independent chapters and 27 evidence checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from benchmark.office.suites.common.timing import reviewed_timings, score_subtasks

RUBRIC_VERSION = "reqdoc-independent-chapters-v11"

CHECKPOINTS = {
    "RD01": [
        "流程痛点目标及下载、添加两个入口有依据",
        "当前user内按大小写敏感包标识判断新建或更新",
        "新建前置、可编辑与只读字段、名称默认值完整",
        "名称和包校验、初始版本、默认安装及保存后资产状态准确",
    ],
    "RD02": [
        "独立说明已有资产识别、更新字段和默认值",
        "版本格式、数字比较及默认安装选择准确",
        "旧市场版本保留与原工作区版本处理正确",
        "至少一处更新相关新旧冲突有最终裁决和来源",
    ],
    "RD03": [
        "四类角色职责及支持、排除范围有依据",
        "市场先保存后安装及本期审批边界准确",
        "仅保存状态、用户提示、市场资产及工作区版本准确",
        "安装成功状态、用户提示、市场资产及工作区版本准确",
        "待审批状态、用户提示及市场和原工作区版本保留准确",
    ],
    "RD04": [
        "市场与安装各类错误分阶段，保留内容与恢复动作明确",
        "提交前取消及提交中按钮防重规则清楚",
        "保存、更新和安装的同key同载荷重放结果、数量及安装效果准确",
        "资料未定义行为列待确认，规则有来源",
    ],
    "RD05": [
        "独立从原工作簿计算M01为24/30=80%",
        "统计期、时区、工作表隐藏状态及目标来源完整",
        "JSON保留有效排除意图、相关市场事件、原字段原行号及排除原因",
        "用一条市场失败记录说明统计影响，区分基准与目标",
        "独立脚本JSON与第5章一致，可仅凭原资料复算",
    ],
    "RD06": [
        "独立从原工作簿计算M02为12/16=75%",
        "按MET00处理筛选、重复导出、跨期及状态修订",
        "以retry_of区分首次重试、排除非终态，并注明指标目标来源",
        "JSON保留原字段及原行号，包括retry_of、workspace_kind及排除原因",
        "独立脚本JSON与第6章一致，集合归属和覆盖可复算",
    ],
}

TITLES = [
    "需求概述、入口与新建规则",
    "版本更新与冲突处理",
    "市场保存、安装与审批状态",
    "失败恢复、取消与幂等",
    "市场添加完成率分析",
    "首次终态安装成功率分析",
]

LAYOUT = {f"RD{number:02}": (title, (number,), None) for number, title in enumerate(TITLES, 1)}

CHECK_CHAPTERS = {}


def rubric_for_spec(spec: dict) -> str:
    version = spec.get("rubric_version", RUBRIC_VERSION)
    if version != RUBRIC_VERSION:
        raise ValueError(f"Unknown scoring rubric: {version}")
    return version


def _task_scope(sections, task, chapter_filter=None):
    title, chapters, subsection = LAYOUT[task]
    if chapter_filter is not None:
        chapters = chapter_filter
    bodies = []
    for number in chapters:
        values = sections.get(number, [])
        if len(values) != 1 or values[0][0] != TITLES[number - 1] or not values[0][1].strip():
            return False, "", title, chapters, subsection
        bodies.append(values[0][1])
    body = "\n".join(bodies)
    if subsection:
        match = re.search(
            rf"^###\s+{re.escape(subsection)}\s+{re.escape(title)}\n(.*?)(?=^###\s|\Z)",
            body,
            re.M | re.S,
        )
        body = match[1] if match else ""
    return bool(body.strip()), body, title, chapters, subsection


def grade(
    folder: Path, reviews_path: Path | None = None, *, rubric_version: str = RUBRIC_VERSION
) -> dict:
    rubric_for_spec({"rubric_version": rubric_version})
    checkpoints, check_chapters = CHECKPOINTS, CHECK_CHAPTERS
    path = folder / "artifacts/requirements.md"
    text = path.read_text(encoding="utf-8") if path.is_file() and not path.is_symlink() else ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if text else None
    sections = {}
    for match in re.finditer(r"^##\s+([1-6])\s+([^\n]+)\n(.*?)(?=^##\s+|\Z)", text, re.M | re.S):
        number = int(match[1])
        sections.setdefault(number, []).append((match[2].strip(), match[3]))
    reviews = {}
    valid_review_keys = {
        f"{task}.{index}"
        for task, criteria in checkpoints.items()
        for index in range(1, len(criteria) + 1)
    }
    if reviews_path is not None:
        data = json.loads(reviews_path.read_text())
        if data.get("rubric_version") != rubric_version or data.get("document_sha256") != digest:
            raise ValueError("Review rubric or document hash mismatch")
        for row in data.get("checks", []):
            key = row["id"]
            if key not in valid_review_keys:
                raise ValueError(f"Unknown checkpoint for this rubric: {key}")
            if key in reviews or row.get("status") not in {"passed", "failed", "pending"}:
                raise ValueError("Duplicate or invalid checkpoint review")
            if not row.get("evidence") or row.get("review_kind") not in {
                "human",
                "analyst",
                "model",
            }:
                raise ValueError("Review needs evidence and explicit reviewer kind")
            if row["review_kind"] == "model" and not row.get("model"):
                raise ValueError("Automatic review needs the actual judge model")
            reviews[key] = row
    rows = []
    for task, checks in checkpoints.items():
        present, body, title, chapters, subsection = _task_scope(sections, task)
        details = []
        for index, criterion in enumerate(checks, 1):
            key = f"{task}.{index}"
            review = reviews.get(key)
            check_present, check_body = present, body
            if task in check_chapters:
                check_present, check_body, *_ = _task_scope(
                    sections, task, check_chapters[task][index - 1]
                )
            status = "failed" if not check_present else review["status"] if review else "pending"
            # Evidence must belong to this task, including its assigned subsection.
            if status == "passed" and not any(
                span in check_body for span in review.get("quotes", []) if len(span.strip()) >= 4
            ):
                raise ValueError(f"Passing review must quote its own chapter/task scope: {key}")
            details.append(
                {
                    "id": key,
                    "criterion": criterion,
                    "weight": 1 / len(checks),
                    "status": status,
                    "review": review,
                }
            )
        earned = sum(d["weight"] for d in details if d["status"] == "passed")
        pending = sum(d["weight"] for d in details if d["status"] == "pending")
        rows.append(
            {
                "id": task,
                "chapter": chapters[0] if len(chapters) == 1 else None,
                "chapters": list(chapters),
                "section": subsection,
                "title": title,
                "points": earned,
                "pending_points": pending,
                "checks": details,
            }
        )
    receipts_file = folder / "artifact-downloads.json"
    receipts = json.loads(receipts_file.read_text()) if receipts_file.exists() else []
    receipt = [r for r in receipts if r.get("filename") == "requirements.md"]
    result = (
        json.loads((folder / "result.json").read_text())
        if (folder / "result.json").exists()
        else {}
    )
    answer = (folder / "answer.md").read_text() if (folder / "answer.md").exists() else ""
    delivery = {
        "file_nonempty": bool(text.strip()),
        "download_hash": len(receipt) == 1
        and receipt[0].get("status") == 200
        and receipt[0].get("sha256") == digest,
        "final_download_link": len(receipt) == 1 and receipt[0].get("url", "missing-url") in answer,
        "execution_done": result.get("statistics", {}).get("status") == "done",
    }
    condition_file = folder / "condition.json"
    condition = json.loads(condition_file.read_text()) if condition_file.exists() else {}
    expected_url = receipt[0].get("url") if len(receipt) == 1 else None
    allowed = {expected_url} if expected_url else set()
    if expected_url and condition.get("chat_url"):
        allowed.add(condition["chat_url"].rstrip("/") + expected_url)
    urls = re.findall(r"https?://[^\s`<>\]\(\)]+|/api/sessions/[^\s`<>\]\(\)]+", answer)
    delivery["artifact_link_correct"] = any(url in allowed for url in urls)
    delivery["final_download_link"] = delivery["artifact_link_correct"]
    value = {
        "rubric_version": rubric_version,
        "document_sha256": digest,
        "subtasks": rows,
        "points": sum(r["points"] for r in rows),
        "pending_points": sum(r["pending_points"] for r in rows),
        "max_points": 6,
        "delivery": delivery,
        "scope": "Six equal-weight tasks; checkpoints divide task weight. Analyst/human evidence review is explicit. Delivery is separate.",
    }
    value["scope"] += " Full-document verification and delivery are separate closing work."
    timed_rows = reviewed_timings(
        folder,
        [
            {
                "id": row["id"],
                "final_passed": all(check["status"] == "passed" for check in row["checks"]),
            }
            for row in rows
        ],
    )
    value["latency"] = score_subtasks(
        "reqdoc_generation",
        condition.get("phase", "Q"),
        result.get("statistics", {}),
        timed_rows,
    )
    (folder / "grade.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return value
