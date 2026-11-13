# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real cross-session user-memory extraction, retrieval and input-injection acceptance."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from benchmark.paths import RESULT_ROOT
from examples.deepagents_ov_context.chat.config import load_service_api_key
from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader
from openviking_cli.client import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity
from tests.st.ov_context.validate_multi_session import ValidationError, require

SEED = (
    "请记住我的长期工作偏好：我每周五整理项目周报；周报固定分为“进展、风险、下一步”三部分；"
    "风险项必须写负责人和预计解决日期；尽量用简短中文。以后帮我写周报时都遵循这些偏好。现在只回复已记住。"
)
QUERY = "请按我以往的偏好列出项目周报模板，项目进展正常但有交付风险。不要猜测未记录的个人偏好。"
REQUIRED = ("进展", "风险", "下一步", "负责人")


def has_preferences(text: str) -> bool:
    return all(word in text for word in REQUIRED) and ("日期" in text or "时间" in text)


@dataclass(frozen=True)
class MemoryRun:
    phase: str
    recall_status: str | None
    recall_count: int
    injected: bool
    answer_uses_preferences: bool
    input_messages: int
    total_tokens: int | None
    directory_only_reference: bool = False


async def create_session(client: httpx.AsyncClient, user: str, agent: str) -> dict[str, Any]:
    response = await client.post("/api/sessions", json={"user_id": user, "agent_id": agent})
    response.raise_for_status()
    return response.json()


async def run_chat(
    client: httpx.AsyncClient,
    session: dict[str, Any],
    prompt: str,
    phase: str,
    *,
    expected_memory: bool | None,
    forbidden_id: str | None = None,
) -> tuple[MemoryRun, str | None]:
    terminal = None
    recall_status = None
    recall_count = input_count = 0
    injected = False
    directory_only = False
    answer = ""
    current_id = None
    async with client.stream(
        "POST", "/api/chat", json={"session_id": session["session_id"], "message": prompt}
    ) as response:
        response.raise_for_status()
        kind, lines = "", []
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                lines.append(line[6:])
            elif not line and lines:
                data = json.loads("\n".join(lines))
                lines = []
                if kind == "stage" and data.get("name") == "memory.recall":
                    recall_status, recall_count = data.get("status"), data.get("count", 0)
                elif kind == "model_input":
                    messages = data["messages"]
                    input_count = len(messages)
                    require(
                        not forbidden_id or all(m.get("id") != forbidden_id for m in messages),
                        "Archived raw message was re-appended",
                    )
                    users = [m for m in messages if m["type"] == "human"]
                    current = users[-1]
                    current_id = current["id"]
                    require(
                        str(current["content"]).endswith(prompt), "Original query lost or altered"
                    )
                    view = current.get("additional_kwargs", {}).get("ov_context_injected", False)
                    if view:
                        injected = has_preferences(str(current["content"]).split("[用户原文]")[0])
                    if expected_memory is True:
                        require(
                            view and injected, "Expected preferences absent from actual model input"
                        )
                    elif expected_memory is False:
                        if view:
                            reference = str(current["content"]).split("[用户原文]")[0]
                            sources = re.findall(r"来源：\s*(viking://\S+)", reference)
                            own_directory = (
                                f"viking://user/{session.get('user_id', '')}/memories/.abstract.md"
                            )
                            # An empty user's own directory description is not
                            # personal memory. Allow only that exact source;
                            # foreign URIs, leaf memories and seed facts fail.
                            directory_only = bool(sources) and all(
                                uri == own_directory for uri in sources
                            )
                            require(
                                directory_only and not injected,
                                "Other user received injected memory",
                            )
                elif kind == "message" and data["message"]["type"] == "ai":
                    answer = str(data["message"]["content"])
                elif kind == "done":
                    terminal = data
                elif kind in {"error", "cancelled"}:
                    raise ValidationError(f"{phase}: {kind} ({data.get('error_type', 'unknown')})")
    require(terminal is not None, f"{phase}: no done event")
    assert terminal is not None
    require(input_count > 0, "No actual model input observed")
    if expected_memory is True:
        require(recall_status == "success" and recall_count > 0, "Recall did not report success")
        require(has_preferences(answer), "Answer did not use extracted preferences")
    elif expected_memory is False:
        require(
            (recall_status == "empty" and recall_count == 0)
            or (directory_only and recall_status == "success" and recall_count == 1),
            "Other user recalled non-directory memory",
        )
    context = terminal["context"]
    require(
        not any(
            m.get("additional_kwargs", {}).get("ov_context_injected") for m in context["messages"]
        ),
        "Run recall leaked into persisted cache",
    )
    require(context["write_index"] == context["raw_message_count"], "Write cursor mismatch")
    return MemoryRun(
        phase,
        recall_status,
        recall_count,
        injected,
        has_preferences(answer),
        input_count,
        terminal["statistics"]["total_tokens"],
        directory_only,
    ), current_id


async def wait_extraction(client: httpx.AsyncClient, sid: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/api/sessions/{sid}/memory")
        response.raise_for_status()
        status = response.json()
        if status["status"] == "completed":
            require(
                sum(status["memories_extracted"].values()) > 0,
                "Task completed without generated memories",
            )
            return status
        require(
            status["status"] in {"pending", "running"},
            f"Extraction did not complete: {status['status']}",
        )
        await asyncio.sleep(2)
    raise ValidationError("Memory extraction deadline exceeded; task may continue on OV")


async def validate(
    chat_url: str, ov_url: str, timeout: float, *, api_key_file: Path | None = None
) -> dict[str, Any]:
    user = "memory-demo-" + uuid.uuid4().hex[:10]
    runs: list[MemoryRun] = []
    async with httpx.AsyncClient(
        base_url=chat_url.rstrip("/"), trust_env=False, timeout=180
    ) as client:
        seed = await create_session(client, user, "memory-agent")
        observed, seed_id = await run_chat(client, seed, SEED, "seed", expected_memory=False)
        runs.append(observed)
        receipt = await client.post(f"/api/sessions/{seed['session_id']}/memory")
        receipt.raise_for_status()
        require(receipt.json()["task_id"] is not None, "Commit returned no extraction task")
        extracted = await wait_extraction(client, seed["session_id"], timeout)
        print(
            json.dumps(
                {
                    "phase": "extraction",
                    "status": "completed",
                    "memories": sum(extracted["memories_extracted"].values()),
                }
            ),
            flush=True,
        )
        sdk = AsyncHTTPClient(
            url=ov_url,
            auth_mode="trusted",
            api_key=load_service_api_key(key_file=api_key_file),
            profile_enabled=False,
        )
        await sdk.initialize()
        try:
            identity = OVIdentity(seed["account_id"], user, seed["agent_id"])
            reader = OVContextReader(sdk, MemoryReaderConfig(recall_timeout=30))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                recalled = await reader.recall(
                    QUERY, identity=identity, session_id=seed["session_id"]
                )
                if recalled.status == "success" and has_preferences(recalled.content):
                    break
                require(
                    recalled.status not in {"error", "timeout"},
                    "Reader recall failed during index readiness check",
                )
                await asyncio.sleep(2)
            else:
                raise ValidationError("Extracted memory did not become searchable before deadline")
            meta = await sdk.get_session(seed["session_id"], identity=identity)
            require(
                meta.get("latest_finished_archive_id") is not None,
                "No terminal archive boundary published",
            )
        finally:
            await sdk.close()
        for phase, target_user, peer, existing in (
            ("seed-after-archive", user, "memory-agent", seed),
            ("new-session", user, "memory-agent", None),
            ("other-agent-same-user", user, "other-agent", None),
            ("other-user", user + "-other", "memory-agent", None),
        ):
            session = existing or await create_session(client, target_user, peer)
            result, _ = await run_chat(
                client,
                session,
                QUERY,
                phase,
                expected_memory=phase != "other-user",
                forbidden_id=seed_id,
            )
            runs.append(result)
            print(
                json.dumps(
                    {"phase": phase, "status": "passed", "recall_count": result.recall_count}
                ),
                flush=True,
            )
    return {
        "status": "passed",
        "runs": [asdict(run) for run in runs],
        "memories_extracted": extracted["memories_extracted"],
        "checks": [
            "native commit task",
            "terminal archive",
            "search readiness",
            "actual input injection",
            "cross-session recall",
            "same-user other-agent recall",
            "other-user isolation",
            "archived raw message removed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-url", default="http://127.0.0.1:1934")
    parser.add_argument("--ov-url", default="http://127.0.0.1:1933")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "st/ov_context/memory/results.json"
    )
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        result = asyncio.run(
            validate(args.chat_url, args.ov_url, args.timeout, api_key_file=args.api_key_file)
        )
    except (ValidationError, httpx.HTTPError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": str(exc) if isinstance(exc, ValidationError) else type(exc).__name__,
                }
            )
        )
        raise SystemExit(1) from None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
