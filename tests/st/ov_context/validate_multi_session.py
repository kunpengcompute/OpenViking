# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Write isolated test sessions to a running Chat; report checks, never raw traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from benchmark.paths import RESULT_ROOT


class ValidationError(RuntimeError):
    """A functional assertion failed; messages do not include model content."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_messages(messages: list[dict[str, Any]], marker: str, forbidden: set[str]) -> None:
    wire = json.dumps(messages, ensure_ascii=False)
    require(marker in wire, "Own session marker missing from model history")
    require(
        not any(value in wire for value in forbidden), "Foreign session marker in model history"
    )
    ids: set[str] = set()
    pending: set[str] = set()
    for message in messages:
        kind = message["type"]
        require(kind in {"system", "human", "ai", "tool"}, "Unexpected model message type")
        if kind == "system":
            continue
        key = message.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise ValidationError("Missing or duplicate message ID")
        ids.add(key)
        if kind == "tool":
            call = message.get("tool_call_id")
            if not isinstance(call, str) or call not in pending:
                raise ValidationError("Orphan or duplicate tool result")
            pending.remove(call)
        else:
            require(not pending, "Incomplete tool round before next message")
            for call in message.get("tool_calls", []):
                require(bool(call["id"]) and call["id"] not in pending, "Duplicate tool call")
                pending.add(call["id"])
    require(not pending, "Incomplete tool round at model boundary")


@dataclass(frozen=True)
class Case:
    alias: str
    user: str
    agent: str
    session: str
    marker: str
    thread: str
    account: str

    @property
    def owner(self) -> list[str]:
        return [self.account, self.user, self.agent, self.session]


@dataclass(frozen=True)
class Observation:
    case: str
    phase: str
    model_inputs: int
    tool_results: int
    cached_messages: int
    raw_messages: int
    total_tokens: int | None


async def detail(client: httpx.AsyncClient, case: Case) -> dict[str, Any]:
    response = await client.get(f"/api/sessions/{case.session}")
    response.raise_for_status()
    return response.json()


async def run_case(
    client: httpx.AsyncClient, case: Case, phase: str, forbidden: set[str]
) -> Observation:
    prompt = (
        f"本会话的测试标记是 {case.marker}。请调用 debug_echo 回显这个标记，最后只回复工具返回的标记。"
        if phase == "tool"
        else "请只回复本会话前面记录的测试标记，不调用工具；不要猜测，也不要使用其他会话的信息。"
    )
    terminal: dict[str, Any] | None = None
    model_inputs = tool_results = 0
    answer = ""
    async with client.stream(
        "POST", "/api/chat", json={"session_id": case.session, "message": prompt}
    ) as response:
        response.raise_for_status()
        kind, rows = "", []
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                rows.append(line[6:])
            elif not line and rows:
                data = json.loads("\n".join(rows))
                rows = []
                if kind == "start":
                    require(data["identity"] == case.owner, "SSE identity mismatch")
                    require(data["thread_id"] == case.thread, "Checkpoint thread mismatch")
                elif kind == "model_input":
                    validate_messages(data["messages"], case.marker, forbidden)
                    model_inputs += 1
                elif kind == "message":
                    message = data["message"]
                    if message["type"] == "tool":
                        tool_results += 1
                    elif message["type"] == "ai" and not message.get("tool_calls"):
                        answer = str(message["content"])
                elif kind == "done":
                    terminal = data
                elif kind in {"error", "cancelled"}:
                    raise ValidationError(
                        f"{case.alias}/{phase}: {kind} ({data.get('error_type', 'unknown')})"
                    )
    if terminal is None:
        raise ValidationError("Stream ended without done")
    require(model_inputs > 0, "No observed model input")
    require(
        case.marker in answer and not any(m in answer for m in forbidden),
        "Answer did not recall own marker",
    )
    if phase == "tool":
        require(tool_results >= 1 and model_inputs >= 2, "Tool loop not executed")
    view = terminal["context"]
    require(view["owner"] == case.owner, "Cache owner mismatch")
    require(view["source"] == "checkpoint", "Missing saved checkpoint")
    require(view["run_id"] is None, "Run temporary state leaked")
    require(view["write_index"] == view["raw_message_count"] > 0, "Incomplete write cursor")
    validate_messages(view["messages"], case.marker, forbidden)
    # Run-only recall must be removed from the saved session cache.
    require(
        not any(
            m.get("additional_kwargs", {}).get("ov_context_injected") for m in view["messages"]
        ),
        "Recall view leaked after Run",
    )
    saved = await detail(client, case)
    require(not saved["busy"] and not saved["session"]["needs_restore"], "Session not released")
    require(
        saved["session"]["last_run"] == terminal["statistics"], "Statistics persistence mismatch"
    )
    return Observation(
        case.alias,
        phase,
        model_inputs,
        tool_results,
        view["cached_message_count"],
        view["raw_message_count"],
        terminal["statistics"]["total_tokens"],
    )


async def validate(url: str, parallelism: int = 2) -> dict[str, Any]:
    prefix = "matrix-" + uuid.uuid4().hex[:10]
    specifications = [
        (f"u{u}-a{a}", f"{prefix}-u{u}", f"agent-{a}") for u in (1, 2) for a in (1, 2)
    ]
    specifications.append(("u1-a1-session2", f"{prefix}-u1", "agent-1"))
    cases: list[Case] = []
    results: list[Observation] = []
    async with httpx.AsyncClient(base_url=url.rstrip("/"), trust_env=False, timeout=180) as client:
        health = await client.get("/api/health")
        health.raise_for_status()
        for alias, user, agent in specifications:
            response = await client.post("/api/sessions", json={"user_id": user, "agent_id": agent})
            response.raise_for_status()
            session = response.json()
            case_id = session["session_id"]
            # The server's thread key is verified from its typed session contract.
            from examples.deepagents_ov_context.chat.server import SessionInfo

            info = SessionInfo.model_validate(session)
            cases.append(
                Case(
                    alias,
                    user,
                    agent,
                    case_id,
                    "CASE-" + uuid.uuid4().hex,
                    info.thread_id,
                    info.account_id,
                )
            )
        require(len({c.thread for c in cases}) == len(cases), "Sessions share checkpoint thread")
        markers = {c.marker for c in cases}
        semaphore = asyncio.Semaphore(parallelism)

        async def execute(case: Case, phase: str) -> Observation:
            async with semaphore:
                return await run_case(client, case, phase, markers - {case.marker})

        for phase in ("tool", "checkpoint", "ov-restore"):
            if phase == "ov-restore":
                for case in cases:
                    response = await client.post(f"/api/sessions/{case.session}/restore")
                    response.raise_for_status()
                    restored = response.json()
                    require(
                        restored["context"]["source"] == "ov",
                        "OV restore did not replace checkpoint",
                    )
                    validate_messages(
                        restored["context"]["messages"], case.marker, markers - {case.marker}
                    )
            observed = await asyncio.gather(*(execute(case, phase) for case in cases))
            results.extend(observed)
            print(
                json.dumps({"phase": phase, "passed_sessions": len(observed)}, ensure_ascii=False),
                flush=True,
            )
    return {
        "status": "passed",
        "users": 2,
        "agents": 2,
        "sessions": len(cases),
        "parallelism": parallelism,
        "runs": len(results),
        "observations": [asdict(value) for value in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1934")
    parser.add_argument("--parallelism", type=int, choices=range(1, 6), default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "st/ov_context/multi-session/results.json",
        help="Write only assertion summaries and counts",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(validate(args.url, args.parallelism))
    except (ValidationError, httpx.HTTPError) as exc:
        # Do not dump request/response bodies or model traces on failure.
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
