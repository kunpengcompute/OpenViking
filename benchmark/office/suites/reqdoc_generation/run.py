"""Run single-document H/Q tasks through Chat with versioned ordinary-memory seeds."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from benchmark.office.cases.ppt_update.files import collect_artifacts, export_tool_evidence
from benchmark.office.cases.reqdoc_generation.files import load_case
from benchmark.office.suites.common.chat import chat_events
from benchmark.office.suites.common.files_config import write_json
from benchmark.office.suites.common.memory_inventory import memory_leaves
from benchmark.office.suites.common.timing import scoring_policy
from examples.deepagents_ov_context.chat.prompts import chat_base_prompt
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity

from .scoring import grade, rubric_for_spec

ORDINARY_TYPES = {"profile", "preferences", "entities", "events"}


def category(uri: str) -> str:
    tail = uri.split("/memories/", 1)[-1]
    return (
        "profile"
        if tail == "profile.md"
        else tail.split("/", 1)[0]
        if "/" in tail
        else "template_or_unknown"
    )


async def inventory(sdk, identity):
    result = {}
    for leaf in await memory_leaves(sdk, identity):
        body = await sdk.read(leaf["uri"], identity=identity)
        result[leaf["uri"]] = {
            "category": category(leaf["uri"]),
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "content": body,
        }
    return result


def fingerprints(pool):
    return {u: v["sha256"] for u, v in pool.items() if v["category"] in ORDINARY_TYPES}


def seed_contract(spec):
    return {
        "case_version": spec["version"],
        "rubric_version": rubric_for_spec(spec),
        "H_prompt_sha256": spec["phases"]["H"]["sha256"],
        "Q_prompt_sha256": spec["phases"]["Q"]["sha256"],
        "input_fingerprints": {item["filename"]: item["sha256"] for item in spec["inputs"]},
    }


def seed_fingerprint(seed, spec):
    if any(seed.get(key) != value for key, value in seed_contract(spec).items()):
        raise ValueError("Frozen H seed belongs to a different case, prompt, inputs or rubric")
    value = seed.get("fingerprint")
    if not isinstance(value, dict) or not value:
        raise ValueError("Ordinary Q needs a nonempty frozen seed fingerprint")
    return value


class FirstInputGate:
    def __init__(self, phase, mode, pool, folder):
        self.phase, self.mode, self.pool, self.folder = phase, mode, pool, folder
        self.checked = False
        self.recalls = []
        self.ov_requests = []
        self.recall_attempts = []
        self.recall_retries = []

    def __call__(self, event):
        data = event["data"]
        if event["event"] == "stage":
            if data.get("name") == "memory_recall_result":
                self.recalls.append(data)
            if data.get("name") == "ov_request":
                self.ov_requests.append(data)
            if data.get("name") == "memory_recall_attempt":
                self.recall_attempts.append(data)
            if data.get("name") == "memory_recall_retry":
                self.recall_retries.append(data)
        if event["event"] != "model_input" or self.checked:
            return
        self.checked = True
        text = "\n".join(
            str(m.get("content", ""))
            for m in data["messages"]
            if m.get("type") in {"human", "user"}
        )
        memory_text = ""
        if "[参考记忆]" in text and "[用户原文]" in text:
            memory_text = text.split("[参考记忆]", 1)[1].split("[用户原文]", 1)[0]
        matches = []
        for uri, row in self.pool.items():
            if row["category"] not in ORDINARY_TYPES:
                continue
            if any(
                line.strip()[:60] in memory_text
                for line in row["content"].splitlines()
                if len(line.strip()) >= 25
            ):
                matches.append(uri)
        task_ok = "output/requirements.md" in text and "RD06" in text
        if self.mode == "ordinary" and self.phase == "Q":
            passed = (
                task_ok
                and bool(matches)
                and any(
                    r.get("status") == "success" and r.get("count", 0) > 0 for r in self.recalls
                )
            )
        else:
            passed = task_ok and not self.recalls and not self.ov_requests
        # H normally emits a single empty recall event; its empty result is valid.
        if self.phase == "H":
            passed = (
                task_ok
                and "【H轮执行方法指导】" in text
                and not matches
                and not any(r.get("count", 0) > 0 for r in self.recalls)
            )
        value = {
            "phase": self.phase,
            "mode": self.mode,
            "passed": bool(passed),
            "task_present": task_ok,
            "matched_memory_uris": matches,
            "recalls": self.recalls,
            "recall_attempts": self.recall_attempts,
            "recall_retries": self.recall_retries,
            "estimated_input_tokens": data.get("estimated_input_tokens"),
            "model_call_id": data.get("model_call_id"),
        }
        write_json(self.folder / "first-input-gate.json", value)
        (self.folder / "first-human-input.txt").write_text(text)
        if not passed:
            raise ValueError("First model input failed the frozen experiment gate")

    def finish(self):
        if not self.checked:
            write_json(
                self.folder / "first-input-gate.json",
                {"passed": False, "reason": "Run ended before first model input"},
            )
            raise ValueError("Run ended before first model input; invalid comparison sample")


async def run(args):
    root = Path.cwd()
    spec, sources = load_case(args.spec)
    rubric_version = rubric_for_spec(spec)
    phase = args.phase
    mode = args.mode or ("current" if phase == "H" else "ordinary")
    if phase == "H" and mode != "current":
        raise ValueError("H uses the standard current middleware profile")
    if phase == "Q" and mode not in {"ordinary", "baseline"}:
        raise ValueError("Q mode must be ordinary or baseline")
    p = root / spec["phases"][phase]["prompt"]
    prompt = p.read_text()
    if hashlib.sha256(p.read_bytes()).hexdigest() != spec["phases"][phase]["sha256"]:
        raise ValueError("Prompt hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = OVIdentity(args.account, args.user, args.peer)
    policy = spec["ordinary_policy"]
    if set(policy["memory_types"]) != ORDINARY_TYPES:
        raise ValueError("Extraction must whitelist ordinary types")
    frozen_seed = None
    expected = None
    if phase == "Q" and mode == "ordinary":
        if not args.expected_memory:
            raise ValueError("Ordinary Q needs a frozen seed fingerprint")
        frozen_seed = json.loads(args.expected_memory.read_text())
        expected = seed_fingerprint(frozen_seed, spec)
    sdk = None
    if mode != "baseline":
        sdk = AsyncHTTPClient(
            url=args.ov_url,
            auth_mode="trusted",
            api_key=args.ov_key.read_text().strip(),
            profile_enabled=False,
        )
        await sdk.initialize()
    try:
        before = await inventory(sdk, identity) if sdk else {}
        write_json(args.output / "memory-before.json", before)
        if phase == "H" and before:
            raise ValueError("H needs a fresh empty pool")
        if phase == "Q" and sdk:
            if fingerprints(before) != expected:
                raise ValueError("Frozen pool differs before Q")
            write_json(args.output / "expected-memory.json", frozen_seed)
        async with httpx.AsyncClient(
            base_url=args.chat_url, trust_env=False, timeout=httpx.Timeout(3660, connect=20)
        ) as client:
            r = await client.get("/api/execution-profiles")
            r.raise_for_status()
            meta = r.json()
            if not meta.get("automatic_benchmark"):
                raise ValueError("Chat must support automatic benchmark snapshots")
            if meta["account_id"] != args.account or meta["ov_url"].rstrip(
                "/"
            ) != args.ov_url.rstrip("/"):
                raise ValueError("Destination mismatch")
            if not {"office-documents", "data-analysis"}.issubset(meta.get("skills", {})):
                raise ValueError("Required skills missing")
            execution = {
                "mode": mode,
                "memory_extraction": phase == "H",
                "context_window": 256000,
                "max_tokens": 16384,
                "thinking": True,
                "recursion_limit": 1000,
                "timeout_seconds": 3600,
                "model_call_timeout_seconds": 180,
                "system_prompt": chat_base_prompt(),
                "memory_budget_tokens": 4000,
                "memory_detail": "full",
                "memory_recall_retry_attempts": args.recall_retry_attempts
                if phase == "Q" and mode == "ordinary"
                else 0,
            }
            r = await client.post(
                "/api/sessions",
                json={
                    "user_id": args.user,
                    "agent_id": args.peer,
                    "execution": execution,
                    "experiment_id": f"office-auto--reqdoc_generation--{phase}",
                },
            )
            r.raise_for_status()
            session = r.json()
            sid = session["session_id"]
            write_json(args.output / "session.json", session)
            metadata = {}
            initialized = {}
            if sdk:
                session_policy = (
                    policy
                    if phase == "H"
                    else {**policy, "self": {"enabled": False}, "peer": {"enabled": False}}
                )
                receipt = await sdk.create_session(
                    session_id=sid, identity=identity, memory_policy=session_policy
                )
                write_json(args.output / "ov-create-receipt.json", receipt)
                metadata = await sdk.get_session(sid, identity=identity)
                write_json(args.output / "ov-session-before.json", metadata)
                if set(metadata.get("memory_policy", {}).get("memory_types", [])) != ORDINARY_TYPES:
                    raise ValueError("Whitelist was not persisted")
                initialized = await inventory(sdk, identity)
                write_json(args.output / "memory-initialized.json", initialized)
            uploads = []
            for source in sources:
                r = await client.post(
                    f"/api/sessions/{sid}/inputs/{source.name}",
                    content=source.read_bytes(),
                    headers={"Content-Type": "application/octet-stream"},
                )
                r.raise_for_status()
                item = r.json()
                if item["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
                    raise ValueError("Upload hash mismatch")
                uploads.append(item)
            write_json(
                args.output / "condition.json",
                {
                    "version": spec["version"],
                    "phase": phase,
                    "group": mode,
                    "memory_policy": metadata.get("memory_policy"),
                    "query": prompt,
                    "prompt_sha256": spec["phases"][phase]["sha256"],
                    "execution": execution,
                    "metadata": meta,
                    "inputs": uploads,
                    "score_rubric": rubric_version,
                    "scoring_policy": scoring_policy("reqdoc_generation"),
                    "closing": spec.get("closing"),
                    "chat_url": args.chat_url.rstrip("/"),
                    "model_calls_via": "Chat HTTP/SSE",
                },
            )
            gate = FirstInputGate(phase, mode, before, args.output)
            started = time.monotonic()
            async with client.stream(
                "POST", "/api/chat", json={"session_id": sid, "message": prompt}
            ) as response:
                events = await chat_events(
                    response, args.output / "stream-events.json", on_event=gate
                )
            rid = next(e["data"]["run_id"] for e in events if e["event"] == "start")
            r = await client.get(f"/api/sessions/{sid}/runs/{rid}")
            r.raise_for_status()
            trace = r.json()
            write_json(args.output / "trace.json", trace)
            export_tool_evidence(args.output, trace)
            r = await client.get(f"/api/sessions/{sid}/state")
            r.raise_for_status()
            write_json(args.output / "state.json", r.json())
            artifacts = await collect_artifacts(client, sid, rid, args.output)
            msgs = [e["data"]["message"] for e in trace["events"] if e["event"] == "message"]
            final = [
                m["content"]
                for m in msgs
                if m["type"] == "ai"
                and not m.get("tool_calls")
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ]
            (args.output / "answer.md").write_text((final[-1] if final else "") + "\n")
            result = {
                "session_id": sid,
                "run_id": rid,
                "phase": phase,
                "group": mode,
                "statistics": trace["statistics"],
                "terminal": events[-1],
                "stream_seconds": time.monotonic() - started,
                "artifacts": artifacts,
                "execution_completed": events[-1]["event"] == "done",
                "ov_requests": gate.ov_requests,
                "recalls": gate.recalls,
                "recall_attempts": gate.recall_attempts,
                "recall_retries": gate.recall_retries,
                "first_input_gate_passed": gate.checked
                and json.loads((args.output / "first-input-gate.json").read_text())["passed"],
            }
            write_json(args.output / "result.json", result)
            automatic = await client.post(f"/api/sessions/{sid}/runs/{rid}/benchmark")
            automatic.raise_for_status()
            auto_grade = automatic.json()
            write_json(args.output / "automatic-grade.json", auto_grade)
            review_path = None
            if auto_grade.get("document_review"):
                review_path = args.output / "automatic-reviews.json"
                write_json(review_path, auto_grade["document_review"])
            content_grade = grade(
                args.output, reviews_path=review_path, rubric_version=rubric_version
            )
            content_grade.update(latency=auto_grade["latency"], automatic=auto_grade)
            write_json(args.output / "grade.json", content_grade)
            print(json.dumps({"execution": result["statistics"]}, ensure_ascii=False), flush=True)
            gate.finish()
        if not sdk:
            return
        persisted = await sdk.get_session(sid, identity=identity)
        write_json(args.output / "ov-session-after.json", persisted)
        if persisted["memory_policy"] != metadata["memory_policy"]:
            raise ValueError("Run changed extraction policy")
        if phase == "Q":
            after = await inventory(sdk, identity)
            write_json(args.output / "memory-after.json", after)
            if before != after or fingerprints(after) != expected:
                raise ValueError("Memory pool changed during Q")
            write_json(args.output / "memory-audit.json", {"unchanged": True, "matches_seed": True})
            return
        if not result["execution_completed"]:
            return
        receipt = await sdk.commit_session(sid, keep_recent_count=0, identity=identity)
        write_json(args.output / "commit-receipt.json", receipt)
        task_id = receipt.get("task_id")
        if not task_id:
            raise ValueError("No extraction task receipt")
        start = time.monotonic()
        while True:
            task = await sdk.get_task(task_id, identity=identity)
            write_json(args.output / "extraction-task.json", task)
            if task.get("status") in {"completed", "failed"}:
                break
            if time.monotonic() - start > 1200:
                raise TimeoutError("Extraction exceeded 1200 seconds")
            await asyncio.sleep(2)
        after = await inventory(sdk, identity)
        write_json(args.output / "memory-after.json", after)
        ordinary = {u: v for u, v in after.items() if v["category"] in ORDINARY_TYPES}
        forbidden = {
            u: v
            for u, v in after.items()
            if v["category"] not in ORDINARY_TYPES
            and (u not in initialized or v["sha256"] != initialized[u]["sha256"])
        }
        audit = {
            "task_status": task["status"],
            "allowed_types": sorted(ORDINARY_TYPES),
            "ordinary_count": len(ordinary),
            "ordinary_uris": sorted(ordinary),
            "forbidden_new_or_changed": sorted(forbidden),
            "strict_type_check_passed": not forbidden,
            "ordinary_memory_produced": bool(ordinary),
            "extraction_seconds": time.monotonic() - start,
            "token_usage": (task.get("result") or {}).get("token_usage"),
        }
        write_json(args.output / "memory-audit.json", audit)
        write_json(
            args.output / "frozen-memory.json",
            {**seed_contract(spec), "fingerprint": fingerprints(ordinary)},
        )
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        if forbidden:
            raise ValueError("Nonordinary memory written despite policy")
        if task["status"] == "failed":
            raise RuntimeError("Extraction failed; evidence retained")
    finally:
        if sdk:
            await sdk.close()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--spec",
        type=Path,
        default=Path("benchmark/office/cases/reqdoc_generation/spec.json"),
        help="Frozen task and scoring specification",
    )
    p.add_argument("--phase", choices=["H", "Q"], default="H")
    p.add_argument("--mode", choices=["current", "ordinary", "baseline"])
    p.add_argument("--expected-memory", type=Path)
    p.add_argument("--recall-retry-attempts", type=int, choices=range(4), default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chat-url", required=True)
    p.add_argument("--ov-url", required=True)
    p.add_argument("--ov-key", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--peer", default="reqdoc-agent")
    return p


def main():
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        if args.output.exists():
            write_json(
                args.output / "runner-error.json", {"type": type(exc).__name__, "message": str(exc)}
            )
        raise


if __name__ == "__main__":
    main()
