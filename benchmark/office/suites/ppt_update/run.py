"""Execute one H/Q PPT task through Chat; observations never fabricate model work."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from benchmark.office.cases.ppt_update.acceptance import memory_in_first_input
from benchmark.office.cases.ppt_update.files import (
    collect_artifacts,
    export_tool_evidence,
    load_case,
)
from benchmark.office.suites.common.chat import chat_events
from benchmark.office.suites.common.files_config import write_json
from benchmark.office.suites.common.memory_inventory import memory_leaves
from benchmark.office.suites.common.timing import scoring_policy
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.identity import OVIdentity

from .gates import CaseContract, FirstInputGate
from .preparation import freeze_automatic_memory
from .scoring import score


async def fingerprint(sdk, identity):
    result = {}
    for item in await memory_leaves(sdk, identity):
        uri = item["uri"]
        body = await sdk.read(uri, identity=identity)
        result[uri] = hashlib.sha256(body.encode()).hexdigest()
    return result


async def run(args):
    root = Path.cwd()
    spec, source = load_case(args.spec)
    phase = spec["phases"][args.phase]
    if args.phase == "H" and args.mode != "current":
        raise ValueError("H uses current mode for Middleware automatic extraction")
    if args.phase == "Q" and args.mode not in {"baseline", "ordinary"}:
        raise ValueError("Q uses baseline or ordinary mode")
    if args.phase == "Q" and args.mode == "ordinary" and not args.expected_memory:
        raise ValueError("Q ordinary requires --expected-memory from H")
    prompt = (root / phase["prompt"]).read_text()
    assert hashlib.sha256(prompt.encode()).hexdigest() == phase["prompt_sha256"]
    case_contract = CaseContract.from_case(spec)
    expected = None
    if args.expected_memory:
        saved_memory = json.loads(args.expected_memory.read_text())
        case_contract.require_accepted(saved_memory)
        expected = saved_memory["fingerprint"]
    args.output.mkdir(parents=True, exist_ok=False)
    identity = OVIdentity(args.account, args.user, args.peer)
    sdk = None
    try:
        if args.mode != "baseline":
            sdk = AsyncHTTPClient(
                url=args.ov_url,
                api_key=args.ov_key.read_text().strip(),
                auth_mode="trusted",
                profile_enabled=False,
            )
            await sdk.initialize()
        before = await fingerprint(sdk, identity) if sdk else None
        write_json(args.output / "memory-before.json", {"fingerprint": before})
        if args.phase == "H" and before:
            raise ValueError("H requires empty initial memory")
        if args.expected_memory:
            if before != expected:
                raise ValueError("Frozen memory differs before Q")
        async with httpx.AsyncClient(
            base_url=args.chat_url, timeout=httpx.Timeout(1900, connect=20), trust_env=False
        ) as client:
            response = await client.get(
                "/api/execution-profiles", params={"environment_id": args.environment_id}
            )
            response.raise_for_status()
            meta = response.json()
            if not meta.get("automatic_benchmark"):
                raise ValueError("Chat must support automatic benchmark snapshots")
            if meta["account_id"] != args.account or meta["ov_url"].rstrip(
                "/"
            ) != args.ov_url.rstrip("/"):
                raise ValueError("Service identity/url mismatch")
            if any(
                meta.get("reader_defaults", {}).get(key) != value
                for key, value in spec["reader_contract"].items()
            ):
                raise ValueError("Chat Reader defaults differ from the frozen PPT case")
            if "pptx" not in meta.get("skills", {}):
                raise ValueError("PPT Skill not enabled")
            if args.expected_memory:
                saved_condition = args.expected_memory.parent / "condition.json"
                if json.loads(saved_condition.read_text())["metadata"] != meta:
                    raise ValueError("Chat model/Reader/Skill profile differs from H")
            execution = json.loads((root / spec["execution_profile"]).read_text())
            execution.update(
                mode=args.mode,
                memory_extraction=args.phase == "H",
                target_uris=["viking://user/memories", f"viking://user/peers/{args.peer}/memories"],
                system_prompt=(root / spec["system_prompt"]).read_text(),
            )
            if args.phase == "H":
                execution["commit"] = {
                    "token_budget": execution["context_window"],
                    "commit_token_threshold_ratio": 0.7,
                    "completed_run_threshold": 1,
                    "keep_recent_count": 0,
                }
            created = await client.post(
                "/api/sessions",
                json={
                    "user_id": args.user,
                    "agent_id": args.peer,
                    "environment_id": args.environment_id,
                    "experiment_id": f"office-auto--ppt_update--{args.phase}",
                    "execution": execution,
                },
            )
            created.raise_for_status()
            session = created.json()
            sid = session["session_id"]
            write_json(args.output / "session.json", session)
            write_json(
                args.output / "condition.json",
                {
                    "phase": args.phase,
                    "mode": args.mode,
                    "query": prompt,
                    "execution": execution,
                    "metadata": meta,
                    "input_sha256": spec["input_sha256"],
                    "prompt_sha256": phase["prompt_sha256"],
                    "support_files": spec["support_files"],
                    "scoring_policy": scoring_policy("ppt_update"),
                },
            )
            inputs = [(source, source.name, spec["input_sha256"])] + [
                (root / item["source"], item["filename"], item["sha256"])
                for item in spec["support_files"]
            ]
            receipts = []
            for path, filename, expected_sha in inputs:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected_sha:
                    raise ValueError("PPT/helper input changed before upload")
                uploaded = await client.post(
                    f"/api/sessions/{sid}/inputs/{filename}",
                    content=raw,
                    headers={"Content-Type": "application/octet-stream"},
                )
                uploaded.raise_for_status()
                receipt = uploaded.json()
                if receipt["sha256"] != expected_sha:
                    raise ValueError("Uploaded PPT/helper differs from case input")
                receipts.append(receipt)
            write_json(args.output / "input-receipt.json", receipts[0])
            write_json(args.output / "input-receipts.json", receipts)
            began = time.monotonic()
            gate = FirstInputGate(
                args.output,
                prompt,
                execution["system_prompt"],
                args.phase,
                args.mode,
            )
            try:
                async with client.stream(
                    "POST", "/api/chat", json={"session_id": sid, "message": prompt}
                ) as response:
                    events = await chat_events(
                        response, args.output / "stream-events.json", on_event=gate
                    )
            except BaseException:
                await client.post(f"/api/sessions/{sid}/cancel")
                raise
            run_id = next(e["data"]["run_id"] for e in events if e["event"] == "start")
            response = await client.get(f"/api/sessions/{sid}/runs/{run_id}")
            response.raise_for_status()
            trace = response.json()
            write_json(args.output / "trace.json", trace)
            export_tool_evidence(args.output, trace)
            response = await client.get(f"/api/sessions/{sid}/state")
            response.raise_for_status()
            state = response.json()
            write_json(args.output / "state.json", state)
            artifacts = await collect_artifacts(client, sid, run_id, args.output)
            messages = [e["data"]["message"] for e in trace["events"] if e["event"] == "message"]
            final = [
                m["content"]
                for m in messages
                if m["type"] == "ai"
                and not m.get("tool_calls")
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ]
            (args.output / "answer.md").write_text((final[-1] if final else "") + "\n")
            after = await fingerprint(sdk, identity) if sdk else None
            write_json(args.output / "memory-after.json", {"fingerprint": after})
            result = {
                "phase": args.phase,
                "mode": args.mode,
                "session_id": sid,
                "run_id": run_id,
                "statistics": trace["statistics"],
                "terminal": events[-1],
                "stream_seconds": time.monotonic() - began,
                "artifacts": artifacts,
                "memory_unchanged": before == after if sdk else None,
                "real_memory_in_first_input": memory_in_first_input(trace) if sdk else False,
            }
            write_json(args.output / "result.json", result)
            audit = args.chat_data / "ppt-audit" / sid / run_id
            automatic = await client.post(f"/api/sessions/{sid}/runs/{run_id}/benchmark")
            automatic.raise_for_status()
            auto_grade = automatic.json()
            write_json(args.output / "automatic-grade.json", auto_grade)
            content_grade = score(spec, args.phase, args.output, audit, source)
            content_grade.update(latency=auto_grade["latency"], automatic=auto_grade)
            write_json(args.output / "grade.json", content_grade)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "phase": args.phase,
                        "terminal": events[-1]["event"],
                        "statistics": trace["statistics"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if args.phase == "H" and sdk:
            grade = json.loads((args.output / "grade.json").read_text())
            if not grade["task_pass"] or trace["statistics"]["status"] != "done":
                raise ValueError("H acceptance failed; stop memory preparation")
            await freeze_automatic_memory(
                sdk,
                identity,
                args.output,
                trace,
                state,
                fingerprint,
                case_contract=case_contract,
            )
        if args.expected_memory and after != expected:
            raise ValueError("Frozen memory changed during Q")
    except Exception as e:
        write_json(
            args.output / "failure.json", {"error_type": type(e).__name__, "message": str(e)}
        )
        raise
    finally:
        if sdk:
            await sdk.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmark/office/cases/ppt_update/spec.json")
    )
    parser.add_argument("--phase", choices=["H", "Q"], required=True)
    parser.add_argument("--mode", choices=["current", "ordinary", "baseline"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-data", type=Path, required=True)
    parser.add_argument("--chat-url", required=True)
    parser.add_argument("--environment-id", default="daily")
    parser.add_argument("--ov-url", required=True)
    parser.add_argument("--ov-key", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--expected-memory", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
