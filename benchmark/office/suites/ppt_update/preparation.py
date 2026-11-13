"""Observe H's Middleware automatic commit; never submit a replacement commit."""

from __future__ import annotations

import asyncio
import hashlib
import time

from benchmark.office.suites.common.files_config import write_json

from .gates import CaseContract


async def freeze_automatic_memory(
    sdk, identity, folder, trace, state, fingerprint, *, case_contract=None
):
    contract = case_contract or CaseContract.from_case()
    frozen = folder / "frozen-memory.json"
    if frozen.exists():
        raise FileExistsError("H memory is already frozen")
    receipts = [
        event["data"]
        for event in trace["events"]
        if event["event"] == "stage"
        and event["data"].get("name") == "context.commit"
        and event["data"].get("status") == "accepted"
        and event["data"].get("task_id")
    ]
    pending = (state.get("ov_context") or {}).get("pending_commit") or {}
    task_ids = list(
        dict.fromkeys(
            [r["task_id"] for r in receipts]
            + ([pending["task_id"]] if pending.get("task_id") else [])
        )
    )
    if not receipts:
        raise RuntimeError("H has no observed Middleware automatic commit")
    evidence = {"automatic_only": True, "task_ids": task_ids, "manual_commit": False}
    write_json(folder / "automatic-commit-evidence.json", evidence)
    began = time.monotonic()
    for task_id in task_ids:
        while True:
            task = await sdk.get_task(task_id, identity=identity)
            write_json(folder / "extraction-task.json", task)
            if task.get("status") == "completed":
                break
            if task.get("status") in {"failed", "cancelled"}:
                raise RuntimeError("H automatic extraction failed")
            if time.monotonic() - began > 900:
                raise TimeoutError("H automatic extraction exceeded observation timeout")
            await asyncio.sleep(2)
    pool = await fingerprint(sdk, identity)
    bodies = [
        {"uri": uri, "sha256": sha, "content": await sdk.read(uri, identity=identity)}
        for uri, sha in pool.items()
    ]
    write_json(folder / "memory-bodies.json", bodies)
    evidence.update(memory_count=len(pool), wait_seconds=time.monotonic() - began)
    write_json(folder / "automatic-commit-evidence.json", evidence)
    acceptance = {"contract_sha256": contract.sha256}
    acceptance["memory_count"] = len(pool)
    acceptance["body_hashes_match"] = all(
        hashlib.sha256(item["content"].encode()).hexdigest() == item["sha256"] for item in bodies
    )
    acceptance["pool_unchanged"] = pool == await fingerprint(sdk, identity)
    acceptance["passed"] = bool(
        pool
        and all(item["content"].strip() for item in bodies)
        and acceptance["body_hashes_match"]
        and acceptance["pool_unchanged"]
    )
    write_json(folder / "memory-acceptance.json", acceptance)
    if not acceptance["passed"]:
        raise RuntimeError("H memory artifact acceptance failed; see memory-acceptance.json")
    # This file is the Q readiness marker and is published only after acceptance.
    write_json(
        frozen,
        {
            "fingerprint": pool,
            **evidence,
            "artifact_acceptance": {
                "passed": True,
                "contract_sha256": contract.sha256,
            },
        },
    )
