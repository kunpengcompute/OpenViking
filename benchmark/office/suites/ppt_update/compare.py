"""Run balanced Q pairs after a separately reviewed H preparation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from benchmark.office.suites.common.files_config import write_json
from benchmark.office.suites.common.policy import acceptance_policy
from benchmark.paths import OFFICE_ROOT

from .gates import CaseContract
from .run import run


async def compare(args):
    root = Path.cwd()
    spec = json.loads(args.spec.read_text())
    preparation = args.preparation
    frozen_memory = preparation / "frozen-memory.json"
    saved_memory = json.loads(frozen_memory.read_text())
    preparation_review = CaseContract.from_case(spec).require_reviewed(
        saved_memory, args.preparation_review
    )
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        args.spec,
        OFFICE_ROOT / "evaluation/policy.json",
        OFFICE_ROOT / "evaluation/AI_REVIEW_PROMPT.md",
        args.preparation_review,
        frozen_memory,
        preparation / "condition.json",
        root / spec["source"],
        root / spec["execution_profile"],
        root / spec["system_prompt"],
        *(root / item["source"] for item in spec["support_files"]),
        *(root / spec["phases"][p]["prompt"] for p in ["H", "Q"]),
    ]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(
        args.output / "manifest.json",
        {
            "files": frozen,
            "pairs": args.pairs,
            "preparation_review": preparation_review,
            "latency_budget_seconds": spec["latency_budget_seconds"],
            "acceptance_policy": acceptance_policy("ppt_update"),
        },
    )
    rows = []
    write_json(args.output / "comparison.json", {"status": "running", "runs": rows})
    phases = []
    for pair in range(1, args.pairs + 1):
        order = ["baseline", "ordinary"] if pair % 2 else ["ordinary", "baseline"]
        for mode in order:
            name = "Q-baseline" if mode == "baseline" else "Q-memory"
            phases.append(("Q", mode, args.output / f"pair-{pair}" / name))
    try:
        for phase, mode, folder in phases:
            if any(
                hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
                for p, digest in frozen.items()
            ):
                raise RuntimeError("PPT case assets changed during comparison")
            values = vars(args) | {
                "phase": phase,
                "mode": mode,
                "output": folder,
                "preparation_review": args.preparation_review,
                "expected_memory": frozen_memory if mode == "ordinary" else None,
            }
            await run(argparse.Namespace(**values))
            if phase == "Q":
                origin = json.loads((preparation / "condition.json").read_text())["metadata"]
                current = json.loads((folder / "condition.json").read_text())["metadata"]
                if current != origin:
                    raise RuntimeError("Chat model/Reader/Skill profile changed during comparison")
            statistics = json.loads((folder / "result.json").read_text())["statistics"]
            rows.append(
                {
                    "phase": phase,
                    "mode": mode,
                    "output": str(folder),
                    "statistics": statistics,
                    "review_status": "pending",
                    "evidence": str(folder / "evidence.json"),
                }
            )
            write_json(args.output / "comparison.json", {"status": "running", "runs": rows})
    except Exception as exc:
        write_json(
            args.output / "comparison.json",
            {
                "status": "stopped",
                "error_type": type(exc).__name__,
                "active_run": {"phase": phase, "mode": mode, "output": str(folder)},
                "runs": rows,
            },
        )
        raise
    write_json(args.output / "comparison.json", {"status": "completed", "runs": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmark/office/cases/ppt_update/spec.json")
    )
    parser.add_argument("--preparation", type=Path, required=True, help="Completed H directory")
    parser.add_argument("--preparation-review", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-url", required=True)
    parser.add_argument("--environment-id", default="daily")
    parser.add_argument("--ov-url", required=True)
    parser.add_argument("--ov-key", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--user", required=True, help="Fresh identity with an empty H memory pool")
    parser.add_argument("--peer", required=True)
    asyncio.run(compare(parser.parse_args()))


if __name__ == "__main__":
    main()
