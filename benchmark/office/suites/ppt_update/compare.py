"""Prepare H once, then run balanced native/ordinary Q pairs through Chat."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from benchmark.office.suites.common.files_config import write_json
from benchmark.office.suites.common.timing import scoring_policy
from benchmark.paths import OFFICE_ROOT

from .run import run


async def compare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    spec = json.loads(args.spec.read_text())
    files = [
        args.spec,
        OFFICE_ROOT / "cases/scoring.json",
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
            "latency_budget_seconds": spec["latency_budget_seconds"],
            "scoring_policy": scoring_policy("ppt_update"),
        },
    )
    rows = []
    write_json(args.output / "comparison.json", {"status": "running", "runs": rows})
    phases = [("H", "current", args.output / "H")]
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
                "expected_memory": args.output / "H/frozen-memory.json"
                if mode == "ordinary"
                else None,
            }
            await run(argparse.Namespace(**values))
            grade = json.loads((folder / "grade.json").read_text())
            if phase == "Q":
                origin = json.loads((args.output / "H/condition.json").read_text())["metadata"]
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
                    "task_pass": grade["task_pass"],
                    "latency": grade["latency"],
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
    parser.add_argument("--pairs", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-url", required=True)
    parser.add_argument("--chat-data", type=Path, required=True)
    parser.add_argument("--environment-id", default="daily")
    parser.add_argument("--ov-url", required=True)
    parser.add_argument("--ov-key", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--user", required=True, help="Fresh identity with an empty H memory pool")
    parser.add_argument("--peer", required=True)
    asyncio.run(compare(parser.parse_args()))


if __name__ == "__main__":
    main()
