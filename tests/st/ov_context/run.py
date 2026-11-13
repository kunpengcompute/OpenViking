# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run OVContext integration against a deployed trusted OV and real model APIs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmark.paths import PROJECT_ROOT, RESULT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ov-url", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "st/ov_context")
    parser.add_argument(
        "--select", help="Optional pytest -k expression, e.g. writer or compression"
    )
    parser.add_argument(
        "--parallelism", type=int, default=4, help="Component concurrent sessions, 1..32"
    )
    parser.add_argument(
        "--case-timeout", type=float, default=300, help="Component phase timeout in seconds"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Enable labelled OV SIGKILL, repeated capacity and sustained load tests",
    )
    parser.add_argument(
        "--server-container",
        help="Isolated ov-middleware-integration-* container with ov.live-test=true",
    )
    parser.add_argument("--stress-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.extended and (
        not args.server_container
        or not args.server_container.startswith("ov-middleware-integration-")
    ):
        parser.error("--extended requires an isolated --server-container")
    if not 60 <= args.stress_seconds <= 3600:
        parser.error("--stress-seconds must be from 60 to 3600")
    if not 1 <= args.parallelism <= 32:
        parser.error("--parallelism must be from 1 to 32")
    if not math.isfinite(args.case_timeout) or args.case_timeout <= 0:
        parser.error("--case-timeout must be finite and positive")
    repo = PROJECT_ROOT
    output = args.output_dir.resolve()
    if output.is_relative_to(repo) and not output.is_relative_to(RESULT_ROOT.resolve()):
        parser.error("Store live service artifacts under result/ or outside the repository")
    for path in (args.api_key_file, args.model_config):
        if not path.is_file():
            parser.error(f"Configuration file missing: {path}")
    directory = output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    env = dict(
        os.environ,
        OV_LIVE_ENABLED="1",
        OV_LIVE_URL=args.ov_url,
        OV_LIVE_API_KEY_FILE=str(args.api_key_file.resolve()),
        OV_LIVE_MODEL_CONFIG=str(args.model_config.resolve()),
        OV_LIVE_COMPONENT_PARALLELISM=str(args.parallelism),
        OV_LIVE_COMPONENT_TIMEOUT=str(args.case_timeout),
    )
    if args.extended:
        env.update(
            OV_LIVE_EXTENDED="1",
            OV_LIVE_CONTAINER=args.server_container,
            OV_LIVE_STRESS_SECONDS=str(args.stress_seconds),
        )
    else:
        for name in ("OV_LIVE_EXTENDED", "OV_LIVE_CONTAINER", "OV_LIVE_STRESS_SECONDS"):
            env.pop(name, None)
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/st/ov_context/live",
        "-q",
        "-o",
        "addopts=",
        "--tb=short",
        f"--basetemp={directory / 'cases'}",
        f"--junitxml={directory / 'junit.xml'}",
    ]
    if args.select:
        command += ["-k", args.select]
    (directory / "environment.json").write_text(
        json.dumps(
            {
                "ov_url": args.ov_url,
                "platform": platform.platform(),
                "python": sys.version,
                "selection": args.select,
                "component_parallelism": args.parallelism,
                "component_timeout": args.case_timeout,
                "extended": args.extended,
                "stress_seconds": args.stress_seconds if args.extended else None,
                "model": json.loads(args.model_config.read_text())["chat_model"],
                "scope": "deployed OV, real model, Chat HTTP/SSE, SQLite and real TCP faults",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Live evidence: {directory}", flush=True)
    with (directory / "pytest.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
