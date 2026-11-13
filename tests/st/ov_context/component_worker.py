"""Internal subprocess for the ST checkpoint restart test."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .components import execute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--ov-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--worker-phase", required=True, choices=("pause", "resume"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--source", default="working-tree")
    args = parser.parse_args(argv)
    args.command = "worker"
    args.parallelism = 1
    os.umask(0o077)
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
