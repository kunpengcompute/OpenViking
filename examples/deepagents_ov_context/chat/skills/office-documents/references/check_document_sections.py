#!/usr/bin/env python3
"""Check caller-specified Markdown sections; contains no business answers."""

import argparse
import json
import re
from pathlib import Path


def assess(text, sections):
    headings = [
        re.sub(r"[*`]", "", m.group(1)).strip() for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)
    ]
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text).strip()
    checks = {"readable_nonempty_body": bool(body) and "\x00" not in text}
    checks.update(
        {"section:" + name: any(name in heading for heading in headings) for name in sections}
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scope": "Structure only; business facts, sources, metrics and acceptance require separate verification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--sections", nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.path.is_symlink():
            raise ValueError("Expected a regular file")
        result = assess(args.path.read_text(encoding="utf-8"), args.sections)
    except (OSError, UnicodeError, ValueError) as exc:
        result = {"passed": False, "checks": {"readable_file": False}, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
