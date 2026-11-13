"""Measure captured browser snapshots without treating fragments as full pages."""

import argparse
import json
import re
from pathlib import Path

from benchmark.paths import RESULT_ROOT


def measure(text, formatter):
    lines = text.splitlines()
    if formatter == "cua-ax":
        node = r"^\s*\d+\s+\S"
        interactive = (
            r"^\s*\d+\s+(?:button|link|combo box|search text field|radio button|text field)\b"
        )
        indent = [len(re.match(r"\s*", line).group().expandtabs(2)) for line in lines]
    else:
        node = r"^\s*-\s+\S"
        interactive = r"\[ref=e\d+\]"
        indent = [len(re.match(r" *", line).group()) for line in lines]
    return {
        "chars": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "lines": len(lines),
        "nodes": len(re.findall(node, text, re.M)),
        "interactive_references": len(re.findall(interactive, text, re.M)),
        "max_indent_columns": max(indent, default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--formatter", choices=["cua-ax", "agent-browser"], required=True)
    parser.add_argument("--mode", choices=["full", "interactive"], required=True)
    parser.add_argument("--browser-version", required=True)
    parser.add_argument(
        "--stage", required=True, help="Include state, e.g. workbench-search-timesheet"
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "office/snapshots/metrics.json"
    )
    args = parser.parse_args()
    result = {
        "formatter": args.formatter,
        "snapshot_mode": args.mode,
        "browser_version": args.browser_version,
        "stage": args.stage,
        "partial": args.partial,
        "metrics": measure(args.snapshot.read_text(), args.formatter),
        "tokens": None,
        "token_note": "Requires the actual model tokenizer; no character-based estimate.",
    }
    if args.partial:
        result["full_page_metrics"] = None
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
