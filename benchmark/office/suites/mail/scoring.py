"""Check three H or six Q exported bodies against the frozen references."""

import re

from benchmark.office.datasets.mail.data import goals


def check_export(path, phase, index):
    from pathlib import Path

    targets = goals(phase)
    if not 1 <= index <= len(targets):
        raise ValueError(f"mail target index must be 1 through {len(targets)}")
    target = targets[index - 1]
    path = Path(path)
    actual = path.read_text(encoding="utf-8") if path.is_file() else ""

    def normalize(text):
        return re.sub(r"\s+", "", text)

    return {
        "file_exists": path.is_file(),
        "filename_correct": path.name == target["filename"],
        "body_complete": bool(actual) and normalize(actual) == normalize(target["message"]["body"]),
    }
