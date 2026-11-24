# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Create deterministic PPT revisions for evidence-capture tests."""

import copy
from typing import Any

from lxml import etree

from benchmark.office.cases.ppt_update.package import slide_parts, xml

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def expected_package(source: dict[str, bytes], case: dict[str, Any]) -> dict[str, bytes]:
    """Locate edits from task anchors; never let an output define its own expected values."""
    result = copy.copy(source)
    parts = slide_parts(source)
    for edit in case["edits"]:
        name = parts[edit["slide"] - 1]
        tree = xml(result[name])
        paragraphs = [
            p for p in tree.findall(".//a:p", NS) if edit["anchor"] in "".join(p.itertext())
        ]
        if len(paragraphs) != 1:
            raise ValueError("Case anchor must identify exactly one paragraph")
        texts = paragraphs[0].findall(".//a:t", NS)
        full = "".join(t.text or "" for t in texts)
        if full.count(edit["anchor"]) != 1 or edit["anchor"].count(edit["old"]) != 1:
            raise ValueError("Case edit is ambiguous")
        start = full.index(edit["anchor"]) + edit["anchor"].index(edit["old"])
        end = start + len(edit["old"])
        offset = 0
        inserted = False
        for text in texts:
            value = text.text or ""
            stop = offset + len(value)
            if offset < end and stop > start:
                left, right = max(0, start - offset), min(len(value), end - offset)
                text.text = value[:left] + (edit["new"] if not inserted else "") + value[right:]
                inserted = True
            offset = stop
        result[name] = etree.tostring(tree)
    return result
