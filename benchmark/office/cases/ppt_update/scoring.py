"""Check the actual PPT package against a minimal-change oracle, independently."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from lxml import etree

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def xml(raw: bytes) -> Any:
    return etree.fromstring(raw, parser=etree.XMLParser(resolve_entities=False, no_network=True))


def slide_parts(package: dict[str, bytes]) -> list[str]:
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
    rels = {r.get("Id"): r.get("Target") for r in xml(package["ppt/_rels/presentation.xml.rels"])}
    ids = xml(package["ppt/presentation.xml"]).findall("p:sldIdLst/p:sldId", ns)
    return [
        (target.lstrip("/") if target.startswith("/") else "ppt/" + target)
        for item in ids
        for target in [
            rels[
                item.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            ]
        ]
    ]


def package_bytes(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        if archive.testzip() is not None or len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError("Corrupt ZIP or duplicate members")
        return {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}


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


def structure(node: Any) -> Any:
    """Compare expanded names, attributes and content, ignoring serializer-only whitespace."""
    if not isinstance(node.tag, str):
        return None
    text = node.text or ""
    if node.tag != f"{{{NS['a']}}}t" and not text.strip():
        text = ""
    # Modification timestamp and revision counters are package bookkeeping.
    if node.tag in {
        "{http://purl.org/dc/terms/}modified",
        "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}revision",
    }:
        text = ""
    children = [structure(child) for child in node if isinstance(child.tag, str)]
    if node.tag in {
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships",
        "{http://schemas.openxmlformats.org/package/2006/content-types}Types",
    }:
        children.sort(key=repr)
    attributes = dict(node.attrib)
    if node.tag == "{http://schemas.openxmlformats.org/package/2006/content-types}Default":
        # OPC content-type extension matching is case-insensitive (GIF -> gif on save).
        attributes["Extension"] = attributes["Extension"].lower()
    return node.tag, tuple(sorted(attributes.items())), text, tuple(children)


def check_deck(source: Path, output: Path, case: dict[str, Any]) -> dict[str, Any]:
    original = package_bytes(source)
    actual = package_bytes(output)
    expected = expected_package(original, case)
    parts = slide_parts(actual)
    changed = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            changed.append(name)
        elif name.endswith((".xml", ".rels")):
            if structure(xml(expected[name])) != structure(xml(actual[name])):
                changed.append(name)
        elif expected[name] != actual[name]:
            changed.append(name)
    page_texts = [
        "\n".join(
            "".join(t.text or "" for t in p.findall(".//a:t", NS))
            for p in xml(actual[name]).findall(".//a:p", NS)
        )
        for name in parts
    ]
    edits = (
        [
            edit["anchor"].replace(edit["old"], edit["new"], 1) in page_texts[edit["slide"] - 1]
            and edit["anchor"] not in page_texts[edit["slide"] - 1]
            for edit in case["edits"]
        ]
        if len(parts) == case["slides"]
        else [False] * len(case["edits"])
    )
    preserved = [
        item["slide"] <= len(parts) and item["text"] in page_texts[item["slide"] - 1]
        for item in case["preserve"]
    ]
    return {
        "slide_count": len(parts),
        "edits": edits,
        "preserved": preserved,
        "unexpected_parts": changed,
        "page_texts": page_texts,
    }
