# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read PPT package members and slide order for evidence and fixture tests."""

from pathlib import Path
from typing import Any
from zipfile import ZipFile

from lxml import etree


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
