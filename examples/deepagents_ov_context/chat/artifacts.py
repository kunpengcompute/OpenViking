# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session-scoped download metadata, independent of artifact generation."""

import re
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel

ArtifactKind: TypeAlias = Literal[
    "presentation", "document", "spreadsheet", "pdf", "data", "text", "image"
]
ARTIFACT_TYPES: dict[str, tuple[ArtifactKind, str]] = {
    ".pptx": (
        "presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".docx": (
        "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".xlsx": ("spreadsheet", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".pdf": ("pdf", "application/pdf"),
    ".csv": ("data", "text/csv; charset=utf-8"),
    ".json": ("data", "application/json"),
    ".md": ("text", "text/markdown; charset=utf-8"),
    ".txt": ("text", "text/plain; charset=utf-8"),
    ".png": ("image", "image/png"),
}


class FileArtifact(BaseModel):
    filename: str
    run_id: str
    url: str
    slides: int | None = None
    bytes: int
    kind: ArtifactKind = "presentation"


def valid_artifact_name(filename: str) -> bool:
    path = Path(filename)
    return (
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\.[a-z0-9]+", filename) is not None
        and path.suffix in ARTIFACT_TYPES
    )


def valid_input_name(filename: str) -> bool:
    """Accept document inputs and Python helpers as session-scoped source files."""
    return valid_artifact_name(filename) or (
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\.py", filename) is not None
    )


def artifact_directory(root: Path, session_id: str, run_id: str) -> Path:
    for value in (session_id, run_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
            raise ValueError("Invalid artifact owner")
    result = root / session_id / run_id
    if any(path.is_symlink() for path in (root, root / session_id, result)):
        raise ValueError("Artifact directories cannot be symlinks")
    return result


def list_artifacts(root: Path, session_id: str) -> list[FileArtifact]:
    parent = artifact_directory(root, session_id, "list").parent
    if not parent.is_dir():
        return []
    result: list[FileArtifact] = []
    for manifest in sorted(parent.glob("*/*.artifact.json")):
        folder = artifact_directory(root, session_id, manifest.parent.name)
        if manifest.is_symlink():
            continue
        try:
            item = FileArtifact.model_validate_json(manifest.read_text())
        except (ValueError, OSError):
            continue
        path = folder / item.filename
        if (
            item.run_id == folder.name
            and valid_artifact_name(item.filename)
            and item.kind == ARTIFACT_TYPES[path.suffix][0]
            and path.is_file()
            and not path.is_symlink()
        ):
            # Rebuild the route from the trusted owner rather than a stored URL.
            item.url = f"/api/sessions/{session_id}/artifacts/{item.run_id}/{item.filename}"
            result.append(item)
    return result
