"""Workspace-scoped SQLite connection shared by login and office cases."""

import sqlite3
from pathlib import Path

from ..schedule.data import NAMESPACE


def connect(registry, namespace):
    if not NAMESPACE.fullmatch(namespace):
        raise ValueError("Invalid namespace")
    folder = Path(registry) / "office-state"
    folder.mkdir(exist_ok=True, mode=0o700)
    db = sqlite3.connect(folder / (namespace + ".sqlite3"), timeout=5)
    db.row_factory = sqlite3.Row
    return db
