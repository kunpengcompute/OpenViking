# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One append-only JSONL stream for page and script-driven Chat observations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class RunEventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        self.lock = Lock()
        info = os.fstat(self.fd)
        self.identity = f"{info.st_dev}:{info.st_ino}"
        try:
            self._repair_tail()
        except BaseException:
            os.close(self.fd)
            raise

    def _repair_tail(self) -> None:
        """Keep complete records when a process exits during its last write."""
        size = self.path.stat().st_size
        if not size:
            return
        with self.path.open("rb") as stream:
            stream.seek(size - 1)
            if stream.read(1) == b"\n":
                return
            position, suffix = size, b""
            while position:
                start = max(0, position - 4096)
                stream.seek(start)
                suffix = stream.read(position - start) + suffix
                boundary = suffix.rfind(b"\n")
                if boundary >= 0:
                    break
                position = start
            else:
                boundary = -1
            tail = suffix[boundary + 1 :]
            try:
                json.loads(tail)
            except (ValueError, UnicodeError):
                os.ftruncate(self.fd, size - len(tail))
            else:
                os.write(self.fd, b"\n")

    def read_batch(self, offset: int, limit: int = 128) -> tuple[list[dict[str, Any]], int]:
        """Read complete records in a bounded batch for the SQLite projection."""
        records = []
        with self.lock, self.path.open("rb") as stream:
            stream.seek(offset)
            for _ in range(limit):
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid observation log record at offset {offset}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid observation log record at offset {offset}")
                records.append(record)
                offset = stream.tell()
        return records, offset

    def append(self, record: dict[str, Any]) -> None:
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with self.lock:
            start = os.lseek(self.fd, 0, os.SEEK_END)
            try:
                while payload:
                    written = os.write(self.fd, payload)
                    if written <= 0:
                        raise OSError("Could not append Run event log")
                    payload = payload[written:]
            except BaseException:
                os.ftruncate(self.fd, start)
                raise

    def flush(self) -> None:
        with self.lock:
            os.fsync(self.fd)

    def close(self) -> None:
        with self.lock:
            if self.fd >= 0:
                try:
                    os.fsync(self.fd)
                finally:
                    os.close(self.fd)
                    self.fd = -1
