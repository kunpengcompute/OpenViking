# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Durable, session-scoped Run observations for the local Chat debugger."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from typing import Any, Callable, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from examples.deepagents_ov_context.chat.event_log import RunEventLog

EventKind = Literal[
    "start", "stage", "model_input", "statistics", "token", "message", "done", "error", "cancelled"
]


class RunStatistics(BaseModel):
    """Provider usage is counted once per completed model call."""

    run_id: str
    status: Literal["running", "done", "error", "cancelled", "interrupted"] = "running"
    first_token_seconds: float | None = None
    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model_calls: int = 0
    usage_calls: int = 0
    observations_complete: bool = True


class RunSummary(BaseModel):
    run_id: str
    created_at: float
    prompt: str
    statistics: RunStatistics


class RunEvent(BaseModel):
    ordinal: int
    event: EventKind
    data: dict[str, Any]
    elapsed_ms: int


class RunTrace(RunSummary):
    events: list[RunEvent]


class RunHistory:
    """Use the session directory's connection; checkpoints remain independent."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db
        self.journal: RunEventLog | None = None
        self._offset = 0
        self._logged_events = 0
        self._identity_reader: Callable[[str], tuple[str, str, str] | None] | None = None
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_runs (
                session_id TEXT NOT NULL, run_id TEXT NOT NULL,
                created_at REAL NOT NULL, prompt TEXT NOT NULL, statistics TEXT NOT NULL,
                PRIMARY KEY (session_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS chat_run_events (
                session_id TEXT NOT NULL, run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, event TEXT NOT NULL, data TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                PRIMARY KEY (session_id, run_id, ordinal),
                FOREIGN KEY (session_id, run_id) REFERENCES chat_runs(session_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS chat_journal_offsets (
                journal_id TEXT PRIMARY KEY, offset INTEGER NOT NULL
            );
            """
        )

    def bind_log(
        self,
        log: RunEventLog,
        identity_reader: Callable[[str], tuple[str, str, str] | None] | None = None,
    ) -> None:
        """Recover the journal suffix before accepting new observations."""
        self.journal = log
        self._identity_reader = identity_reader
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO chat_journal_offsets VALUES (?, 0)", (log.identity,)
            )
            self._offset = self.db.execute(
                "SELECT offset FROM chat_journal_offsets WHERE journal_id = ?", (log.identity,)
            ).fetchone()[0]
            if self._offset > log.path.stat().st_size:
                self._offset = 0
                self.db.execute(
                    "UPDATE chat_journal_offsets SET offset = 0 WHERE journal_id = ?",
                    (log.identity,),
                )
        self.flush()

    def record_logged_event(self, *, final: bool = False) -> None:
        """Keep JSONL as the write-ahead source; commit small SQLite batches."""
        self._logged_events += 1
        if final or self._logged_events >= 10:
            self.flush()

    def flush(self) -> None:
        log = self.journal
        if log is None:
            return
        while True:
            records, offset = log.read_batch(self._offset)
            if not records:
                self._logged_events = 0
                return
            rows = []
            runs: dict[tuple[str, str], Any] = {}
            owners: dict[str, tuple[str, str, str] | None] = {}
            for record in records:
                sid, rid = record["session_id"], record["run_id"]
                key = sid, rid
                if key not in runs:
                    runs[key] = self.db.execute(
                        "SELECT created_at FROM chat_runs WHERE session_id = ? AND run_id = ?",
                        key,
                    ).fetchone()
                if runs[key] is None:
                    # Deleted/imported histories retain their application ownership.
                    continue
                if self._identity_reader is not None:
                    if sid not in owners:
                        owners[sid] = self._identity_reader(sid)
                    if owners[sid] != tuple(
                        record.get(k) for k in ("account_id", "user_id", "actor_peer_id")
                    ):
                        raise ValueError("Observation identity does not match its session")
                kind = record.get("event_kind")
                if kind is None:
                    kind = {
                        "run.start": "start",
                        "run.statistics": "statistics",
                        "message.full": "message",
                        "model_input": "model_input",
                    }.get(record["event"], "stage")
                    if record["event"] == "run.end":
                        kind = record["data"].get("statistics", {}).get("status", "done")
                elapsed = record.get("elapsed_ms")
                if elapsed is None:
                    elapsed = max(
                        0,
                        round(
                            (datetime.fromisoformat(record["timestamp"]).timestamp() - runs[key][0])
                            * 1000
                        ),
                    )
                event = RunEvent(
                    ordinal=record["sequence"], event=kind, data=record["data"], elapsed_ms=elapsed
                )
                rows.append(
                    (
                        sid,
                        rid,
                        event.ordinal,
                        event.event,
                        json.dumps(event.data, ensure_ascii=False, allow_nan=False),
                        event.elapsed_ms,
                    )
                )
            with self.db:
                inserted = self.db.executemany(
                    "INSERT INTO chat_run_events VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, run_id, ordinal) DO NOTHING",
                    rows,
                ).rowcount
                if inserted < len(rows):
                    for sid, rid, ordinal, kind, data, _ in rows:
                        saved = self.db.execute(
                            "SELECT event, data FROM chat_run_events "
                            "WHERE session_id = ? AND run_id = ? AND ordinal = ?",
                            (sid, rid, ordinal),
                        ).fetchone()
                        if saved[0] != kind or json.loads(saved[1]) != json.loads(data):
                            raise ValueError("Conflicting observation sequence in journal")
                self.db.execute(
                    "UPDATE chat_journal_offsets SET offset = MAX(offset, ?) WHERE journal_id = ?",
                    (offset, log.identity),
                )
            self._offset = offset

    def begin(self, session_id: str, run_id: str, prompt: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO chat_runs VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    run_id,
                    time.time(),
                    prompt,
                    RunStatistics(run_id=run_id).model_dump_json(),
                ),
            )

    def update(self, session_id: str, statistics: RunStatistics) -> None:
        # The caller commits this together with SessionInfo.last_run.
        data = statistics.model_dump_json()
        self.db.execute(
            "UPDATE chat_runs SET statistics = ? "
            "WHERE session_id = ? AND run_id = ? AND statistics != ?",
            (data, session_id, statistics.run_id, data),
        )

    def interrupt_running(self) -> set[str]:
        """Keep already committed observations after an unclean process exit."""
        interrupted: set[str] = set()
        with self.db:
            for session_id, value in self.db.execute(
                "SELECT session_id, statistics FROM chat_runs"
            ).fetchall():
                statistics = RunStatistics.model_validate_json(value)
                if statistics.status == "running":
                    statistics.status = "interrupted"
                    self.update(session_id, statistics)
                    interrupted.add(session_id)
        return interrupted

    def append(self, session_id: str, run_id: str, event: RunEvent) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO chat_run_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    run_id,
                    event.ordinal,
                    event.event,
                    json.dumps(event.data, ensure_ascii=False, allow_nan=False),
                    event.elapsed_ms,
                ),
            )

    def list(self, session_id: str) -> list[RunSummary]:
        self.flush()
        rows = self.db.execute(
            "SELECT run_id, created_at, prompt, statistics FROM chat_runs "
            "WHERE session_id = ? ORDER BY created_at DESC, rowid DESC",
            (session_id,),
        ).fetchall()
        return [self._summary(row) for row in rows]

    def get(self, session_id: str, run_id: str) -> RunTrace:
        self.flush()
        row = self.db.execute(
            "SELECT run_id, created_at, prompt, statistics FROM chat_runs "
            "WHERE session_id = ? AND run_id = ?",
            (session_id, run_id),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "本会话没有该 Run 的调试记录")
        events = self.db.execute(
            "SELECT ordinal, event, data, elapsed_ms FROM chat_run_events "
            "WHERE session_id = ? AND run_id = ? ORDER BY ordinal",
            (session_id, run_id),
        ).fetchall()
        return RunTrace(
            **self._summary(row).model_dump(),
            events=[
                RunEvent(ordinal=n, event=kind, data=json.loads(data), elapsed_ms=elapsed)
                for n, kind, data, elapsed in events
            ],
        )

    @staticmethod
    def _summary(row: tuple[str, float, str, str]) -> RunSummary:
        run_id, created_at, prompt, statistics = row
        return RunSummary(
            run_id=run_id,
            created_at=created_at,
            prompt=prompt,
            statistics=RunStatistics.model_validate_json(statistics),
        )
