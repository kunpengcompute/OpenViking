# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#  SPDX-License-Identifier: AGPL-3.0
"""Bounded in-process latency samples; production hosts can supply their own sink."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class MetricsSink(Protocol):
    def __call__(self, event: str, **fields: Any) -> None: ...


@dataclass
class Metrics:
    """Structured operation samples. The sink must be synchronous and nonblocking."""

    sink: MetricsSink | None = None

    def emit(self, event: str, **fields: Any) -> None:
        import logging

        try:
            if self.sink is not None:
                self.sink(event, **fields)
            else:
                logging.getLogger(__name__).info("ov_context.%s %s", event, fields)
        except Exception as exc:
            context = {
                name: fields[name]
                for name in (
                    "account_id",
                    "user_id",
                    "actor_peer_id",
                    "session_id",
                    "run_id",
                    "model_call_id",
                )
                if isinstance(fields.get(name), str)
            }
            logging.getLogger(__name__).warning(
                "ov_context.metrics_sink_failed event=%s context=%s error_type=%s",
                event,
                context,
                type(exc).__name__,
                exc_info=True,
            )


class LatencySummary(TypedDict):
    event: str
    status: str
    count: int
    sum_ms: float
    avg_ms: float
    p90_ms: float
    max_ms: float


class LatencySamples:
    """Per-event rolling samples; p90 uses the nearest-rank definition.

    Store only durations, never prompts, URLs, credentials or response bodies.
    Nested/concurrent event sums must not be interpreted as wall-clock latency.
    """

    def __init__(self, max_samples: int = 10000) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self._samples: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def __call__(self, event: str, **fields: Any) -> None:
        duration = fields.get("duration_ms")
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            self._samples[(event, str(fields.get("status", "ok")))].append(float(duration))

    def summary(self) -> list[LatencySummary]:
        rows: list[LatencySummary] = []
        for (event, status), samples in sorted(self._samples.items()):
            values = sorted(samples)
            total = sum(values)
            rows.append(
                {
                    "event": event,
                    "status": status,
                    "count": len(values),
                    "sum_ms": total,
                    "avg_ms": total / len(values),
                    "p90_ms": values[math.ceil(0.9 * len(values)) - 1],
                    "max_ms": values[-1],
                }
            )
        return rows
