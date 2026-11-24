# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Load declarative review budgets; no task success is inferred here."""

import json
import math

from benchmark.paths import OFFICE_ROOT


def acceptance_policy(case: str) -> dict:
    policy = json.loads((OFFICE_ROOT / "evaluation/policy.json").read_text())
    return {**policy, "case": case, "budget_seconds": policy["budget_seconds"][case]}


def valid_seconds(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )
