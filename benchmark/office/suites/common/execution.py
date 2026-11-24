# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Load the versioned execution settings sent to Chat by office cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_execution_profile(path: Path) -> dict[str, Any]:
    from examples.deepagents_ov_context.chat.profiles import ExecutionProfile

    profile = ExecutionProfile.model_validate_json(path.read_text(encoding="utf-8"))
    return profile.model_dump(mode="json", exclude_unset=True)
