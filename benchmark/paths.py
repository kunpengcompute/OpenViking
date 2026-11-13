# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared checkout and result locations for benchmark command-line entry points."""

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parent
OFFICE_ROOT = BENCHMARK_ROOT / "office"
RESULT_ROOT = PROJECT_ROOT / "result"
