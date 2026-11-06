# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Both integrations expose one contract while Reader imports stay framework independent."""

import subprocess
import sys


def test_reader_and_assembler_export_identical_snapshot_and_exception_classes():
    from openviking.integrations.langchain.ov_context import (
        ContextHistoryBudgetError,
        ContextReadError,
        ContextSnapshot,
    )
    from openviking.integrations.ov_memoryRetrieve import types

    assert types.ContextSnapshot is ContextSnapshot
    assert types.ContextReadError is ContextReadError
    assert types.ContextHistoryBudgetError is ContextHistoryBudgetError
    assert issubclass(types.ContextHistoryBudgetError, ContextReadError)


def test_standalone_reader_does_not_require_langchain_or_langgraph():
    script = """
import importlib.abc
import sys

class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'langchain', 'langgraph'}:
            raise ModuleNotFoundError(f'Unexpected framework dependency: {fullname}')

sys.meta_path.insert(0, RejectFrameworks())
from openviking.integrations.ov_memoryRetrieve import MemoryReaderConfig, OVContextReader
assert MemoryReaderConfig().search_mode == 'context'
assert not {'langchain', 'langgraph'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
