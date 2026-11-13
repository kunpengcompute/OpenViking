# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Both integrations expose one contract while Reader imports stay framework independent."""

import subprocess
import sys

import pytest


def test_reader_and_assembler_export_identical_snapshot_and_exception_classes():
    from openviking.integrations.langchain.ov_context import (
        ContextHistoryBudgetError,
        ContextReadError,
        ContextSnapshot,
        reader,
    )

    assert reader.ContextSnapshot is ContextSnapshot
    assert reader.ContextReadError is ContextReadError
    assert reader.ContextHistoryBudgetError is ContextHistoryBudgetError
    assert issubclass(reader.ContextHistoryBudgetError, ContextReadError)


@pytest.mark.parametrize(
    "statement",
    [
        "from openviking.integrations.langchain.ov_context import MemoryReaderConfig, OVContextReader",
        "from openviking.integrations.langchain.ov_context.reader import MemoryReaderConfig, OVContextReader",
    ],
)
def test_standalone_reader_does_not_require_langchain_or_langgraph(statement):
    script = """
import importlib.abc
import sys

class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'langchain', 'langgraph'}:
            raise ModuleNotFoundError(f'Unexpected framework dependency: {fullname}')

sys.meta_path.insert(0, RejectFrameworks())
READER_IMPORT
assert MemoryReaderConfig().search_mode == 'context'
assert not {'langchain', 'langgraph'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script.replace("READER_IMPORT", statement)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_public_exports_remain_available_with_framework_dependencies():
    from openviking.integrations.langchain import ov_context

    namespace = {}
    exec("from openviking.integrations.langchain.ov_context import *", namespace)
    assert set(ov_context.__all__) <= namespace.keys()
    assert namespace["ContextSnapshot"] is ov_context.ContextSnapshot
    assert namespace["ContextReadError"] is ov_context.reader.ContextReadError
    assert namespace["MessageWriterBase"] is ov_context.writer.MessageWriterBase
