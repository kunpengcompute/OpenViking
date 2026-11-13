"""Build limits must survive container/host CPU-count differences."""

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
from setuptools import Distribution, Extension


@pytest.mark.parametrize("limit", [None, "8"])
def test_cmake_build_honors_explicit_parallel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str | None
) -> None:
    root = Path(__file__).resolve().parents[2]
    with patch("setuptools.setup"):
        scope = runpy.run_path(str(root / "setup.py"))
    monkeypatch.setattr(scope["os"], "cpu_count", lambda: 320)
    if limit is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", limit)

    command = scope["OpenVikingBuildExt"](Distribution())
    command.extensions = [Extension("openviking.storage.vectordb.engine._native", sources=[])]
    command.cmake_executable = "cmake"
    invocations: list[list[str]] = []
    command.spawn = invocations.append
    command._build_extension_impl(tmp_path / "_native.abi3.so", tmp_path, tmp_path / "build")
    build = invocations[-1]
    assert build[:2] == ["cmake", "--build"]
    assert build[build.index("--parallel") + 1] == (limit or "320")
