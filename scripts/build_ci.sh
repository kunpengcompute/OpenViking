#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
# 在 CI 镜像内执行：bash scripts/build_ci.sh
# OV_SOURCE_DIR：待编译源码；UV_PROJECT_ENVIRONMENT：虚拟环境；REPORT_DIR：产物目录。
# OV_SKIP_UV_SYNC=1：复用流水线已同步的依赖；CI_BUILD_JOBS：原生编译并行度。
set -euo pipefail

SOURCE_DIR="$(cd "${OV_SOURCE_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
cd "$SOURCE_DIR"
[[ -f setup.py && -f pyproject.toml && -f Cargo.toml ]] || {
    echo 'ERROR: OV_SOURCE_DIR 必须指向完整的 OpenViking 源码目录' >&2; exit 2;
}
REPORT_DIR="${REPORT_DIR:-$SOURCE_DIR/build/ci-reports}"
mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
exec > >(tee "$REPORT_DIR/build.log") 2>&1
trap 'status=$?; echo "ERROR: 编译步骤失败，行号 $LINENO，退出码 $status"; exit "$status"' ERR

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    if [[ -x /opt/ov-venv/bin/python ]]; then
        export UV_PROJECT_ENVIRONMENT=/opt/ov-venv
    else
        export UV_PROJECT_ENVIRONMENT="$SOURCE_DIR/.venv"
    fi
fi
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export LITELLM_LOCAL_MODEL_COST_MAP=true UV_LINK_MODE=copy UV_NO_PROGRESS=1
export OV_SKIP_STUDIO_BUILD="${OV_SKIP_STUDIO_BUILD:-1}"
export OV_REQUIRE_RAGFS_BUILD=1
# 编译门禁从当前源码构建三个原生产物，增量缓存由构建工具管理。
unset OV_SKIP_OV_BUILD OV_SKIP_RAGFS_BUILD OV_PREBUILT_BIN_DIR
export CMAKE_BUILD_PARALLEL_LEVEL="${CI_BUILD_JOBS:-4}"
export CARGO_BUILD_JOBS="$CMAKE_BUILD_PARALLEL_LEVEL"
[[ "$CMAKE_BUILD_PARALLEL_LEVEL" =~ ^[1-9][0-9]*$ ]] || {
    echo 'ERROR: CI_BUILD_JOBS 必须为正整数' >&2; exit 2;
}
if [[ -n "${OPENVIKING_VERSION:-}" ]]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING="$OPENVIKING_VERSION"
fi

for tool in uv cmake gcc g++ cargo rustc; do
    command -v "$tool" >/dev/null || { echo "ERROR: 缺少工具 $tool" >&2; exit 2; }
done
uv --version
cmake --version
rustc --version
cargo --version
if [[ "${OV_SKIP_UV_SYNC:-0}" != 1 ]]; then
    uv sync --frozen --inexact --no-install-project --extra test --extra gemini --extra langgraph
fi
[[ -x "$PYTHON" ]] || { echo "ERROR: Python 虚拟环境不存在：$PYTHON" >&2; exit 2; }
"$PYTHON" - <<'PY'
import importlib.util
import sys
import tomllib

assert sys.version_info >= (3, 11), 'CI 镜像要求 Python >= 3.11'
required = ('setuptools', 'setuptools_scm', 'wheel', 'maturin')
missing = [name for name in required if importlib.util.find_spec(name) is None]
assert not missing, f'镜像缺少构建后端依赖：{missing}'
print('Python:', sys.executable, sys.version.split()[0])
print('Build requirements:', tomllib.load(open('pyproject.toml', 'rb'))['build-system']['requires'])
PY

# editable 构建生成与源码对应的 Rust CLI、RAGFS 和 VectorDB 扩展。
uv pip install --python "$PYTHON" --no-deps --no-build-isolation --reinstall --editable .
uv build --python "$PYTHON" --wheel --no-build-isolation --out-dir "$REPORT_DIR/wheels"
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$SOURCE_DIR" "$REPORT_DIR" <<'PY'
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import openviking
from openviking.pyagfs import get_binding_client
from openviking.storage.vectordb import engine

root, reports = map(Path, sys.argv[1:])
assert Path(openviking.__file__).resolve().parent == root / 'openviking'
assert engine.ENGINE_VARIANT != 'unavailable', engine.AVAILABLE_ENGINE_VARIANTS
get_binding_client()
cli = root / 'openviking/bin/ov'
subprocess.run([str(cli), 'language', '--help'], check=True)
wheels = sorted((reports / 'wheels').glob('openviking-*.whl'), key=lambda p: p.stat().st_mtime)
assert wheels, '未生成 OpenViking wheel'
with zipfile.ZipFile(wheels[-1]) as wheel:
    names = wheel.namelist()
    assert any(n.endswith('/bin/ov') for n in names), 'wheel 缺少 Rust CLI'
    assert any('/lib/ragfs_python' in n and n.endswith('.so') for n in names), 'wheel 缺少 RAGFS'
    assert any('/vectordb/engine/_' in n and n.endswith('.so') for n in names), 'wheel 缺少 VectorDB'
summary = {'source': str(root), 'python': sys.version.split()[0],
           'engine_variant': engine.ENGINE_VARIANT, 'wheel': str(wheels[-1]),
           'native_checks': ['ov language --help', 'RAGFS binding', 'VectorDB import']}
(reports / 'build-result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
echo "编译验证通过：$REPORT_DIR"
