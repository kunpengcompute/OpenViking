#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
# 在 CI 镜像内执行：bash ut_ci.sh [pytest 用例路径及参数]
# UT_SCOPE=unit|gate|full；OV_SKIP_BUILD=1 复用本作业已验证的编译结果。
# unit/gate 只覆盖记忆读写路径优化点的回归用例；full 才跑全量。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "${OV_SOURCE_DIR:-$SCRIPT_DIR}" && pwd)"
cd "$SOURCE_DIR"
[[ -f pyproject.toml && -d tests ]] || {
    echo 'ERROR: OV_SOURCE_DIR 必须指向完整的 OpenViking 源码目录' >&2; exit 2;
}
export REPORT_DIR="${REPORT_DIR:-$SOURCE_DIR/build/ci-reports}"
mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
export REPORT_DIR
exec > >(tee "$REPORT_DIR/ut.log") 2>&1
trap 'status=$?; echo "ERROR: UT 步骤失败，行号 $LINENO，退出码 $status"; exit "$status"' ERR

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    if [[ -x /opt/ov-venv/bin/python ]]; then
        export UV_PROJECT_ENVIRONMENT=/opt/ov-venv
    else
        export UV_PROJECT_ENVIRONMENT="$SOURCE_DIR/.venv"
    fi
fi
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export LITELLM_LOCAL_MODEL_COST_MAP=true
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${OV_SKIP_BUILD:-0}" != 1 ]]; then
    OV_SOURCE_DIR="$SOURCE_DIR" bash "$SCRIPT_DIR/build_ci.sh"
fi
[[ -x "$PYTHON" ]] || { echo "ERROR: Python 虚拟环境不存在：$PYTHON" >&2; exit 2; }
"$PYTHON" - <<'PY'
import importlib

from openviking.pyagfs import get_binding_client
from openviking.storage.vectordb import engine

for name in ('pytest', 'pytest_asyncio', 'pytest_cov'):
    importlib.import_module(name)
assert engine.ENGINE_VARIANT != 'unavailable', '请先执行 build_ci.sh 编译原生扩展'
get_binding_client()
print('UT native engine:', engine.ENGINE_VARIANT)
PY

# 记忆读写路径优化点回归用例。这些文件随优化点代码一并合入，缺失时回退到
# tests/unit，避免仅合入 CI 脚本时因文件不存在把流水线弄红。
OPT_TARGETS=(
    tests/session/test_extract_prompt_schema_cache.py
    tests/storage/test_acl_enabled_cache.py
    tests/storage/test_viking_fs_read_cache.py
)

if (( $# )); then
    targets=("$@")
else
    case "${UT_SCOPE:-unit}" in
        unit|gate)
            targets=()
            for path in "${OPT_TARGETS[@]}"; do
                [[ ! -e "$path" ]] || targets+=("$path")
            done
            (( ${#targets[@]} )) || {
                echo "WARN: 优化点用例不存在，回退 tests/unit"
                targets=(tests/unit)
            }
            ;;
        full)
            targets=(tests)
            ;;
        *) echo 'ERROR: UT_SCOPE 必须为 unit、gate 或 full' >&2; exit 2 ;;
    esac
fi
workers="${UT_WORKERS:-1}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: UT_WORKERS 必须为正整数' >&2; exit 2; }
pytest_args=(-o addopts='' -q)
if (( workers > 1 )); then
    pytest_args+=(-n "$workers" --dist=loadscope)
fi
"$PYTHON" -m pytest "${pytest_args[@]}" \
    --junitxml="$REPORT_DIR/junit.xml" \
    --cov=openviking --cov=openviking_cli \
    --cov-report="xml:$REPORT_DIR/coverage.xml" \
    --cov-report="html:$REPORT_DIR/htmlcov" --cov-report=term:skip-covered \
    "${targets[@]}"
echo "UT 验证通过：$REPORT_DIR/junit.xml"
