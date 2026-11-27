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

# 基础用例取上游 lite CI 的 vectordb 子集：不依赖外部服务，构建完原生扩展即可跑。
BASE_TARGETS=(
    tests/vectordb/test_cuvs_config.py
    tests/vectordb/test_cuvs_index.py
    tests/vectordb/test_cuvs_collection.py
    tests/vectordb/test_str_to_uint64.py
)
# 记忆读写路径优化点回归用例。这些文件随优化点 PR 一并合入，存在时追加执行；
# 仅合入本 CI 脚本时它们尚不存在，不影响门禁。
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
            for path in "${BASE_TARGETS[@]}" "${OPT_TARGETS[@]}"; do
                [[ ! -e "$path" ]] || targets+=("$path")
            done
            (( ${#targets[@]} )) || {
                echo 'ERROR: 未找到可执行的用例文件' >&2; exit 2;
            }
            if [[ "${UT_SCOPE:-unit}" == gate ]]; then
                # 不含 langchain_langgraph/test_smoke：langchain_openviking 是
                # examples/langchain 下的独立包，本作业未安装；pytest.importorskip
                # 会让整个模块被跳过，门禁会假绿。
                for path in tests/integration/ov_context tests/client/test_http_request_identity.py \
                            benchmark/office/tests; do
                    [[ ! -e "$path" ]] || targets+=("$path")
                done
            fi
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
