#!/usr/bin/env bash
# 门禁 UT 执行脚本：按 UT_SCOPE 分发用例集，并产出 junit / coverage 报告
#
# 用法：
#   bash .gitcode/scripts/ut_ci.sh
# 可用环境变量：
#   UT_SCOPE=smoke|gate|full   smoke=快速冒烟；gate=单元测试+冒烟（默认）；full=全量 tests/
#   REPORT_DIR=...             报告输出目录（默认 build/ut-reports）
set -euo pipefail
echo "ut 执行脚本"
exit 1