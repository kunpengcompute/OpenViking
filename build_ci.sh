#!/usr/bin/env bash
# 门禁编译脚本：校验工具链 -> 安装依赖 -> 编译 C++ 扩展 / Rust ragfs 绑定 -> 冒烟验证
#
# 用法：
#   bash .gitcode/scripts/build_ci.sh
# 可用环境变量：
#   PYTHON=python3.10            指定解释器
#   OV_SKIP_STUDIO_BUILD=0       同时编译 web-studio（需要 npm）
#   OV_REQUIRE_RAGFS_BUILD=0     本地调试时允许跳过 ragfs 原生库
set -euo pipefail

echo "请补全编译脚本"
exit 1