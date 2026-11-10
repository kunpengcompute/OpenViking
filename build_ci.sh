#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bundle_dir="$repo_root/patches/OpenViking_V0.4.1"
manifest_value() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$bundle_dir/manifest.json" "$1"
}

echo "[1/3] 校验补丁清单和 SHA256"
python3 "$bundle_dir/patch_manager.py" verify

echo "[2/3] 获取固定开源基线"
base_ref="$(manifest_value base_ref)"
verification_root="$(mktemp -d "${TMPDIR:-/tmp}/ov-patch-ci.XXXXXX")"
trap 'rm -rf -- "$verification_root"' EXIT
source_tree="$verification_root/baseline"
git init --quiet "$source_tree"
if [[ -n "${OV_PATCH_SOURCE_TREE:-}" ]]; then
    # OV_PATCH_SOURCE_TREE=/path/to/ov-source 可复用本地对象进行离线验证。
    source_cache="$(cd -- "$OV_PATCH_SOURCE_TREE" && pwd)"
    git -C "$source_tree" fetch --quiet --depth=1 --no-tags -- "$source_cache" "$base_ref"
else
    git -C "$source_tree" fetch --quiet --depth=1 --no-tags -- \
        "$(manifest_value base_repository)" "refs/tags/$(manifest_value base_tag)"
fi
if [[ "$(git -C "$source_tree" rev-parse 'FETCH_HEAD^{commit}')" != "$base_ref" ]]; then
    echo "error: Upstream tag moved away from the pinned baseline" >&2
    exit 1
fi
git -C "$source_tree" checkout --quiet --detach "$base_ref"

echo "[3/3] 在干净基线上应用各 Profile 补丁并核对源码 tree"
# 每个 Profile 执行一次 git apply --index --check、git apply --index 和 tree 核对。
python3 "$bundle_dir/patch_manager.py" verify --source-tree "$source_tree"
echo "OpenViking 0.4.1 补丁应用验证通过"
